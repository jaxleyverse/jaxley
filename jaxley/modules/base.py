# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, vmap
from jax.lax import ScatterDimensionNumbers, scatter_add
from matplotlib.axes import Axes

from jaxley.channels import Channel
from jaxley.solver_voltage import (
    step_voltage_explicit,
    step_voltage_implicit_with_jax_spsolve,
    step_voltage_implicit_with_jaxley_spsolve,
)
from jaxley.synapses import Synapse
from jaxley.utils.cell_utils import (
    _compute_index_of_child,
    _compute_num_children,
    compute_axial_conductances,
    compute_levels,
    convert_point_process_to_distributed,
    interpolate_xyz,
    loc_of_index,
    query_channel_states_and_params,
    v_interp,
)
from jaxley.utils.debug_solver import compute_morphology_indices
from jaxley.utils.misc_utils import (
    childview,
    concat_and_ignore_empty,
    cumsum_leading_zero,
)
from jaxley.utils.plot_utils import plot_comps, plot_graph, plot_morph
from jaxley.utils.solver_utils import convert_to_csc
from jaxley.utils.swc import build_radiuses_from_xyzr


class Module(ABC):
    """Module base class.

    Modules are everything that can be passed to `jx.integrate`, i.e. compartments,
    branches, cells, and networks.

    This base class defines the scaffold for all jaxley modules (compartments,
    branches, cells, networks).
    """

    def __init__(self):
        self.nseg: int = None
        self.total_nbranches: int = 0
        self.nbranches_per_cell: List[int] = None

        self.group_nodes = {}

        self.nodes: Optional[pd.DataFrame] = None

        self.edges = pd.DataFrame(
            columns=[
                "pre_locs",
                "pre_branch_index",
                "pre_cell_index",
                "post_locs",
                "post_branch_index",
                "post_cell_index",
                "type",
                "type_ind",
                "global_pre_comp_index",
                "global_post_comp_index",
                "global_pre_branch_index",
                "global_post_branch_index",
            ]
        )

        self.cumsum_nbranches: Optional[jnp.ndarray] = None

        self.comb_parents: jnp.ndarray = jnp.asarray([-1])

        self.initialized_morph: bool = False
        self.initialized_syns: bool = False

        # List of all types of `jx.Synapse`s.
        self.synapses: List = []
        self.synapse_param_names = []
        self.synapse_state_names = []
        self.synapse_names = []

        # List of types of all `jx.Channel`s.
        self.channels: List[Channel] = []
        self.membrane_current_names: List[str] = []

        # For trainable parameters.
        self.indices_set_by_trainables: List[jnp.ndarray] = []
        self.trainable_params: List[Dict[str, jnp.ndarray]] = []
        self.allow_make_trainable: bool = True
        self.num_trainable_params: int = 0

        # For recordings.
        self.recordings: pd.DataFrame = pd.DataFrame().from_dict({})

        # For stimuli or clamps.
        # E.g. `self.externals = {"v": zeros(1000,2), "i": ones(1000, 2)}`
        # for 1000 timesteps and two compartments.
        self.externals: Dict[str, jnp.ndarray] = {}
        # E.g. `self.external)inds = {"v": jnp.asarray([0,1]), "i": jnp.asarray([2,3])}`
        self.external_inds: Dict[str, jnp.ndarray] = {}

        # x, y, z coordinates and radius.
        self.xyzr: List[np.ndarray] = []
        self._radius_generating_fns = None  # Defined by `.read_swc()`.

        # For debugging the solver. Will be empty by default and only filled if
        # `self._init_morph_for_debugging` is run.
        self.debug_states = {}

    def _update_nodes_with_xyz(self):
        """Add xyz coordinates of compartment centers to nodes.

        Centers are the midpoint between the comparment endpoints on the morphology
        as defined by xyzr.

        Note: For sake of performance, interpolation is not done for each branch
        individually, but only once along a concatenated (and padded) array of all branches.
        This means for nsegs = [2,4] and normalized cum_branch_lens of [[0,1],[0,1]] we would
        interpolate xyz at the locations comp_ends = [[0,0.5,1], [0,0.25,0.5,0.75,1]],
        where 0 is the start of the branch and 1 is the end point at the full branch_len.
        To avoid do this in one go we set comp_ends = [0,0.5,1,2,2.25,2.5,2.75,3], and
        norm_cum_branch_len = [0,1,2,3] incrememting and also padding them by 1 to
        avoid overlapping branch_lens i.e. norm_cum_branch_len = [0,1,1,2] for only
        incrementing.
        """
        nsegs = self.nodes.groupby("branch_index")["comp_index"].nunique().to_numpy()

        comp_ends = np.hstack(
            [np.linspace(0, 1, nseg + 1) + 2 * i for i, nseg in enumerate(nsegs)]
        )
        comp_ends = comp_ends.reshape(-1)
        cum_branch_lens = []
        for i, xyzr in enumerate(self.xyzr):
            branch_len = np.sqrt(np.sum(np.diff(xyzr[:, :3], axis=0) ** 2, axis=1))
            cum_branch_len = np.cumsum(np.concatenate([np.array([0]), branch_len]))
            max_len = cum_branch_len.max()
            # add padding like above
            cum_branch_len = cum_branch_len / (max_len if max_len > 0 else 1) + 2 * i
            cum_branch_len[np.isnan(cum_branch_len)] = 0
            cum_branch_lens.append(cum_branch_len)
        cum_branch_lens = np.hstack(cum_branch_lens)
        xyz = np.vstack(self.xyzr)[:, :3]
        xyz = v_interp(comp_ends, cum_branch_lens, xyz).T
        centers = (xyz[:-1] + xyz[1:]) / 2  # unaware of inter vs intra comp centers
        cum_nsegs = np.cumsum(nsegs)
        # this means centers between comps have to be removed here
        between_comp_inds = (cum_nsegs + np.arange(len(cum_nsegs)))[:-1]
        centers = np.delete(centers, between_comp_inds, axis=0)
        idcs = self.nodes["comp_index"]
        self.nodes.loc[idcs, ["x", "y", "z"]] = centers
        return centers, xyz

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channels)} different channels. Use `.show()` for details."

    def __str__(self):
        return f"jx.{type(self).__name__}"

    def __dir__(self):
        base_dir = object.__dir__(self)
        return sorted(base_dir + self.synapse_names + list(self.group_nodes.keys()))

    @property
    def _module_type(self):
        """Return type of the module (compartment, branch, cell, network) as string.

        This is used to perform asserts for some modules (e.g. network cannot use
        `set_ncomp`) without having to import the module in `base.py`."""
        return self.__class__.__name__.lower()

    def _append_params_and_states(self, param_dict: Dict, state_dict: Dict):
        """Insert the default params of the module (e.g. radius, length).

        This is run at `__init__()`. It does not deal with channels.
        """
        for param_name, param_value in param_dict.items():
            self.nodes[param_name] = param_value
        for state_name, state_value in state_dict.items():
            self.nodes[state_name] = state_value

    def _gather_channels_from_constituents(self, constituents: List):
        """Modify `self.channels` and `self.nodes` with channel info from constituents.

        This is run at `__init__()`. It takes all branches of constituents (e.g.
        of all branches when the are assembled into a cell) and adds columns to
        `.nodes` for the relevant channels.
        """
        for module in constituents:
            for channel in module.channels:
                if channel._name not in [c._name for c in self.channels]:
                    self.channels.append(channel)
                if channel.current_name not in self.membrane_current_names:
                    self.membrane_current_names.append(channel.current_name)
        # Setting columns of channel names to `False` instead of `NaN`.
        for channel in self.channels:
            name = channel._name
            self.nodes.loc[self.nodes[name].isna(), name] = False

    def to_jax(self):
        """Move `.nodes` to `.jaxnodes`.

        Before the actual simulation is run (via `jx.integrate`), all parameters of
        the `jx.Module` are stored in `.nodes` (a `pd.DataFrame`). However, for
        simulation, these parameters have to be moved to be `jnp.ndarrays` such that
        they can be processed on GPU/TPU and such that the simulation can be
        differentiated. `.to_jax()` copies the `.nodes` to `.jaxnodes`.
        """
        self.jaxnodes = {}
        for key, value in self.nodes.to_dict(orient="list").items():
            inds = jnp.arange(len(value))
            self.jaxnodes[key] = jnp.asarray(value)[inds]

        # `jaxedges` contains only parameters (no indices).
        # `jaxedges` contains only non-Nan elements. This is unlike the channels where
        # we allow parameter sharing.
        self.jaxedges = {}
        edges = self.edges.to_dict(orient="list")
        for i, synapse in enumerate(self.synapses):
            for key in synapse.synapse_params:
                condition = np.asarray(edges["type_ind"]) == i
                self.jaxedges[key] = jnp.asarray(np.asarray(edges[key])[condition])
            for key in synapse.synapse_states:
                self.jaxedges[key] = jnp.asarray(np.asarray(edges[key])[condition])

    def show(
        self,
        param_names: Optional[Union[str, List[str]]] = None,  # TODO.
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
        channel_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Print detailed information about the Module or a view of it.

        Args:
            param_names: The names of the parameters to show. If `None`, all parameters
                are shown. NOT YET IMPLEMENTED.
            indices: Whether to show the indices of the compartments.
            params: Whether to show the parameters of the compartments.
            states: Whether to show the states of the compartments.
            channel_names: The names of the channels to show. If `None`, all channels are
                shown.

        Returns:
            A `pd.DataFrame` with the requested information.
        """
        return self._show(
            self.nodes, param_names, indices, params, states, channel_names
        )

    def _show(
        self,
        view: pd.DataFrame,
        param_names: Optional[Union[str, List[str]]] = None,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
        channel_names: Optional[List[str]] = None,
    ):
        """Print detailed information about the entire Module."""
        printable_nodes = deepcopy(view)

        for channel in self.channels:
            name = channel._name
            param_names = list(channel.channel_params.keys())
            state_names = list(channel.channel_states.keys())
            if channel_names is not None and name not in channel_names:
                printable_nodes = printable_nodes.drop(name, axis=1)
                printable_nodes = printable_nodes.drop(param_names, axis=1)
                printable_nodes = printable_nodes.drop(state_names, axis=1)
            else:
                if not params:
                    printable_nodes = printable_nodes.drop(param_names, axis=1)
                if not states:
                    printable_nodes = printable_nodes.drop(state_names, axis=1)

        if not indices:
            for name in ["comp_index", "branch_index", "cell_index"]:
                printable_nodes = printable_nodes.drop(name, axis=1)

        return printable_nodes

    def init_morph(self):
        """Initialize the morphology such that it can be processed by the solvers."""
        self._init_morph_jaxley_spsolve()
        self._init_morph_jax_spsolve()
        self.initialized_morph = True

    @abstractmethod
    def _init_morph_jax_spsolve(self):
        """Initialize the morphology for the JAX sparse solver."""
        raise NotImplementedError

    @abstractmethod
    def _init_morph_jaxley_spsolve(self):
        """Initialize the morphology for the custom Jaxley solver."""
        raise NotImplementedError

    def _compute_axial_conductances(self, params: Dict[str, jnp.ndarray]):
        """Given radius, length, r_a, compute the axial coupling conductances."""
        return compute_axial_conductances(self._comp_edges, params)

    def _append_channel_to_nodes(self, view: pd.DataFrame, channel: "jx.Channel"):
        """Adds channel nodes from constituents to `self.channel_nodes`."""
        name = channel._name

        # Channel does not yet exist in the `jx.Module` at all.
        if name not in [c._name for c in self.channels]:
            self.channels.append(channel)
            self.nodes[name] = False  # Previous columns do not have the new channel.

        if channel.current_name not in self.membrane_current_names:
            self.membrane_current_names.append(channel.current_name)

        # Add a binary column that indicates if a channel is present.
        self.nodes.loc[view.index.values, name] = True

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_params:
            self.nodes.loc[view.index.values, key] = channel.channel_params[key]

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_states:
            self.nodes.loc[view.index.values, key] = channel.channel_states[key]

    def set(self, key: str, val: Union[float, jnp.ndarray]):
        """Set parameter of module (or its view) to a new value.

        Note that this function can not be called within `jax.jit` or `jax.grad`.
        Instead, it should be used set the parameters of the module **before** the
        simulation. Use `.data_set()` to set parameters during `jax.jit` or
        `jax.grad`.

        Args:
            key: The name of the parameter to set.
            val: The value to set the parameter to. If it is `jnp.ndarray` then it
                must be of shape `(len(num_compartments))`.
        """
        # TODO(@michaeldeistler) should we allow `.set()` for synaptic parameters
        # without using the `SynapseView`, purely for consistency with `make_trainable`?
        view = (
            self.edges
            if key in self.synapse_param_names or key in self.synapse_state_names
            else self.nodes
        )
        self._set(key, val, view, view)

    def _set(
        self,
        key: str,
        val: Union[float, jnp.ndarray],
        view: pd.DataFrame,
        table_to_update: pd.DataFrame,
    ):
        if key in view.columns:
            view = view[~np.isnan(view[key])]
            table_to_update.loc[view.index.values, key] = val
        else:
            raise KeyError("Key not recognized.")

    def data_set(
        self,
        key: str,
        val: Union[float, jnp.ndarray],
        param_state: Optional[List[Dict]],
    ):
        """Set parameter of module (or its view) to a new value within `jit`.

        Args:
            key: The name of the parameter to set.
            val: The value to set the parameter to. If it is `jnp.ndarray` then it
                must be of shape `(len(num_compartments))`.
            param_state: State of the setted parameters, internally used such that this
                function does not modify global state.
        """
        view = (
            self.edges
            if key in self.synapse_param_names or key in self.synapse_state_names
            else self.nodes
        )
        return self._data_set(key, val, view, param_state=param_state)

    def _data_set(
        self,
        key: str,
        val: Tuple[float, jnp.ndarray],
        view: pd.DataFrame,
        param_state: Optional[List[Dict]] = None,
    ):
        # Note: `data_set` does not support arrays for `val`.
        if key in view.columns:
            view = view[~np.isnan(view[key])]
            added_param_state = [
                {
                    "indices": np.atleast_2d(view.index.values),
                    "key": key,
                    "val": jnp.atleast_1d(jnp.asarray(val)),
                }
            ]
            if param_state is not None:
                param_state += added_param_state
            else:
                param_state = added_param_state
        else:
            raise KeyError("Key not recognized.")
        return param_state

    def _set_ncomp(
        self,
        ncomp: int,
        view: pd.DataFrame,
        all_nodes: pd.DataFrame,
        start_idx: int,
        nseg_per_branch: jnp.asarray,
        channel_names: List[str],
        channel_param_names: List[str],
        channel_state_names: List[str],
        radius_generating_fns: List[Callable],
        min_radius: Optional[float],
    ):
        """Set the number of compartments with which the branch is discretized."""
        within_branch_radiuses = view["radius"].to_numpy()
        compartment_lengths = view["length"].to_numpy()
        num_previous_ncomp = len(within_branch_radiuses)
        branch_indices = pd.unique(view["branch_index"])

        error_msg = lambda name: (
            f"You previously modified the {name} of individual compartments, but "
            f"now you are modifying the number of compartments in this branch. "
            f"This is not allowed. First build the morphology with `set_ncomp()` and "
            f"then modify the radiuses and lengths of compartments."
        )

        if (
            ~np.all(within_branch_radiuses == within_branch_radiuses[0])
            and radius_generating_fns is None
        ):
            raise ValueError(error_msg("radius"))

        for property_name in ["length", "capacitance", "axial_resistivity"]:
            compartment_properties = view[property_name].to_numpy()
            if ~np.all(compartment_properties == compartment_properties[0]):
                raise ValueError(error_msg(property_name))

        if not (view[channel_names].var() == 0.0).all():
            raise ValueError(
                "Some channel exists only in some compartments of the branch which you"
                "are trying to modify. This is not allowed. First specify the number"
                "of compartments with `.set_ncomp()` and then insert the channels"
                "accordingly."
            )

        if not (view[channel_param_names + channel_state_names].var() == 0.0).all():
            raise ValueError(
                "Some channel has different parameters or states between the "
                "different compartments of the branch which you are trying to modify. "
                "This is not allowed. First specify the number of compartments with "
                "`.set_ncomp()` and then insert the channels accordingly."
            )

        # Add new rows as the average of all rows. Special case for the length is below.
        average_row = view.mean(skipna=False)
        average_row = average_row.to_frame().T
        view = pd.concat([*[average_row] * ncomp], axis="rows")

        # If the `view` is not the entire `Module`, but a `View` (i.e. if one changes
        # the number of comps within a branch of a cell), then the `self.pointer.view`
        # will contain the additional `global_xyz_index` columns. However, the
        # `self.nodes` will not have these columns.
        #
        # Note that we assert that there are no trainables, so `controlled_by_params`
        # of the `self.nodes` has to be empty.
        if "global_comp_index" in view.columns:
            view = view.drop(
                columns=[
                    "global_comp_index",
                    "global_branch_index",
                    "global_cell_index",
                    "controlled_by_param",
                ]
            )

        # Set the correct datatype after having performed an average which cast
        # everything to float.
        integer_cols = ["comp_index", "branch_index", "cell_index"]
        view[integer_cols] = view[integer_cols].astype(int)

        # Whether or not a channel exists in a compartment is a boolean.
        boolean_cols = channel_names
        view[boolean_cols] = view[boolean_cols].astype(bool)

        # Special treatment for the lengths and radiuses. These are not being set as
        # the average because we:
        # 1) Want to maintain the total length of a branch.
        # 2) Want to use the SWC inferred radius.
        #
        # Compute new compartment lengths.
        comp_lengths = np.sum(compartment_lengths) / ncomp
        view["length"] = comp_lengths

        # Compute new compartment radiuses.
        if radius_generating_fns is not None:
            view["radius"] = build_radiuses_from_xyzr(
                radius_fns=radius_generating_fns,
                branch_indices=branch_indices,
                min_radius=min_radius,
                nseg=ncomp,
            )
        else:
            view["radius"] = within_branch_radiuses[0] * np.ones(ncomp)

        # Update `.nodes`.
        #
        # 1) Delete N rows starting from start_idx
        number_deleted = num_previous_ncomp
        all_nodes = all_nodes.drop(index=range(start_idx, start_idx + number_deleted))

        # 2) Insert M new rows at the same location
        df1 = all_nodes.iloc[:start_idx]  # Rows before the insertion point
        df2 = all_nodes.iloc[start_idx:]  # Rows after the insertion point

        # 3) Combine the parts: before, new rows, and after
        all_nodes = pd.concat([df1, view, df2]).reset_index(drop=True)

        # Override `comp_index` to just be a consecutive list.
        all_nodes["comp_index"] = np.arange(len(all_nodes))

        # Update compartment structure arguments.
        nseg_per_branch[branch_indices] = ncomp
        nseg = int(np.max(nseg_per_branch))
        cumsum_nseg = cumsum_leading_zero(nseg_per_branch)
        internal_node_inds = np.arange(cumsum_nseg[-1])

        return all_nodes, nseg_per_branch, nseg, cumsum_nseg, internal_node_inds

    def make_trainable(
        self,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        """Make a parameter trainable.

        If a parameter is made trainable, it will be returned by `get_parameters()`
        and should then be passed to `jx.integrate(..., params=params)`.

        Args:
            key: Name of the parameter to make trainable.
            init_val: Initial value of the parameter. If `float`, the same value is
                used for every created parameter. If `list`, the length of the list has
                to match the number of created parameters. If `None`, the current
                parameter value is used and if parameter sharing is performed that the
                current parameter value is averaged over all shared parameters.
            verbose: Whether to print the number of parameters that are added and the
                total number of parameters.
        """
        assert (
            key not in self.synapse_param_names and key not in self.synapse_state_names
        ), "Parameters of synapses can only be made trainable via the `SynapseView`."
        view = self.nodes
        view = deepcopy(view.assign(controlled_by_param=0))
        self._make_trainable(view, key, init_val, verbose=verbose)

    def _make_trainable(
        self,
        view: pd.DataFrame,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        assert (
            self.allow_make_trainable
        ), "network.cell('all').make_trainable() is not supported. Use a for-loop over cells."

        if key in view.columns:
            view = view[~np.isnan(view[key])]
            grouped_view = view.groupby("controlled_by_param")
            num_elements_being_set = grouped_view.apply(len).to_numpy()
            assert np.all(num_elements_being_set == num_elements_being_set[0]), (
                "You are using `make_trainable()` with parameter sharing (e.g. same "
                "parameter for an entire cell, or same parameter for entire branches). "
                "This error is caused because you are trying to share a parameter "
                "across an inhomogenous number of compartments. To overcome this "
                "error, write a for-loop across cells (or branches). For example, "
                "change `net.cell('all').make_trainable('HH_gNa')` to "
                "`for i in range(num_cells): net.cell(i).make_trainable('HH_gNa')`"
            )
            # Because of this `x.index.values` we cannot support `make_trainable()` on
            # the module level for synapse parameters (but only for `SynapseView`).
            inds_of_comps = list(grouped_view.apply(lambda x: x.index.values))

            # Sorted inds are only used to infer the correct starting values.
            param_vals = jnp.asarray(
                [view.loc[inds, key].to_numpy() for inds in inds_of_comps]
            )
        else:
            raise KeyError(f"Parameter {key} not recognized.")

        indices_per_param = jnp.stack(inds_of_comps)
        self.indices_set_by_trainables.append(indices_per_param)

        # Set the value which the trainable parameter should take.
        num_created_parameters = len(indices_per_param)
        if init_val is not None:
            if isinstance(init_val, float):
                new_params = jnp.asarray([init_val] * num_created_parameters)
            elif isinstance(init_val, list):
                assert (
                    len(init_val) == num_created_parameters
                ), f"len(init_val)={len(init_val)}, but trying to create {num_created_parameters} parameters."
                new_params = jnp.asarray(init_val)
            else:
                raise ValueError(
                    f"init_val must a float, list, or None, but it is a {type(init_val).__name__}."
                )
        else:
            new_params = jnp.mean(param_vals, axis=1)

        self.trainable_params.append({key: new_params})
        self.num_trainable_params += num_created_parameters
        if verbose:
            print(
                f"Number of newly added trainable parameters: {num_created_parameters}. Total number of trainable parameters: {self.num_trainable_params}"
            )

    def delete_trainables(self):
        """Removes all trainable parameters from the module."""
        self.indices_set_by_trainables: List[jnp.ndarray] = []
        self.trainable_params: List[Dict[str, jnp.ndarray]] = []
        self.num_trainable_params: int = 0

    def add_to_group(self, group_name: str):
        """Add a view of the module to a group.

        Groups can then be indexed. For example:
        ```python
        net.cell(0).add_to_group("excitatory")
        net.excitatory.set("radius", 0.1)
        ```

        Args:
            group_name: The name of the group.
        """
        raise ValueError("`add_to_group()` makes no sense for an entire module.")

    def _add_to_group(self, group_name: str, view: pd.DataFrame):
        if group_name in self.group_nodes:
            view = pd.concat([self.group_nodes[group_name], view])
        self.group_nodes[group_name] = view

    def get_parameters(self) -> List[Dict[str, jnp.ndarray]]:
        """Get all trainable parameters.

        The returned parameters should be passed to `jx.integrate(..., params=params).

        Returns:
            A list of all trainable parameters in the form of
                [{"gNa": jnp.array([0.1, 0.2, 0.3])}, ...].
        """
        return self.trainable_params

    def get_all_parameters(
        self, pstate: List[Dict], voltage_solver: str
    ) -> Dict[str, jnp.ndarray]:
        """Return all parameters (and coupling conductances) needed to simulate.

        Runs `_compute_axial_conductances()` and return every parameter that is needed
        to solve the ODE. This includes conductances, radiuses, lengths,
        axial_resistivities, but also coupling conductances.

        This is done by first obtaining the current value of every parameter (not only
        the trainable ones) and then replacing the trainable ones with the value
        in `trainable_params()`. This function is run within `jx.integrate()`.

        pstate can be obtained by calling `params_to_pstate()`.
        ```
        params = module.get_parameters() # i.e. [0, 1, 2]
        pstate = params_to_pstate(params, module.indices_set_by_trainables)
        module.to_jax() # needed for call to module.jaxnodes
        ```

        Args:
            pstate: The state of the trainable parameters. pstate takes the form
                [{
                    "key": "gNa", "indices": jnp.array([0, 1, 2]),
                    "val": jnp.array([0.1, 0.2, 0.3])
                }, ...].
            voltage_solver: The voltage solver that is used. Since `jax.sparse` and
                `jaxley.xyz` require different formats of the axial conductances, this
                function will default to different building methods.

        Returns:
            A dictionary of all module parameters.
        """
        params = {}
        for key in ["radius", "length", "axial_resistivity", "capacitance"]:
            params[key] = self.jaxnodes[key]

        for channel in self.channels:
            for channel_params in channel.channel_params:
                params[channel_params] = self.jaxnodes[channel_params]

        for synapse_params in self.synapse_param_names:
            params[synapse_params] = self.jaxedges[synapse_params]

        # Override with those parameters set by `.make_trainable()`.
        for parameter in pstate:
            key = parameter["key"]
            inds = parameter["indices"]
            set_param = parameter["val"]
            if key in params:  # Only parameters, not initial states.
                # `inds` is of shape `(num_params, num_comps_per_param)`.
                # `set_param` is of shape `(num_params,)`
                # We need to unsqueeze `set_param` to make it `(num_params, 1)` for the
                # `.set()` to work. This is done with `[:, None]`.
                params[key] = params[key].at[inds].set(set_param[:, None])

        # Compute conductance params and add them to the params dictionary.
        params["axial_conductances"] = self._compute_axial_conductances(params=params)
        return params

    def get_states_from_nodes_and_edges(self):
        """Return states as they are set in the `.nodes` and `.edges` tables."""
        self.to_jax()  # Create `.jaxnodes` from `.nodes` and `.jaxedges` from `.edges`.
        states = {"v": self.jaxnodes["v"]}
        # Join node and edge states into a single state dictionary.
        for channel in self.channels:
            for channel_states in channel.channel_states:
                states[channel_states] = self.jaxnodes[channel_states]
        for synapse_states in self.synapse_state_names:
            states[synapse_states] = self.jaxedges[synapse_states]
        return states

    def get_all_states(
        self, pstate: List[Dict], all_params, delta_t: float
    ) -> Dict[str, jnp.ndarray]:
        """Get the full initial state of the module from jaxnodes and trainables.

        Args:
            pstate: The state of the trainable parameters.
            all_params: All parameters of the module.
            delta_t: The time step.

        Returns:
            A dictionary of all states of the module.
        """
        states = self.get_states_from_nodes_and_edges()

        # Override with the initial states set by `.make_trainable()`.
        for parameter in pstate:
            key = parameter["key"]
            inds = parameter["indices"]
            set_param = parameter["val"]
            if key in states:  # Only initial states, not parameters.
                # `inds` is of shape `(num_params, num_comps_per_param)`.
                # `set_param` is of shape `(num_params,)`
                # We need to unsqueeze `set_param` to make it `(num_params, 1)` for the
                # `.set()` to work. This is done with `[:, None]`.
                states[key] = states[key].at[inds].set(set_param[:, None])

        # Add to the states the initial current through every channel.
        states, _ = self._channel_currents(
            states, delta_t, self.channels, self.nodes, all_params
        )

        # Add to the states the initial current through every synapse.
        states, _ = self._synapse_currents(
            states, self.synapses, all_params, delta_t, self.edges
        )
        return states

    @property
    def initialized(self):
        """Whether the `Module` is ready to be solved or not."""
        return self.initialized_morph and self.initialized_syns

    def initialize(self):
        """Initialize the module."""
        self.init_morph()
        return self

    def init_states(self, delta_t: float = 0.025):
        """Initialize all mechanisms in their steady state.

        This considers the voltages and parameters of each compartment.

        Args:
            delta_t: Passed on to `channel.init_state()`.
        """
        # Update states of the channels.
        channel_nodes = self.nodes
        states = self.get_states_from_nodes_and_edges()

        # We do not use any `pstate` for initializing. In principle, we could change
        # that by allowing an input `params` and `pstate` to this function.
        # `voltage_solver` could also be `jax.sparse` here, because both of them
        # build the channel parameters in the same way.
        params = self.get_all_parameters([], voltage_solver="jaxley.thomas")

        for channel in self.channels:
            name = channel._name
            channel_indices = channel_nodes.loc[channel_nodes[name]][
                "comp_index"
            ].to_numpy()
            voltages = channel_nodes.loc[channel_indices, "v"].to_numpy()

            channel_param_names = list(channel.channel_params.keys())
            channel_state_names = list(channel.channel_states.keys())
            channel_states = query_channel_states_and_params(
                states, channel_state_names, channel_indices
            )
            channel_params = query_channel_states_and_params(
                params, channel_param_names, channel_indices
            )

            init_state = channel.init_state(
                channel_states, voltages, channel_params, delta_t
            )

            # `init_state` might not return all channel states. Only the ones that are
            # returned are updated here.
            for key, val in init_state.items():
                # Note that we are overriding `self.nodes` here, but `self.nodes` is
                # not used above to actually compute the current states (so there are
                # no issues with overriding states).
                self.nodes.loc[channel_indices, key] = val

    def _init_morph_for_debugging(self):
        """Instandiates row and column inds which can be used to solve the voltage eqs.

        This is important only for expert users who try to modify the solver for the
        voltage equations. By default, this function is never run.

        This is useful for debugging the solver because one can use
        `scipy.linalg.sparse.spsolve` after every step of the solve.

        Here is the code snippet that can be used for debugging then (to be inserted in
        `solver_voltage`):
        ```python
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        from jaxley.utils.debug_solver import build_voltage_matrix_elements

        elements, solve, num_entries, start_ind_for_branchpoints = (
            build_voltage_matrix_elements(
                uppers,
                lowers,
                diags,
                solves,
                branchpoint_conds_children[debug_states["child_inds"]],
                branchpoint_conds_parents[debug_states["par_inds"]],
                branchpoint_weights_children[debug_states["child_inds"]],
                branchpoint_weights_parents[debug_states["par_inds"]],
                branchpoint_diags,
                branchpoint_solves,
                debug_states["nseg"],
                nbranches,
            )
        )
        sparse_matrix = csc_matrix(
            (elements, (debug_states["row_inds"], debug_states["col_inds"])),
            shape=(num_entries, num_entries),
        )
        solution = spsolve(sparse_matrix, solve)
        solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
        solves = jnp.reshape(solution, (debug_states["nseg"], nbranches))
        return solves
        ```
        """
        # For scipy and jax.scipy.
        row_and_col_inds = compute_morphology_indices(
            len(self.par_inds),
            self.child_belongs_to_branchpoint,
            self.par_inds,
            self.child_inds,
            self.nseg,
            self.total_nbranches,
        )

        num_elements = len(row_and_col_inds["row_inds"])
        data_inds, indices, indptr = convert_to_csc(
            num_elements=num_elements,
            row_ind=row_and_col_inds["row_inds"],
            col_ind=row_and_col_inds["col_inds"],
        )
        self.debug_states["row_inds"] = row_and_col_inds["row_inds"]
        self.debug_states["col_inds"] = row_and_col_inds["col_inds"]
        self.debug_states["data_inds"] = data_inds
        self.debug_states["indices"] = indices
        self.debug_states["indptr"] = indptr

        self.debug_states["nseg"] = self.nseg
        self.debug_states["child_inds"] = self.child_inds
        self.debug_states["par_inds"] = self.par_inds

    def record(self, state: str = "v", verbose: bool = True):
        """Insert a recording into the compartment.

        Args:
            state: The name of the state to record.
            verbose: Whether to print number of inserted recordings."""
        view = deepcopy(self.nodes)
        view["state"] = state
        recording_view = view[["comp_index", "state"]]
        recording_view = recording_view.rename(columns={"comp_index": "rec_index"})
        self._record(recording_view, verbose=verbose)

    def _record(self, view: pd.DataFrame, verbose: bool = True):
        self.recordings = pd.concat([self.recordings, view], ignore_index=True)
        if verbose:
            print(f"Added {len(view)} recordings. See `.recordings` for details.")

    def delete_recordings(self):
        """Removes all recordings from the module."""
        self.recordings = pd.DataFrame().from_dict({})

    def stimulate(self, current: Optional[jnp.ndarray] = None, verbose: bool = True):
        """Insert a stimulus into the compartment.

        current must be a 1d array or have batch dimension of size `(num_compartments, )`
        or `(1, )`. If 1d, the same stimulus is added to all compartments.

        This function cannot be run during `jax.jit` and `jax.grad`. Because of this,
        it should only be used for static stimuli (i.e., stimuli that do not depend
        on the data and that should not be learned). For stimuli that depend on data
        (or that should be learned), please use `data_stimulate()`.

        Args:
            current: Current in `nA`.
        """
        self._external_input("i", current, self.nodes, verbose=verbose)

    def clamp(self, state_name: str, state_array: jnp.ndarray, verbose: bool = True):
        """Clamp a state to a given value across specified compartments.

        Args:
            state_name: The name of the state to clamp.
            state_array (jnp.nd: Array of values to clamp the state to.
            verbose : If True, prints details about the clamping.

        This function sets external states for the compartments.
        """
        if state_name not in self.nodes.columns:
            raise KeyError(f"{state_name} is not a recognized state in this module.")
        self._external_input(state_name, state_array, self.nodes, verbose=verbose)

    def _external_input(
        self,
        key: str,
        values: Optional[jnp.ndarray],
        view: pd.DataFrame,
        verbose: bool = True,
    ):
        values = values if values.ndim == 2 else jnp.expand_dims(values, axis=0)
        batch_size = values.shape[0]
        is_multiple = len(view) == batch_size
        values = values if is_multiple else jnp.repeat(values, len(view), axis=0)
        assert batch_size in [1, len(view)], "Number of comps and stimuli do not match."

        if key in self.externals.keys():
            self.externals[key] = jnp.concatenate([self.externals[key], values])
            self.external_inds[key] = jnp.concatenate(
                [self.external_inds[key], view.comp_index.to_numpy()]
            )
        else:
            self.externals[key] = values
            self.external_inds[key] = view.comp_index.to_numpy()

        if verbose:
            print(f"Added {len(view)} external_states. See `.externals` for details.")

    def data_stimulate(
        self,
        current: jnp.ndarray,
        data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
        verbose: bool = False,
    ) -> Tuple[jnp.ndarray, pd.DataFrame]:
        """Insert a stimulus into the module within jit (or grad).

        Args:
            current: Current in `nA`.
            verbose: Whether or not to print the number of inserted stimuli. `False`
                by default because this method is meant to be jitted.
        """
        return self._data_external_input(
            "i", current, data_stimuli, self.nodes, verbose=verbose
        )

    def data_clamp(
        self,
        state_name: str,
        state_array: jnp.ndarray,
        data_clamps: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
        verbose: bool = False,
    ):
        """Insert a clamp into the module within jit (or grad).

        Args:
            state_name: Name of the state variable to set.
            state_array: Time series of the state variable in the default Jaxley unit.
                State array should be of shape (num_clamps, simulation_time) or
                (simulation_time, ) for a single clamp.
            verbose: Whether or not to print the number of inserted clamps. `False`
                by default because this method is meant to be jitted.
        """
        return self._data_external_input(
            state_name, state_array, data_clamps, self.nodes, verbose=verbose
        )

    def _data_external_input(
        self,
        state_name: str,
        state_array: jnp.ndarray,
        data_external_input: Optional[Tuple[jnp.ndarray, pd.DataFrame]],
        view: pd.DataFrame,
        verbose: bool = False,
    ):
        state_array = (
            state_array
            if state_array.ndim == 2
            else jnp.expand_dims(state_array, axis=0)
        )
        batch_size = state_array.shape[0]
        is_multiple = len(view) == batch_size
        state_array = (
            state_array if is_multiple else jnp.repeat(state_array, len(view), axis=0)
        )
        assert batch_size in [1, len(view)], "Number of comps and clamps do not match."

        if data_external_input is not None:
            external_input = data_external_input[1]
            external_input = jnp.concatenate([external_input, state_array])
            inds = data_external_input[2]
        else:
            external_input = state_array
            inds = pd.DataFrame().from_dict({})

        inds = pd.concat([inds, view])

        if verbose:
            if state_name == "i":
                print(f"Added {len(view)} stimuli.")
            else:
                print(f"Added {len(view)} clamps.")

        return (state_name, external_input, inds)

    def delete_stimuli(self):
        """Removes all stimuli from the module."""
        self.externals.pop("i", None)
        self.external_inds.pop("i", None)

    def delete_clamps(self, state_name: str):
        """Removes all clamps of the given state from the module."""
        self.externals.pop(state_name, None)
        self.external_inds.pop(state_name, None)

    def insert(self, channel: Channel):
        """Insert a channel into the module.

        Args:
            channel: The channel to insert."""
        self._insert(channel, self.nodes)

    def _insert(self, channel, view):
        self._append_channel_to_nodes(view, channel)

    def init_syns(self):
        self.initialized_syns = True

    def step(
        self,
        u: Dict[str, jnp.ndarray],
        delta_t: float,
        external_inds: Dict[str, jnp.ndarray],
        externals: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
        solver: str = "bwd_euler",
        voltage_solver: str = "jaxley.stone",
    ) -> Dict[str, jnp.ndarray]:
        """One step of solving the Ordinary Differential Equation.

        This function is called inside of `integrate` and increments the state of the
        module by one time step. Calls `_step_channels` and `_step_synapse` to update
        the states of the channels and synapses using fwd_euler.

        Args:
            u: The state of the module. voltages = u["v"]
            delta_t: The time step.
            external_inds: The indices of the external inputs.
            externals: The external inputs.
            params: The parameters of the module.
            solver: The solver to use for the voltages. Either of ["bwd_euler",
                "fwd_euler", "crank_nicolson"].
            voltage_solver: The tridiagonal solver used to diagonalize the
                coefficient matrix of the ODE system. Either of ["jaxley.thomas",
                "jaxley.stone"].

        Returns:
            The updated state of the module.
        """

        # Extract the voltages
        voltages = u["v"]

        # Extract the external inputs
        if "i" in externals.keys():
            i_current = externals["i"]
            i_inds = external_inds["i"]
            i_ext = self._get_external_input(
                voltages, i_inds, i_current, params["radius"], params["length"]
            )
        else:
            i_ext = 0.0

        # Step of the channels.
        u, (v_terms, const_terms) = self._step_channels(
            u, delta_t, self.channels, self.nodes, params
        )

        # Step of the synapse.
        u, (syn_v_terms, syn_const_terms) = self._step_synapse(
            u,
            self.synapses,
            params,
            delta_t,
            self.edges,
        )

        # Clamp for channels and synapses.
        for key in externals.keys():
            if key not in ["i", "v"]:
                u[key] = u[key].at[external_inds[key]].set(externals[key])

        # Voltage steps.
        cm = params["capacitance"]  # Abbreviation.

        # Arguments used by all solvers.
        solver_kwargs = {
            "voltages": voltages,
            "voltage_terms": (v_terms + syn_v_terms) / cm,
            "constant_terms": (const_terms + i_ext + syn_const_terms) / cm,
            "axial_conductances": params["axial_conductances"],
            "internal_node_inds": self._internal_node_inds,
        }

        # Add solver specific arguments.
        if voltage_solver == "jax.sparse":
            solver_kwargs.update(
                {
                    "sinks": np.asarray(self._comp_edges["sink"].to_list()),
                    "data_inds": self._data_inds,
                    "indices": self._indices_jax_spsolve,
                    "indptr": self._indptr_jax_spsolve,
                    "n_nodes": self._n_nodes,
                }
            )
            # Only for `bwd_euler` and `cranck-nicolson`.
            step_voltage_implicit = step_voltage_implicit_with_jax_spsolve
        else:
            # Our custom sparse solver requires a different format of all conductance
            # values to perform triangulation and backsubstution optimally.
            #
            # Currently, the forward Euler solver also uses this format. However,
            # this is only for historical reasons and we are planning to change this in
            # the future.
            solver_kwargs.update(
                {
                    "sinks": np.asarray(self._comp_edges["sink"].to_list()),
                    "sources": np.asarray(self._comp_edges["source"].to_list()),
                    "types": np.asarray(self._comp_edges["type"].to_list()),
                    "nseg_per_branch": self.nseg_per_branch,
                    "par_inds": self.par_inds,
                    "child_inds": self.child_inds,
                    "nbranches": self.total_nbranches,
                    "solver": voltage_solver,
                    "idx": self.solve_indexer,
                    "debug_states": self.debug_states,
                }
            )
            # Only for `bwd_euler` and `cranck-nicolson`.
            step_voltage_implicit = step_voltage_implicit_with_jaxley_spsolve

        if solver == "bwd_euler":
            u["v"] = step_voltage_implicit(**solver_kwargs, delta_t=delta_t)
        elif solver == "crank_nicolson":
            # Crank-Nicolson advances by half a step of backward and half a step of
            # forward Euler.
            half_step_delta_t = delta_t / 2
            half_step_voltages = step_voltage_implicit(
                **solver_kwargs, delta_t=half_step_delta_t
            )
            # The forward Euler step in Crank-Nicolson can be performed easily as
            # `V_{n+1} = 2 * V_{n+1/2} - V_n`. See also NEURON book Chapter 4.
            u["v"] = 2 * half_step_voltages - voltages
        elif solver == "fwd_euler":
            u["v"] = step_voltage_explicit(**solver_kwargs, delta_t=delta_t)
        else:
            raise ValueError(
                f"You specified `solver={solver}`. The only allowed solvers are "
                "['bwd_euler', 'fwd_euler', 'crank_nicolson']."
            )

        # Clamp for voltages.
        if "v" in externals.keys():
            u["v"] = u["v"].at[external_inds["v"]].set(externals["v"])

        return u

    def _step_channels(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """One step of integration of the channels and of computing their current."""
        states = self._step_channels_state(
            states, delta_t, channels, channel_nodes, params
        )
        states, current_terms = self._channel_currents(
            states, delta_t, channels, channel_nodes, params
        )
        return states, current_terms

    def _step_channels_state(
        self,
        states,
        delta_t,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """One integration step of the channels."""
        voltages = states["v"]

        # Update states of the channels.
        indices = channel_nodes["comp_index"].to_numpy()
        for channel in channels:
            channel_param_names = list(channel.channel_params)
            channel_param_names += [
                "radius",
                "length",
                "axial_resistivity",
                "capacitance",
            ]
            channel_state_names = list(channel.channel_states)
            channel_state_names += self.membrane_current_names
            channel_indices = indices[channel_nodes[channel._name].astype(bool)]

            channel_params = query_channel_states_and_params(
                params, channel_param_names, channel_indices
            )
            channel_states = query_channel_states_and_params(
                states, channel_state_names, channel_indices
            )

            states_updated = channel.update_states(
                channel_states, delta_t, voltages[channel_indices], channel_params
            )
            # Rebuild state. This has to be done within the loop over channels to allow
            # multiple channels which modify the same state.
            for key, val in states_updated.items():
                states[key] = states[key].at[channel_indices].set(val)

        return states

    def _channel_currents(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return the current through each channel.

        This is also updates `state` because the `state` also contains the current.
        """
        voltages = states["v"]

        # Compute current through channels.
        voltage_terms = jnp.zeros_like(voltages)
        constant_terms = jnp.zeros_like(voltages)
        # Run with two different voltages that are `diff` apart to infer the slope and
        # offset.
        diff = 1e-3

        current_states = {}
        for name in self.membrane_current_names:
            current_states[name] = jnp.zeros_like(voltages)

        for channel in channels:
            name = channel._name
            channel_param_names = list(channel.channel_params.keys())
            channel_state_names = list(channel.channel_states.keys())
            indices = channel_nodes.loc[channel_nodes[name]]["comp_index"].to_numpy()

            channel_params = {}
            for p in channel_param_names:
                channel_params[p] = params[p][indices]
            channel_params["radius"] = params["radius"][indices]
            channel_params["length"] = params["length"][indices]
            channel_params["axial_resistivity"] = params["axial_resistivity"][indices]

            channel_states = {}
            for s in channel_state_names:
                channel_states[s] = states[s][indices]

            v_and_perturbed = jnp.stack([voltages[indices], voltages[indices] + diff])
            membrane_currents = vmap(channel.compute_current, in_axes=(None, 0, None))(
                channel_states, v_and_perturbed, channel_params
            )
            voltage_term = (membrane_currents[1] - membrane_currents[0]) / diff
            constant_term = membrane_currents[0] - voltage_term * voltages[indices]
            voltage_terms = voltage_terms.at[indices].add(voltage_term)
            constant_terms = constant_terms.at[indices].add(-constant_term)

            # Save the current (for the unperturbed voltage) as a state that will
            # also be passed to the state update.
            current_states[channel.current_name] = (
                current_states[channel.current_name]
                .at[indices]
                .add(membrane_currents[0])
            )

        # Copy the currents into the `state` dictionary such that they can be
        # recorded and used by `Channel.update_states()`.
        for name in self.membrane_current_names:
            states[name] = current_states[name]

        return states, (voltage_terms, constant_terms)

    def _step_synapse(
        self,
        u: Dict[str, jnp.ndarray],
        syn_channels: List[Channel],
        params: Dict[str, jnp.ndarray],
        delta_t: float,
        edges: pd.DataFrame,
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """One step of integration of the channels.

        `Network` overrides this method (because it actually has synapses), whereas
        `Compartment`, `Branch`, and `Cell` do not override this.
        """
        voltages = u["v"]
        return u, (jnp.zeros_like(voltages), jnp.zeros_like(voltages))

    def _synapse_currents(
        self, states, syn_channels, params, delta_t, edges: pd.DataFrame
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        return states, (None, None)

    @staticmethod
    def _get_external_input(
        voltages: jnp.ndarray,
        i_inds: jnp.ndarray,
        i_stim: jnp.ndarray,
        radius: float,
        length_single_compartment: float,
    ) -> jnp.ndarray:
        """
        Return external input to each compartment in uA / cm^2.

        Args:
            voltages: mV.
            i_stim: nA.
            radius: um.
            length_single_compartment: um.
        """
        zero_vec = jnp.zeros_like(voltages)
        current = convert_point_process_to_distributed(
            i_stim, radius[i_inds], length_single_compartment[i_inds]
        )

        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        stim_at_timestep = scatter_add(zero_vec, i_inds[:, None], current, dnums)
        return stim_at_timestep

    def vis(
        self,
        ax: Optional[Axes] = None,
        col: str = "k",
        dims: Tuple[int] = (0, 1),
        type: str = "line",
        morph_plot_kwargs: Dict = {},
    ) -> Axes:
        """Visualize the module.

        Modules can be visualized on one of the cardinal planes (xy, xz, yz) or
        even in 3D.

        Several options are available:
        - `line`: All points from the traced morphology (`xyzr`), are connected
        with a line plot.
        - `scatter`: All traced points, are plotted as scatter points.
        - `comp`: Plots the compartmentalized morphology, including radius
        and shape. (shows the true compartment lengths per default, but this can
        be changed via the `morph_plot_kwargs`, for details see
        `jaxley.utils.plot_utils.plot_comps`).
        - `morph`: Reconstructs the 3D shape of the traced morphology. For details see
        `jaxley.utils.plot_utils.plot_morph`. Warning: For 3D plots and morphologies
        with many traced points this can be very slow.

        Args:
            ax: An axis into which to plot.
            col: The color for all branches.
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
            type: The type of plot. One of ["line", "scatter", "comp", "morph"].
            morph_plot_kwargs: Keyword arguments passed to the plotting function.
        """
        return self._vis(
            dims=dims,
            col=col,
            ax=ax,
            view=self.nodes,
            type=type,
            morph_plot_kwargs=morph_plot_kwargs,
        )

    def _vis(
        self,
        ax: Axes,
        col: str,
        dims: Tuple[int],
        view: pd.DataFrame,
        type: str,
        morph_plot_kwargs: Dict,
    ) -> Axes:
        branches_inds = view["branch_index"].to_numpy()

        if "comp" in type.lower():
            return plot_comps(
                self, view, dims=dims, ax=ax, col=col, **morph_plot_kwargs
            )
        if "morph" in type.lower():
            return plot_morph(
                self, view, dims=dims, ax=ax, col=col, **morph_plot_kwargs
            )

        coords = []
        for branch_ind in branches_inds:
            assert not np.any(
                np.isnan(self.xyzr[branch_ind][:, dims])
            ), "No coordinates available. Use `vis(detail='point')` or run `.compute_xyz()` before running `.vis()`."
            coords.append(self.xyzr[branch_ind])

        ax = plot_graph(
            coords,
            dims=dims,
            col=col,
            ax=ax,
            type=type,
            morph_plot_kwargs=morph_plot_kwargs,
        )

        return ax

    def _scatter(self, ax, col, dims, view, morph_plot_kwargs):
        """Scatter visualization (used only for compartments)."""
        assert len(view) == 1, "Scatter only deals with compartments."
        branch_ind = view["branch_index"].to_numpy().item()
        comp_ind = view["comp_index"].to_numpy().item()
        assert not np.any(
            np.isnan(self.xyzr[branch_ind][:, dims])
        ), "No coordinates available. Use `vis(detail='point')` or run `.compute_xyz()` before running `.vis()`."

        comp_fraction = loc_of_index(
            comp_ind,
            branch_ind,
            self.nseg_per_branch,
        )
        coords = self.xyzr[branch_ind]
        interpolated_xyz = interpolate_xyz(comp_fraction, coords)

        ax = plot_graph(
            np.asarray([[interpolated_xyz]]),
            dims=dims,
            col=col,
            ax=ax,
            type="scatter",
            morph_plot_kwargs=morph_plot_kwargs,
        )

        return ax

    def compute_xyz(self):
        """Return xyz coordinates of every branch, based on the branch length.

        This function should not be called if the morphology was read from an `.swc`
        file. However, for morphologies that were constructed from scratch, this
        function **must** be called before `.vis()`. The computed `xyz` coordinates
        are only used for plotting.
        """
        max_y_multiplier = 5.0
        min_y_multiplier = 0.5

        parents = self.comb_parents
        num_children = _compute_num_children(parents)
        index_of_child = _compute_index_of_child(parents)
        levels = compute_levels(parents)

        # Extract branch.
        inds_branch = self.nodes.groupby("branch_index")["comp_index"].apply(list)
        branch_lens = [np.sum(self.nodes["length"][np.asarray(i)]) for i in inds_branch]
        endpoints = []

        # Different levels will get a different "angle" at which the children emerge from
        # the parents. This angle is defined by the `y_offset_multiplier`. This value
        # defines the range between y-location of the first and of the last child of a
        # parent.
        y_offset_multiplier = np.linspace(
            max_y_multiplier, min_y_multiplier, np.max(levels) + 1
        )

        for b in range(self.total_nbranches):
            # For networks with mixed SWC and from-scatch neurons, only update those
            # branches that do not have coordingates yet.
            if np.any(np.isnan(self.xyzr[b])):
                if parents[b] > -1:
                    start_point = endpoints[parents[b]]
                    num_children_of_parent = num_children[parents[b]]
                    if num_children_of_parent > 1:
                        y_offset = (
                            ((index_of_child[b] / (num_children_of_parent - 1))) - 0.5
                        ) * y_offset_multiplier[levels[b]]
                    else:
                        y_offset = 0.0
                else:
                    start_point = [0, 0, 0]
                    y_offset = 0.0

                len_of_path = np.sqrt(y_offset**2 + 1.0)

                end_point = [
                    start_point[0] + branch_lens[b] / len_of_path * 1.0,
                    start_point[1] + branch_lens[b] / len_of_path * y_offset,
                    start_point[2],
                ]
                endpoints.append(end_point)

                self.xyzr[b][:, :3] = np.asarray([start_point, end_point])
            else:
                # Dummy to keey the index `endpoints[parent[b]]` above working.
                endpoints.append(np.zeros((2,)))

    def move(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, update_nodes: bool = True
    ):
        """Move cells or networks by adding to their (x, y, z) coordinates.

        This function is used only for visualization. It does not affect the simulation.

        Args:
            x: The amount to move in the x direction in um.
            y: The amount to move in the y direction in um.
            z: The amount to move in the z direction in um.
            update_nodes: Whether `.nodes` should be updated or not. Setting this to
                `False` largely speeds up moving, especially for big networks, but
                `.nodes` or `.show` will not show the new xyz coordinates.
        """
        self._move(x, y, z, self.nodes, update_nodes)

    def _move(self, x: float, y: float, z: float, view, update_nodes: bool):
        # Need to cast to set because this will return one columnn per compartment,
        # not one column per branch.
        indizes = set(view["branch_index"].to_numpy().tolist())
        for i in indizes:
            self.xyzr[i][:, 0] += x
            self.xyzr[i][:, 1] += y
            self.xyzr[i][:, 2] += z
        if update_nodes:
            self._update_nodes_with_xyz()

    def move_to(
        self,
        x: Union[float, np.ndarray] = 0.0,
        y: Union[float, np.ndarray] = 0.0,
        z: Union[float, np.ndarray] = 0.0,
        update_nodes: bool = True,
    ):
        """Move cells or networks to a location (x, y, z).

        If x, y, and z are floats, then the first compartment of the first branch
        of the first cell is moved to that float coordinate, and everything else is
        shifted by the difference between that compartment's previous coordinate and
        the new float location.

        If x, y, and z are arrays, then they must each have a length equal to the number
        of cells being moved. Then the first compartment of the first branch of each
        cell is moved to the specified location.

        Args:
            update_nodes: Whether `.nodes` should be updated or not. Setting this to
                `False` largely speeds up moving, especially for big networks, but
                `.nodes` or `.show` will not show the new xyz coordinates.
        """
        self._move_to(x, y, z, self.nodes, update_nodes)

    def _move_to(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        view: pd.DataFrame,
        update_nodes: bool,
    ):
        # Test if any coordinate values are NaN which would greatly affect moving
        if np.any(np.concatenate(self.xyzr, axis=0)[:, :3] == np.nan):
            raise ValueError(
                "NaN coordinate values detected. Shift amounts cannot be computed. Please run compute_xyzr() or assign initial coordinate values."
            )

        # Get the indices of the cells and branches to move
        cell_inds = list(view.cell_index.unique())
        branch_inds = view.branch_index.unique()

        if (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and isinstance(z, np.ndarray)
        ):
            assert (
                x.shape == y.shape == z.shape == (len(cell_inds),)
            ), "x, y, and z array shapes are not all equal to the number of cells to be moved."

            # Split the branches by cell id
            tup_indices = np.array([view.cell_index, view.branch_index])
            view_cell_branch_inds = np.unique(tup_indices, axis=1)[0]
            _, branch_split_inds = np.unique(view_cell_branch_inds, return_index=True)
            branches_by_cell = np.split(
                view.branch_index.unique(), branch_split_inds[1:]
            )

            # Calculate the amount to shift all of the branches of each cell
            shift_amounts = (
                np.array([x, y, z]).T - np.stack(self[cell_inds, 0].xyzr)[:, 0, :3]
            )

        else:
            # Treat as if all branches belong to the same cell to be moved
            branches_by_cell = [branch_inds]
            # Calculate the amount to shift all branches by the 1st branch of 1st cell
            shift_amounts = [np.array([x, y, z]) - self[cell_inds].xyzr[0][0, :3]]

        # Move all of the branches
        for i, branches in enumerate(branches_by_cell):
            for b in branches:
                self.xyzr[b][:, :3] += shift_amounts[i]

        if update_nodes:
            self._update_nodes_with_xyz()

    def rotate(self, degrees: float, rotation_axis: str = "xy"):
        """Rotate jaxley modules clockwise. Used only for visualization.

        This function is used only for visualization. It does not affect the simulation.

        Args:
            degrees: How many degrees to rotate the module by.
            rotation_axis: Either of {`xy` | `xz` | `yz`}.
        """
        self._rotate(degrees=degrees, rotation_axis=rotation_axis, view=self.nodes)

    def _rotate(self, degrees: float, rotation_axis: str, view: pd.DataFrame):
        degrees = degrees / 180 * np.pi
        if rotation_axis == "xy":
            dims = [0, 1]
        elif rotation_axis == "xz":
            dims = [0, 2]
        elif rotation_axis == "yz":
            dims = [1, 2]
        else:
            raise ValueError

        rotation_matrix = np.asarray(
            [[np.cos(degrees), np.sin(degrees)], [-np.sin(degrees), np.cos(degrees)]]
        )
        indizes = set(view["branch_index"].to_numpy().tolist())
        for i in indizes:
            rot = np.dot(rotation_matrix, self.xyzr[i][:, dims].T).T
            self.xyzr[i][:, dims] = rot

    @property
    def shape(self) -> Tuple[int]:
        """Returns the number of submodules contained in a module.

        ```
        network.shape = (num_cells, num_branches, num_compartments)
        cell.shape = (num_branches, num_compartments)
        branch.shape = (num_compartments,)
        ```"""
        mod_name = self.__class__.__name__.lower()
        if "comp" in mod_name:
            return (1,)
        elif "branch" in mod_name:
            return self[:].shape[1:]
        return self[:].shape

    def __getitem__(self, index):
        return self._getitem(self, index)

    def _getitem(
        self,
        module: Union["Module", "View"],
        index: Union[Tuple, int],
        child_name: Optional[str] = None,
    ) -> "View":
        """Return View which is created from indexing the module.

        Args:
            module: The module to be indexed. Will be a `Module` if `._getitem` is
                called from `__getitem__` in a `Module` and will be a `View` if it was
                called from `__getitem__` in a `View`.
            index: The index (or indices) to index the module.
            child_name: If passed, this will be the key that is used to index the
                `module`, e.g. if it is the string `branch` then we will try to call
                `module.xyz(index)`. If `None` then we try to infer automatically what
                the childview should be, given the name of the `module`.

        Returns:
            An indexed `View`.
        """
        if isinstance(index, tuple):
            if len(index) > 1:
                return childview(module, index[0], child_name)[index[1:]]
            return childview(module, index[0], child_name)
        return childview(module, index, child_name)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]


class View:
    """View of a `Module`."""

    def __init__(self, pointer: Module, view: pd.DataFrame):
        self.pointer = pointer
        self.view = view
        self.allow_make_trainable = True

    def __repr__(self):
        return f"{type(self).__name__}. Use `.show()` for details."

    def __str__(self):
        return f"{type(self).__name__}"

    def show(
        self,
        param_names: Optional[Union[str, List[str]]] = None,  # TODO.
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
        channel_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Print detailed information about the Module or a view of it.

        Args:
            param_names: The names of the parameters to show. If `None`, all parameters
                are shown. NOT YET IMPLEMENTED.
            indices: Whether to show the indices of the compartments.
            params: Whether to show the parameters of the compartments.
            states: Whether to show the states of the compartments.
            channel_names: The names of the channels to show. If `None`, all channels are
                shown.

        Returns:
            A `pd.DataFrame` with the requested information.
        """
        view = self.pointer._show(
            self.view, param_names, indices, params, states, channel_names
        )
        if not indices:
            for name in [
                "global_comp_index",
                "global_branch_index",
                "global_cell_index",
                "controlled_by_param",
            ]:
                if name in view.columns:
                    view = view.drop(name, axis=1)
        return view

    def set_global_index_and_index(self, nodes: pd.DataFrame) -> pd.DataFrame:
        """Use the global compartment, branch, and cell index as the index."""
        nodes = nodes.drop("controlled_by_param", axis=1)
        nodes = nodes.drop("comp_index", axis=1)
        nodes = nodes.drop("branch_index", axis=1)
        nodes = nodes.drop("cell_index", axis=1)
        nodes = nodes.rename(
            columns={
                "global_comp_index": "comp_index",
                "global_branch_index": "branch_index",
                "global_cell_index": "cell_index",
            }
        )
        return nodes

    def insert(self, channel: Channel):
        """Insert a channel into the module at the currently viewed location(s).

        Args:
            channel: The channel to insert.
        """
        assert not inspect.isclass(
            channel
        ), """
            Channel is a class, but it was not initialized. Use `.insert(Channel())`
            instead of `.insert(Channel)`.
            """
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._insert(channel, nodes)

    def record(self, state: str = "v", verbose: bool = True):
        """Record a state variable of the compartment(s) at the currently view location(s).

        Args:
            state: The name of the state to record.
            verbose: Whether to print number of inserted recordings."""
        nodes = self.set_global_index_and_index(self.view)
        view = deepcopy(nodes)
        view["state"] = state
        recording_view = view[["comp_index", "state"]]
        recording_view = recording_view.rename(columns={"comp_index": "rec_index"})
        self.pointer._record(recording_view, verbose=verbose)

    def stimulate(self, current: Optional[jnp.ndarray] = None, verbose: bool = True):
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._external_input("i", current, nodes, verbose=verbose)

    def data_stimulate(
        self,
        current: jnp.ndarray,
        data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]],
        verbose: bool = False,
    ):
        """Insert a stimulus into the module within jit (or grad).

        Args:
            current: Current in `nA`.
            verbose: Whether or not to print the number of inserted stimuli. `False`
                by default because this method is meant to be jitted.
        """
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._data_external_input(
            "i", current, data_stimuli, nodes, verbose=verbose
        )

    def clamp(self, state_name: str, state_array: jnp.ndarray, verbose: bool = True):
        """Clamp a state to a given value across specified compartments.

        Args:
            state_name: The name of the state to clamp.
            state_array: Array of values to clamp the state to.
            verbose: If True, prints details about the clamping.

        This function sets external states for the compartments.
        """
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._external_input(state_name, state_array, nodes, verbose=verbose)

    def data_clamp(
        self,
        state_name: str,
        state_array: jnp.ndarray,
        data_clamps: Optional[Tuple[jnp.ndarray, pd.DataFrame]],
        verbose: bool = False,
    ):
        """Insert a clamp into the module within jit (or grad)."""
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._data_external_input(
            state_name, state_array, data_clamps, nodes, verbose=verbose
        )

    def set(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.pointer._set(key, val, self.view, self.pointer.nodes)

    def data_set(
        self,
        key: str,
        val: Union[float, jnp.ndarray],
        param_state: Optional[List[Dict]] = None,
    ):
        """Set parameter of module (or its view) to a new value within `jit`."""
        return self.pointer._data_set(key, val, self.view, param_state)

    def make_trainable(
        self,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        """Make a parameter trainable."""
        self.pointer._make_trainable(self.view, key, init_val, verbose=verbose)

    def add_to_group(self, group_name: str):
        self.pointer._add_to_group(group_name, self.view)

    def vis(
        self,
        ax: Optional[Axes] = None,
        col: str = "k",
        dims: Tuple[int] = (0, 1),
        type: str = "line",
        morph_plot_kwargs: Dict = {},
    ) -> Axes:
        """Visualize the module.

        Modules can be visualized on one of the cardinal planes (xy, xz, yz) or
        even in 3D.

        Several options are available:
        - `line`: All points from the traced morphology (`xyzr`), are connected
        with a line plot.
        - `scatter`: All traced points, are plotted as scatter points.
        - `comp`: Plots the compartmentalized morphology, including radius
        and shape. (shows the true compartment lengths per default, but this can
        be changed via the `morph_plot_kwargs`, for details see
        `jaxley.utils.plot_utils.plot_comps`).
        - `morph`: Reconstructs the 3D shape of the traced morphology. For details see
        `jaxley.utils.plot_utils.plot_morph`. Warning: For 3D plots and morphologies
        with many traced points this can be very slow.

        Args:
            ax: An axis into which to plot.
            col: The color for all branches.
            type: The type of plot. One of ["line", "scatter", "comp", "morph"].
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
            morph_plot_kwargs: Keyword arguments passed to the plotting function.
        """
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._vis(
            ax=ax,
            col=col,
            dims=dims,
            view=nodes,
            type=type,
            morph_plot_kwargs=morph_plot_kwargs,
        )

    def move(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, update_nodes: bool = True
    ):
        """Move cells or networks by adding to their (x, y, z) coordinates.

        This function is used only for visualization. It does not affect the simulation.

        Args:
            x: The amount to move in the x direction in um.
            y: The amount to move in the y direction in um.
            z: The amount to move in the z direction in um.
        """
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._move(x, y, z, nodes, update_nodes=update_nodes)

    def move_to(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, update_nodes: bool = True
    ):
        """Move cells or networks to a location (x, y, z).

        If x, y, and z are floats, then the first compartment of the first branch
        of the first cell is moved to that float coordinate, and everything else is
        shifted by the difference between that compartment's previous coordinate and
        the new float location.

        If x, y, and z are arrays, then they must each have a length equal to the number
        of cells being moved. Then the first compartment of the first branch of each
        cell is moved to the specified location.
        """
        # Ensuring here that the branch indices in the view passed are global
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._move_to(x, y, z, nodes, update_nodes=update_nodes)

    def adjust_view(
        self, key: str, index: Union[int, str, list, range, slice]
    ) -> "View":
        """Update view.

        Select a subset, range, slice etc. of the self.view based on the index key,
        i.e. (cell_index, [1,2]). returns a view of all compartments of cell 1 and 2.

        Args:
            key: The key to adjust the view by.
            index: The index to adjust the view by.

        Returns:
            A new view.
        """
        if isinstance(index, int) or isinstance(index, np.int64):
            self.view = self.view[self.view[key] == index]
        elif isinstance(index, list) or isinstance(index, range):
            self.view = self.view[self.view[key].isin(index)]
        elif isinstance(index, slice):
            index = list(range(self.view[key].max() + 1))[index]
            return self.adjust_view(key, index)
        else:
            assert index == "all"
        self.view["controlled_by_param"] -= self.view["controlled_by_param"].iloc[0]
        return self

    def _get_local_indices(self) -> pd.DataFrame:
        """Computes local from global indices.

        #cell_index, branch_index, comp_index
        0, 0, 0     -->     0, 0, 0 # 1st compartment of 1st branch of 1st cell
        0, 0, 1     -->     0, 0, 1 # 2nd compartment of 1st branch of 1st cell
        0, 1, 2     -->     0, 1, 0 # 1st compartment of 2nd branch of 1st cell
        0, 1, 3     -->     0, 1, 1 # 2nd compartment of 2nd branch of 1st cell
        1, 2, 4     -->     1, 0, 0 # 1st compartment of 1st branch of 2nd cell
        1, 2, 5     -->     1, 0, 1 # 2nd compartment of 1st branch of 2nd cell
        1, 3, 6     -->     1, 1, 0 # 1st compartment of 2nd branch of 2nd cell
        1, 3, 7     -->     1, 1, 1 # 2nd compartment of 2nd branch of 2nd cell
        """

        def reindex_a_by_b(df, a, b):
            df.loc[:, a] = df.groupby(b)[a].rank(method="dense").astype(int) - 1
            return df

        idcs = self.view[["cell_index", "branch_index", "comp_index"]]
        idcs = reindex_a_by_b(idcs, "branch_index", "cell_index")
        idcs = reindex_a_by_b(idcs, "comp_index", ["cell_index", "branch_index"])
        return idcs

    def __getitem__(self, index):
        return self.pointer._getitem(self, index)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def rotate(self, degrees: float, rotation_axis: str = "xy"):
        """Rotate jaxley modules clockwise. Used only for visualization.

        Args:
            degrees: How many degrees to rotate the module by.
            rotation_axis: Either of {`xy` | `xz` | `yz`}.
        """
        raise NotImplementedError(
            "Only entire `jx.Module`s or entire cells within a network can be rotated."
        )

    @property
    def shape(self) -> Tuple[int]:
        """Returns the number of elements currently in view.

        ```
        network.shape = (num_cells, num_branches, num_compartments)
        cell.shape = (num_branches, num_compartments)
        branch.shape = (num_compartments,)
        ```"""
        local_idcs = self._get_local_indices()
        return tuple(local_idcs.nunique())

    @property
    def xyzr(self) -> List[np.ndarray]:
        """Returns the xyzr entries of a branch, cell, or network.

        If called on a compartment or location, it will return the (x, y, z) of the
        center of the compartment.
        """
        idxs = self.view.global_branch_index.unique()
        if self.__class__.__name__ == "CompartmentView":
            loc = loc_of_index(
                self.view["global_comp_index"].to_numpy(),
                self.view["global_branch_index"].to_numpy(),
                self.pointer.nseg_per_branch,
            )
            return list(interpolate_xyz(loc, self.pointer.xyzr[idxs[0]]))
        else:
            return [self.pointer.xyzr[i] for i in idxs]

    def _append_multiple_synapses(
        self, pre_rows: pd.DataFrame, post_rows: pd.DataFrame, synapse_type: Synapse
    ):
        """Append multiple rows to the `self.edges` table.

        This is used, e.g. by `fully_connect` and `connect`.

        Args:
            pre_rows: The pre-synaptic compartments.
            post_rows: The post-synaptic compartments.
            synapse_type: The synapse to append.

        both `pre_rows` and `post_rows` can be obtained from self.view.
        """
        # Add synapse types to the module and infer their unique identifier.
        synapse_name = synapse_type._name
        index = len(self.pointer.edges)
        type_ind, is_new = self._infer_synapse_type_ind(synapse_name)
        if is_new:  # synapse is not known
            self._update_synapse_state_names(synapse_type)

        post_loc = loc_of_index(
            post_rows["global_comp_index"].to_numpy(),
            post_rows["global_branch_index"].to_numpy(),
            self.pointer.nseg_per_branch,
        )
        pre_loc = loc_of_index(
            pre_rows["global_comp_index"].to_numpy(),
            pre_rows["global_branch_index"].to_numpy(),
            self.pointer.nseg_per_branch,
        )

        # Define new synapses. Each row is one synapse.
        new_rows = dict(
            pre_locs=pre_loc,
            post_locs=post_loc,
            pre_branch_index=pre_rows["branch_index"].to_numpy(),
            post_branch_index=post_rows["branch_index"].to_numpy(),
            pre_cell_index=pre_rows["cell_index"].to_numpy(),
            post_cell_index=post_rows["cell_index"].to_numpy(),
            type=synapse_name,
            type_ind=type_ind,
            global_pre_comp_index=pre_rows["global_comp_index"].to_numpy(),
            global_post_comp_index=post_rows["global_comp_index"].to_numpy(),
            global_pre_branch_index=pre_rows["global_branch_index"].to_numpy(),
            global_post_branch_index=post_rows["global_branch_index"].to_numpy(),
        )

        # Update edges.
        self.pointer.edges = concat_and_ignore_empty(
            [self.pointer.edges, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

        indices = [idx for idx in range(index, index + len(pre_loc))]
        self._add_params_to_edges(synapse_type, indices)

    def _infer_synapse_type_ind(self, synapse_name: str) -> Tuple[int, bool]:
        """Return the unique identifier for every synapse type.

        Also returns a boolean indicating whether the synapse is already in the
        `module`.

        Used during `self._append_multiple_synapses`.

        Args:
            synapse_name: The name of the synapse.

        Returns:
            type_ind: Index referencing the synapse type in self.synapses.
            is_new_type: Whether the synapse is new to the module.
        """
        syn_names = self.pointer.synapse_names
        is_new_type = False if synapse_name in syn_names else True
        type_ind = len(syn_names) if is_new_type else syn_names.index(synapse_name)
        return type_ind, is_new_type

    def _add_params_to_edges(self, synapse_type: Synapse, indices: list):
        """Fills parameter and state columns of new synapses in the `edges` table.

        This method does not create new rows in the `.edges` table. It only fills
        columns of already existing rows.

        Used during `self._append_multiple_synapses`.

        Args:
            synapse_type: The synapse to append.
            indices: The indices of the synapses according to self.synapses.
        """
        # Add parameters and states to the `.edges` table.
        for key, param_val in synapse_type.synapse_params.items():
            self.pointer.edges.loc[indices, key] = param_val

        # Update synaptic state array.
        for key, state_val in synapse_type.synapse_states.items():
            self.pointer.edges.loc[indices, key] = state_val

    def _update_synapse_state_names(self, synapse_type: Synapse):
        """Update attributes with information about the synapses.

        Used during `self._append_multiple_synapses`.

        Args:
            synapse_type: The synapse to append.
        """
        # (Potentially) update variables that track meta information about synapses.
        self.pointer.synapse_names.append(synapse_type._name)
        self.pointer.synapse_param_names += list(synapse_type.synapse_params.keys())
        self.pointer.synapse_state_names += list(synapse_type.synapse_states.keys())
        self.pointer.synapses.append(synapse_type)


class GroupView(View):
    """GroupView (aka sectionlist).

    Unlike the standard `View` it sets `controlled_by_param` to
    0 for all compartments. This means that a group will be controlled by a single
    parameter (unless it is subclassed).
    """

    def __init__(
        self,
        pointer: Module,
        view: pd.DataFrame,
        childview: type,
        childview_keys: List[str],
    ):
        """Initialize group.

        Args:
            pointer: The module from which the group was created.
            view: The dataframe which defines the compartments, branches, and cells in
                the group.
            childview: An uninitialized view (e.g. `CellView`). Depending on the module,
                subclassing groups will return a different `View`. E.g., `net.group[0]`
                will return a `CellView`, whereas `cell.group[0]` will return a
                `BranchView`. The childview argument defines which view is created. We
                do not automatically infer this because that would force us to import
                `CellView`, `BranchView`, and `CompartmentView` in the `base.py` file.
            childview_keys: The names by which the group can be subclassed. Used to
                raise `KeyError` if one does, e.g. `net.group.branch(0)` (i.e. `.cell`
                is skipped).
        """
        self.childview_of_group = childview
        self.names_of_childview = childview_keys
        view["controlled_by_param"] = 0
        super().__init__(pointer, view)

    def __getattr__(self, key: str) -> View:
        """Subclass the group.

        This first checks whether the key that is used to subclass the view is allowed.
        For example, one cannot `net.group.branch(0)` but instead must use
        `net.group.cell("all").branch(0).` If this is valid, then it instantiates the
        correct `View` which had been passed to `__init__()`.

        Args:
            key: The key which is used to subclass the group.

        Return:
            View of the subclassed group.
        """
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key in self.names_of_childview:
            view = deepcopy(self.view)
            view["global_comp_index"] = view["comp_index"]
            view["global_branch_index"] = view["branch_index"]
            view["global_cell_index"] = view["cell_index"]
            return self.childview_of_group(self.pointer, view)
        else:
            raise KeyError(f"Key {key} not recognized.")

    def __getitem__(self, index):
        """Subclass the group with lazy indexing."""
        return self.pointer._getitem(self, index, self.names_of_childview[0])
