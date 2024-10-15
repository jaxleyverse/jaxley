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

        self.groups = {}

        self.nodes: Optional[pd.DataFrame] = None
        self._scope = "local"  # defaults to local scope
        self._in_view = None

        self.edges = pd.DataFrame(
            columns=[
                f"{scope}_{lvl}_index"
                for lvl in [
                    "pre_comp",
                    "pre_branch",
                    "pre_cell",
                    "post_comp",
                    "post_branch",
                    "post_cell",
                ]
                for scope in ["global", "local"]
            ]
            + ["pre_locs", "post_locs", "type", "type_ind"]
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

        # needs to be set at the end
        self.base = self

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
        nsegs = (
            self.nodes.groupby("global_branch_index")["global_comp_index"]
            .nunique()
            .to_numpy()
        )

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
        self.base.nodes.loc[self._in_view, ["x", "y", "z"]] = centers
        return centers, xyz

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channels)} different channels. Use `.show()` for details."

    def __str__(self):
        return f"jx.{type(self).__name__}"

    def __dir__(self):
        base_dir = object.__dir__(self)
        return sorted(base_dir + self.synapse_names + list(self.group_nodes.keys()))

    def _update_local_indices(self) -> pd.DataFrame:
        idx_cols = ["global_comp_index", "global_branch_index", "global_cell_index"]
        self.nodes.rename(
            columns={col.replace("global_", ""): col for col in idx_cols}, inplace=True
        )
        idcs = self.nodes[idx_cols]

        def reindex_a_by_b(df, a, b):
            df.loc[:, a] = df.groupby(b)[a].rank(method="dense").astype(int) - 1
            return df

        idcs = reindex_a_by_b(idcs, idx_cols[1], idx_cols[2])
        idcs = reindex_a_by_b(idcs, idx_cols[0], idx_cols[1:])
        idcs.columns = [col.replace("global", "local") for col in idx_cols]
        self.nodes[["local_comp_index", "local_branch_index", "local_cell_index"]] = (
            idcs[["local_comp_index", "local_branch_index", "local_cell_index"]]
        )
        # TODO: place indices at the front of the dataframe

    def _reformat_index(self, idx):
        idx = np.array([], dtype=int) if idx is None else idx
        idx = np.array([idx]) if isinstance(idx, (int, np.int64)) else idx
        idx = np.array(idx) if isinstance(idx, (list, range)) else idx
        idx = np.arange(len(self._in_view) + 1)[idx] if isinstance(idx, slice) else idx
        if isinstance(idx, str):
            assert idx == "all", "Only 'all' is allowed"
            idx = np.arange(len(self._in_view) + 1)
        assert isinstance(idx, np.ndarray), "Invalid type"
        assert idx.dtype == np.int64, "Invalid dtype"
        return idx.reshape(-1)

    def _set_controlled_by_param(self, key):
        if key in ["comp", "branch", "cell"]:
            self.nodes["controlled_by_param"] = self.nodes[f"global_{key}_index"]
            self.edges["controlled_by_param"] = self.edges[f"global_pre_{key}_index"]
        else:
            self.nodes["controlled_by_param"] = 0
            self.edges["controlled_by_param"] = 0

    def at(self, idx, sorted=False):
        idx = self._reformat_index(idx)
        new_indices = self._in_view[idx]
        new_indices = np.sort(new_indices) if sorted else new_indices
        return View(self, at=new_indices)

    def set_scope(self, scope):
        self._scope = scope

    def scope(self, scope):
        view = self.view
        view.set_scope(scope)
        return view

    def _at_level(self, level: str, idx):
        idx = self._reformat_index(idx)
        where = self.nodes[self._scope + f"_{level}_index"].isin(idx)
        inds = np.where(where)[0]
        view = self.at(inds)
        view._set_controlled_by_param(level)
        return view

    def cell(self, idx):
        return self._at_level("cell", idx)

    def branch(self, idx):
        return self._at_level("branch", idx)

    def comp(self, idx):
        return self._at_level("comp", idx)

    def loc(self, at: float):
        comp_edges = np.linspace(0, 1, self.base.nseg + 1)
        idx = np.digitize(at, comp_edges)
        view = self.comp(idx)
        return view

    def __getattr__(self, key):
        if key.startswith("__"):
            return super().__getattribute__(key)

        if key in self.base.groups:
            view = self.at(self.groups[key]) if key in self.groups else self.at(None)
            view._set_controlled_by_param(key)
            return view

        if key in [c._name for c in self.base.channels]:
            channel_names = [c._name for c in self.channels]
            inds = self.nodes.index[self.nodes[key]].to_numpy()
            view = self.at(inds) if key in channel_names else self.at(None)
            view._set_controlled_by_param(key)
            return view

        if key in self.base.synapse_names:
            # if the same 2 nodes are connected by 2 different synapses,
            # module.SynapseA.edges will still contain both synapses
            # since view filters based on index ONLY. Returning only the row
            # that contains SynapseA is not possible currently. For setting
            # synapse parameters this has no effect however.
            has_syn = self.edges["type"] == key
            where = has_syn, ["global_pre_comp_index", "global_post_comp_index"]
            comp_inds_in_view = pd.unique(self.edges.loc[where].values.ravel("K"))
            inds = np.where(self.nodes["global_comp_index"].isin(comp_inds_in_view))[0]
            view = self.at(inds) if key in self.synapse_names else self.at(None)
            view._set_controlled_by_param(key)
            return view

    def _iter_level(self, level):
        col = self._scope + f"_{level}_index"
        idxs = self.nodes[col].unique()
        for idx in idxs:
            yield self._at_level(level, idx)

    @property
    def cells(self):
        yield from self._iter_level("cell")

    @property
    def branches(self):
        yield from self._iter_level("branch")

    @property
    def comps(self):
        yield from self._iter_level("comp")

    @property
    def shape(self) -> Tuple[int]:
        """Returns the number of submodules contained in a module.

        ```
        network.shape = (num_cells, num_branches, num_compartments)
        cell.shape = (num_branches, num_compartments)
        branch.shape = (num_compartments,)
        ```"""
        cols = ["global_cell_index", "global_branch_index", "global_comp_index"]
        raw_shape = self.nodes[cols].nunique().to_list()

        # ensure (net.shape -> dim=3, cell.shape -> dim=2, branch.shape -> dim=1, comp.shape -> dim=0)
        levels = ["network", "cell", "branch", "comp"]
        module = self.base.__class__.__name__.lower()
        module = "comp" if module == "compartment" else module
        shape = tuple(raw_shape[levels.index(module) :])
        return shape

    def copy(self, reset_index=False, as_module=False):
        view = deepcopy(self)
        # TODO: add reset_index, i.e. for parents, nodes, edges etc. such that they
        # start from 0/-1 and are contiguous
        if as_module:
            # TODO: initialize a new module with the same attributes
            pass
        return view

    @property
    def view(self):
        return View(self, self._in_view)

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
        nodes = self.nodes.copy()  # prevents this from being edited

        cols = []
        inds = ["comp_index", "branch_index", "cell_index"]
        scopes = ["local", "global"]
        cols += (
            [f"{scope}_{idx}" for idx in inds for scope in scopes] if indices else []
        )
        cols += [ch._name for ch in self.channels] if channel_names else []
        cols += (
            sum([list(ch.channel_params) for ch in self.channels], []) if params else []
        )
        cols += (
            sum([list(ch.channel_states) for ch in self.channels], []) if states else []
        )

        if not param_names is None:
            cols = (
                [c for c in cols if c in param_names] if params else list(param_names)
            )

        return nodes[cols]

    def copy(self, reset_index=False, as_module=False):
        view = deepcopy(self)
        # TODO: add reset_index, i.e. for parents, nodes, edges etc. such that they
        # start from 0/-1 and are contiguous
        if as_module:
            # TODO: initialize a new module with the same attributes
            pass
        return view

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
        if key in self.nodes.columns:
            not_nan = ~self.nodes[key].isna()
            self.base.nodes.loc[self._in_view[not_nan], key] = val
        elif key in self.edges.columns:
            not_nan = ~self.edges[key].isna()
            self.base.edges.loc[self._edges_in_view[not_nan], key] = val
        else:
            raise KeyError(f"Key '{key}' not found in nodes or edges")

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
        # Note: `data_set` does not support arrays for `val`.
        if key in self.nodes.columns:
            not_nan = ~self.nodes[key].isna()
            added_param_state = [
                {
                    "indices": np.atleast_2d(self._in_view[not_nan]),
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

    # TODO: MAKE SURE THIS WORKS
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
        branch_indices = pd.unique(view["global_branch_index"])

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
        integer_cols = ["global_comp_index", "global_branch_index", "global_cell_index"]
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
        all_nodes["global_comp_index"] = np.arange(len(all_nodes))

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
            self.allow_make_trainable
        ), "network.cell('all').make_trainable() is not supported. Use a for-loop over cells."

        data = self.nodes if key in self.nodes.columns else None
        data = self.edges if key in self.edges.columns else data
        assert data is not None, f"Key '{key}' not found in nodes or edges"
        not_nan = ~data[key].isna()
        data = data.loc[not_nan]
        assert (
            len(data) > 0
        ), "No settable parameters found in the selected compartments."

        grouped_view = data.groupby("controlled_by_param")
        # Because of this `x.index.values` we cannot support `make_trainable()` on
        # the module level for synapse parameters (but only for `SynapseView`).
        inds_of_comps = list(
            grouped_view.apply(lambda x: x.index.values, include_groups=False)
        )
        indices_per_param = jnp.stack(inds_of_comps)
        # Sorted inds are only used to infer the correct starting values.
        param_vals = jnp.asarray(
            [data.loc[inds, key].to_numpy() for inds in inds_of_comps]
        )

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
        self.base.trainable_params.append({key: new_params})
        self.base.indices_set_by_trainables.append(indices_per_param)
        if verbose:
            print(
                f"Number of newly added trainable parameters: {num_created_parameters}. Total number of trainable parameters: {self.num_trainable_params}"
            )

    # TODO: MAKE THIS WORK FOR VIEW?
    def delete_trainables(self):
        """Removes all trainable parameters from the module."""
        self.base.indices_set_by_trainables = []
        self.base.trainable_params = []
        self.base.num_trainable_params = 0

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
        self.base.groups[group_name] = self._in_view

    def get_parameters(self) -> List[Dict[str, jnp.ndarray]]:
        """Get all trainable parameters.

        The returned parameters should be passed to `jx.integrate(..., params=params).

        Returns:
            A list of all trainable parameters in the form of
                [{"gNa": jnp.array([0.1, 0.2, 0.3])}, ...].
        """
        return self.trainable_params

    # TODO: ENSURE THIS WORKS FOR VIEW?
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

    # TODO: ENSURE THIS WORKS FOR VIEW?
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

    # TODO: ENSURE THIS WORKS FOR VIEW?
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

    # TODO: ENSURE THIS WORKS FOR VIEW?
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
                "global_comp_index"
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

    # TODO: ENSURE THIS WORKS
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

    def record(self, state, verbose=True):
        new_recs = pd.DataFrame(self._in_view, columns=["rec_index"])
        new_recs["state"] = state
        self.base.recordings = pd.concat([self.base.recordings, new_recs])
        has_duplicates = self.base.recordings.duplicated()
        self.base.recordings = self.base.recordings.loc[~has_duplicates]
        if verbose:
            print(
                f"Added {len(self._in_view)-sum(has_duplicates)} recordings. See `.recordings` for details."
            )

    # TODO: MAKE THIS WORK FOR VIEW?
    def delete_recordings(self):
        """Removes all recordings from the module."""
        self.base.recordings = pd.DataFrame().from_dict({})

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
        self._external_input("i", current, verbose=verbose)

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
        verbose: bool = True,
    ):
        values = values if values.ndim == 2 else jnp.expand_dims(values, axis=0)
        batch_size = values.shape[0]
        num_inserted = len(self._in_view)
        is_multiple = num_inserted == batch_size
        values = (
            values if is_multiple else jnp.repeat(values, len(self._in_view), axis=0)
        )
        assert batch_size in [
            1,
            num_inserted,
        ], "Number of comps and stimuli do not match."

        if key in self.base.externals.keys():
            self.base.externals[key] = jnp.concatenate(
                [self.base.externals[key], values]
            )
            self.base.external_inds[key] = jnp.concatenate(
                [self.base.external_inds[key], self._in_view]
            )
        else:
            self.base.externals[key] = values
            self.base.external_inds[key] = self._in_view

        if verbose:
            print(
                f"Added {num_inserted} external_states. See `.externals` for details."
            )

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
        num_inserted = len(self._in_view)
        is_multiple = num_inserted == batch_size
        state_array = (
            state_array if is_multiple else jnp.repeat(state_array, len(view), axis=0)
        )
        assert batch_size in [
            1,
            num_inserted,
        ], "Number of comps and clamps do not match."

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

    # TODO: MAKE THIS WORK FOR VIEW?
    def delete_stimuli(self):
        """Removes all stimuli from the module."""
        self.base.externals.pop("i", None)
        self.base.external_inds.pop("i", None)

    # TODO: MAKE THIS WORK FOR VIEW?
    def delete_clamps(self, state_name: str):
        """Removes all clamps of the given state from the module."""
        self.base.externals.pop(state_name, None)
        self.base.external_inds.pop(state_name, None)

    def insert(self, channel: Channel):
        """Insert a channel into the module.

        Args:
            channel: The channel to insert."""
        name = channel._name

        # Channel does not yet exist in the `jx.Module` at all.
        if name not in [c._name for c in self.base.channels]:
            self.base.channels.append(channel)
            self.base.nodes[name] = (
                False  # Previous columns do not have the new channel.
            )

        if channel.current_name not in self.base.membrane_current_names:
            self.base.membrane_current_names.append(channel.current_name)

        # Add a binary column that indicates if a channel is present.
        self.base.nodes.loc[self._in_view, name] = True

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_params:
            self.base.nodes.loc[self._in_view, key] = channel.channel_params[key]

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_states:
            self.base.nodes.loc[self._in_view, key] = channel.channel_states[key]

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
        indices = channel_nodes["global_comp_index"].to_numpy()
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
            indices = channel_nodes.loc[channel_nodes[name]][
                "global_comp_index"
            ].to_numpy()

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
        if "comp" in type.lower():
            return plot_comps(
                self, self.nodes, dims=dims, ax=ax, col=col, **morph_plot_kwargs
            )
        if "morph" in type.lower():
            return plot_morph(
                self, self.nodes, dims=dims, ax=ax, col=col, **morph_plot_kwargs
            )

        coords = []
        branches_inds = self.view._branches_in_view
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
        inds_branch = self.nodes.groupby("global_branch_index")[
            "global_comp_index"
        ].apply(list)
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
        indizes = self.nodes["global_branch_index"].unique()
        for i in indizes:
            self.base.xyzr[i][:, :3] += np.array([x, x, y])
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
        # Test if any coordinate values are NaN which would greatly affect moving
        if np.any(np.concatenate(self.xyzr, axis=0)[:, :3] == np.nan):
            raise ValueError(
                "NaN coordinate values detected. Shift amounts cannot be computed. Please run compute_xyzr() or assign initial coordinate values."
            )

        indizes = self.nodes["global_branch_index"].unique()
        move_by = (
            np.array([x, y, z]).T - self.xyzr[0][0, :3]
        )  # move with respect to root idx

        for idx in indizes:
            self.base.xyzr[idx][:, :3] += move_by
        if update_nodes:
            self._update_nodes_with_xyz()

    def rotate(
        self, degrees: float, rotation_axis: str = "xy", update_nodes: bool = True
    ):
        """Rotate jaxley modules clockwise. Used only for visualization.

        This function is used only for visualization. It does not affect the simulation.

        Args:
            degrees: How many degrees to rotate the module by.
            rotation_axis: Either of {`xy` | `xz` | `yz`}.
        """
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
        indizes = self.nodes["global_branch_index"].unique()
        for i in indizes:
            rot = np.dot(rotation_matrix, self.base.xyzr[i][:, dims].T).T
            self.base.xyzr[i][:, dims] = rot
        if update_nodes:
            self._update_nodes_with_xyz()

    def __getitem__(self, index):
        levels = ["network", "cell", "branch", "comp"]
        module = self.base.__class__.__name__.lower()  #
        module = "comp" if module == "compartment" else module

        children = levels[levels.index(module) + 1 :]
        index = index if isinstance(index, tuple) else (index,)
        view = self
        for i, child in enumerate(children):
            view = view._at_level(child, index[i])
        return view


class View(Module):
    def __init__(self, pointer, at=None):
        # attrs with a static view
        self._scope = pointer._scope
        self.base = pointer.base
        self.initialized_morph = pointer.initialized_morph
        self.initialized_syns = pointer.initialized_syns
        self.allow_make_trainable = pointer.allow_make_trainable

        # attrs affected by view
        self.nseg = pointer.nseg
        self._in_view = pointer._in_view if at is None else at

        self.nodes = pointer.nodes.loc[self._in_view]
        self.edges = pointer.edges.loc[self._edges_in_view]
        self.xyzr = self._xyzr_in_view(pointer)
        self.nseg = 1 if len(self.nodes) == 1 else pointer.nseg
        self.total_nbranches = len(self._branches_in_view)
        self.nbranches_per_cell = self._nbranches_per_cell_in_view()
        self.cumsum_nbranches = np.cumsum(self.nbranches_per_cell)
        self.comb_branches_in_each_level = pointer.comb_branches_in_each_level
        self.branch_edges = pointer.branch_edges.loc[self._branch_edges_in_view]

        self.synapse_names = np.unique(self.edges["type"]).tolist()
        self.synapses, self.synapse_param_names, self.synapse_state_names = (
            self._synapses_in_view(pointer)
        )

        if pointer.recordings.empty:
            self.recordings = pd.DataFrame()
        else:
            self.recordings = pointer.recordings.loc[
                pointer.recordings["rec_index"].isin(self._comps_in_view)
            ]

        self.channels = self._channels_in_view(pointer)
        self.membrane_current_names = [c._name for c in self.channels]

        self.indices_set_by_trainables, self.trainable_params = (
            self._trainables_in_view()
        )
        self.num_trainable_params = np.sum(
            [len(inds) for inds in self.indices_set_by_trainables]
        )

        self.nseg_per_branch = pointer.base.nseg_per_branch[self._branches_in_view]
        self.comb_parents = self.base.comb_parents[self._branches_in_view]
        self.externals, self.external_inds = self._externals_in_view()
        self.groups = {
            k: np.intersect1d(v, self._in_view) for k, v in pointer.groups.items()
        }

        # TODO:
        # self.debug_states

        if len(self.nodes) == 0:
            raise ValueError("Nothing in view. Check your indices.")

    def _externals_in_view(self):
        externals_in_view = {}
        external_inds_in_view = []
        for (name, inds), data in zip(
            self.base.external_inds.items(), self.base.externals.values()
        ):
            in_view = np.isin(inds, self._in_view)
            inds_in_view = inds[in_view]
            if len(inds_in_view) > 0:
                externals_in_view[name] = data[in_view]
                external_inds_in_view.append(inds_in_view)
        return externals_in_view, external_inds_in_view

    def _trainables_in_view(self):
        trainable_inds = self.base.indices_set_by_trainables
        trainable_inds = (
            np.unique(np.hstack([inds.reshape(-1) for inds in trainable_inds]))
            if len(trainable_inds) > 0
            else []
        )
        trainable_inds_in_view = np.intersect1d(trainable_inds, self._in_view)

        índices_set_by_trainables_in_view = []
        trainable_params_in_view = []
        for inds, params in zip(
            self.base.indices_set_by_trainables, self.base.trainable_params
        ):
            in_view = np.isin(inds, trainable_inds_in_view)

            completely_in_view = in_view.all(axis=1)
            índices_set_by_trainables_in_view.append(inds[completely_in_view])
            trainable_params_in_view.append(
                {k: v[completely_in_view] for k, v in params.items()}
            )

            partially_in_view = in_view.any(axis=1) & ~completely_in_view
            índices_set_by_trainables_in_view.append(
                inds[partially_in_view][in_view[partially_in_view]]
            )
            trainable_params_in_view.append(
                {k: v[partially_in_view] for k, v in params.items()}
            )

        índices_set_by_trainables_in_view = [
            inds for inds in índices_set_by_trainables_in_view if len(inds) > 0
        ]
        trainable_params_in_view = [
            p for p in trainable_params_in_view if len(next(iter(p.values()))) > 0
        ]
        return índices_set_by_trainables_in_view, trainable_params_in_view

    def _channels_in_view(self, pointer):
        names = [name._name for name in pointer.channels]
        channel_in_view = self.nodes[names].any(axis=0)
        channel_in_view = channel_in_view[channel_in_view].index
        return [c for c in pointer.channels if c._name in channel_in_view]

    def _synapses_in_view(self, pointer):
        viewed_synapses = []
        viewed_params = []
        viewed_states = []
        if not pointer.synapses is None:
            for syn in pointer.synapses:
                if syn is not None:  # needed for recurive viewing
                    in_view = syn._name in self.synapse_names
                    viewed_synapses += (
                        [syn] if in_view else [None]
                    )  # padded with None to keep indices consistent
                    viewed_params += list(syn.synapse_params.keys()) if in_view else []
                    viewed_states += list(syn.synapse_states.keys()) if in_view else []

        return viewed_synapses, viewed_params, viewed_states

    def _nbranches_per_cell_in_view(self):
        cell_nodes = self.nodes.groupby("global_cell_index")
        return cell_nodes["global_branch_index"].nunique().to_numpy()

    def _xyzr_in_view(self, pointer):
        viewed_branch_inds = self._branches_in_view
        prev_branch_inds = pointer._branches_in_view
        if prev_branch_inds is None:
            xyzr = pointer.xyzr.copy()  # copy to prevent editing original
        else:
            branches2keep = np.isin(prev_branch_inds, viewed_branch_inds)
            branch_inds2keep = np.where(branches2keep)[0]
            xyzr = [pointer.xyzr[i] for i in branch_inds2keep].copy()

        # Currently viewing with `.loc` will show the closest compartment
        # rather than the actual loc along the branch!
        viewed_nseg_for_branch = self.nodes.groupby("global_branch_index").size()
        incomplete_inds = np.where(viewed_nseg_for_branch != self.base.nseg)[0]
        incomplete_branch_inds = viewed_branch_inds[incomplete_inds]

        # TODO: FIX THIS
        # cond = self.nodes["global_branch_index"].isin(incomplete_branch_inds)
        # interp_inds = self.nodes.loc[cond]
        # local_inds_per_branch = interp_inds.groupby("global_branch_index")[
        #     "local_comp_index"
        # ]
        # locs = [
        #     loc_of_index(inds.to_numpy(), self. self.nseg_per_branch[i])
        #     for _, inds in local_inds_per_branch
        # ]

        # for i, loc in zip(incomplete_inds, locs):
        #     xyzr[i] = interpolate_xyz(loc, xyzr[i]).T
        return xyzr

    # needs abstract method to allow init of View
    # forward to self.base for now
    def _init_morph_jax_spsolve(self):
        return self.base._init_morph_jax_spsolve()

    def _init_morph_jaxley_spsolve(self):
        return self.base._init_morph_jax_spsolve()

    @property
    def _nodes_in_view(self):
        return self._in_view

    @property
    def _branches_in_view(self):
        return self.nodes["global_branch_index"].unique()

    @property
    def _cells_in_view(self):
        return self.nodes["global_cell_index"].unique()

    @property
    def _comps_in_view(self):
        return self.nodes["global_comp_index"].unique()

    @property
    def _branch_edges_in_view(self):
        incl_branches = self.nodes["global_branch_index"].unique()
        pre = self.base.branch_edges["parent_branch_index"].isin(incl_branches)
        post = self.base.branch_edges["child_branch_index"].isin(incl_branches)
        viewed_branch_inds = self.base.branch_edges.index.to_numpy()[pre & post]
        return viewed_branch_inds

    @property
    def _edges_in_view(self):
        incl_comps = self.nodes["global_comp_index"].unique()
        pre = self.base.edges["global_pre_comp_index"].isin(incl_comps).to_numpy()
        post = self.base.edges["global_post_comp_index"].isin(incl_comps).to_numpy()
        viewed_edge_inds = self.base.edges.index.to_numpy()[(pre & post).flatten()]
        return viewed_edge_inds

    def __getattr__(self, name):
        # Delegate attribute access to the pointer if not found in View
        return getattr(self.pointer, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


class GroupView:
    # KEEP AROUND FOR NOW TO NOT BREAK EXISTING CODE
    pass
