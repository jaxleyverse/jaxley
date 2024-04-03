import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from math import pi
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import vmap
from jax.lax import ScatterDimensionNumbers, scatter_add
from matplotlib.axes import Axes

from jaxley.channels import Channel
from jaxley.solver_voltage import step_voltage_explicit, step_voltage_implicit
from jaxley.synapses import Synapse
from jaxley.utils.cell_utils import (
    _compute_index_of_child,
    _compute_num_children,
    compute_levels,
    flip_comp_indices,
    interpolate_xyz,
    loc_of_index,
)
from jaxley.utils.plot_utils import plot_morph


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
        self.comb_branches_in_each_level: List[jnp.ndarray] = [jnp.asarray([0])]

        self.initialized_morph: bool = False
        self.initialized_syns: bool = False

        # List of all types of `jx.Synapse`s.
        self.synapses: List = []
        self.synapse_param_names = []
        self.synapse_state_names = []
        self.synapse_names = []

        # List of types of all `jx.Channel`s.
        self.channels: List[Channel] = []

        # For trainable parameters.
        self.indices_set_by_trainables: List[jnp.ndarray] = []
        self.trainable_params: List[Dict[str, jnp.ndarray]] = []
        self.allow_make_trainable: bool = True
        self.num_trainable_params: int = 0

        # For recordings.
        self.recordings: pd.DataFrame = pd.DataFrame().from_dict({})

        # For stimuli.
        self.currents: Optional[jnp.ndarray] = None
        self.current_inds: pd.DataFrame = pd.DataFrame().from_dict({})

        # x, y, z coordinates and radius.
        self.xyzr: List[np.ndarray] = []

    def _update_nodes_with_xyz(self):
        """Add xyz coordinates to nodes."""
        loc = np.linspace(0.5 / self.nseg, 1 - 0.5 / self.nseg, self.nseg)
        xyz = (
            [interpolate_xyz(loc, xyzr).T for xyzr in self.xyzr]
            if len(loc) > 0
            else [self.xyzr]
        )
        idcs = self.nodes["comp_index"]
        self.nodes.loc[idcs, ["x", "y", "z"]] = np.vstack(xyz)

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channels)} different channels. Use `.show()` for details."

    def __str__(self):
        return f"jx.{type(self).__name__}"

    def __dir__(self):
        base_dir = object.__dir__(self)
        return sorted(base_dir + self.synapse_names + list(self.group_nodes.keys()))

    def _append_params_and_states(self, param_dict, state_dict):
        """Insert the default params of the module (e.g. radius, length).

        This is run at `__init__()`. It does not deal with channels.
        """
        for param_name, param_value in param_dict.items():
            self.nodes[param_name] = param_value
        for state_name, state_value in state_dict.items():
            self.nodes[state_name] = state_value

    def _gather_channels_from_constituents(self, constituents: List) -> None:
        """Modify `self.channels` and `self.nodes` with channel info from constituents.

        This is run at `__init__()`. It takes all branches of constituents (e.g.
        of all branches when the are assembled into a cell) and adds columns to
        `.nodes` for the relevant channels.
        """
        for module in constituents:
            for channel in module.channels:
                if channel._name not in [c._name for c in self.channels]:
                    self.channels.append(channel)
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
            inds = flip_comp_indices(inds, self.nseg)  # See #305
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
    ):
        """Print detailed information about the Module or a view of it."""
        return self._show(
            self.nodes, param_names, indices, params, states, channel_names
        )

    def _show(
        self,
        view,
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

    @abstractmethod
    def init_conds(self, params):
        """Initialize coupling conductances.

        Args:
            params: Conductances and morphology parameters, not yet including
                coupling conductances.
        """
        raise NotImplementedError

    def _append_channel_to_nodes(self, view, channel: "jx.Channel"):
        """Adds channel nodes from constituents to `self.channel_nodes`."""
        name = channel._name

        # Channel does not yet exist in the `jx.Module` at all.
        if name not in [c._name for c in self.channels]:
            self.channels.append(channel)
            self.nodes[name] = False  # Previous columns do not have the new channel.

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
        view,
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

    def add_to_group(self, group_name):
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

    def _add_to_group(self, group_name, view):
        if group_name in self.group_nodes.keys():
            view = pd.concat([self.group_nodes[group_name], view])
        self.group_nodes[group_name] = view

    def get_parameters(self):
        """Get all trainable parameters.

        The returned parameters should be passed to `jx.integrate(..., params=params).
        """
        return self.trainable_params

    def get_all_parameters(self, trainable_params):
        """Return all parameters (and coupling conductances) needed to simulate.

        This is done by first obtaining the current value of every parameter (not only
        the trainable ones) and then replacing the trainable ones with the value
        in `trainable_params()`. This function is run within `jx.integrate()`.
        """
        params = {}
        for key in ["radius", "length", "axial_resistivity", "capacitance"]:
            params[key] = self.jaxnodes[key]

        for channel in self.channels:
            for channel_params in list(channel.channel_params.keys()):
                params[channel_params] = self.jaxnodes[channel_params]

        for synapse_params in self.synapse_param_names:
            params[synapse_params] = self.jaxedges[synapse_params]

        # Override with those parameters set by `.make_trainable()`.
        for parameter in trainable_params:
            key = parameter["key"]
            inds = parameter["indices"]
            if key not in self.synapse_param_names:
                inds = flip_comp_indices(parameter["indices"], self.nseg)  # See #305
            set_param = parameter["val"]
            if key in list(params.keys()):  # Only parameters, not initial states.
                params[key] = params[key].at[inds].set(set_param[:, None])

        # Compute conductance params and append them.
        cond_params = self.init_conds(params)
        for key in cond_params:
            params[key] = cond_params[key]

        return params

    @property
    def initialized(self):
        """Whether the `Module` is ready to be solved or not."""
        return self.initialized_morph and self.initialized_syns

    def initialize(self):
        """Initialize the module."""
        self.init_morph()
        return self

    def init_states(self) -> None:
        """Initialize all mechanisms in their steady state.

        This considers the voltages and parameters of each compartment."""
        # Update states of the channels.
        channel_nodes = self.nodes

        for channel in self.channels:
            name = channel._name
            indices = channel_nodes.loc[channel_nodes[name]]["comp_index"].to_numpy()
            indices = flip_comp_indices(indices, self.nseg)  # See #305
            voltages = channel_nodes.loc[indices, "v"].to_numpy()

            channel_param_names = list(channel.channel_params.keys())
            channel_params = {}
            for p in channel_param_names:
                channel_params[p] = channel_nodes[p][indices].to_numpy()

            init_state = channel.init_state(voltages, channel_params)

            # `init_state` might not return all channel states. Only the ones that are
            # returned are updated here.
            for key, val in init_state.items():
                self.nodes.loc[indices, key] = val

    def record(self, state: str = "v", verbose: bool = True):
        """Insert a recording into the compartment."""
        view = deepcopy(self.nodes)
        view["state"] = state
        recording_view = view[["comp_index", "state"]]
        recording_view = recording_view.rename(columns={"comp_index": "rec_index"})
        self._record(recording_view, verbose=verbose)

    def _record(self, view, verbose: bool = True):
        self.recordings = pd.concat([self.recordings, view], ignore_index=True)
        if verbose:
            num_comps = "ALL(!)" if len(view) == len(self.nodes) else len(view)
            warning = f"Added {num_comps} compartments to recordings. If this was not intended, run `delete_recordings`."
            if len(view) > 1:
                warnings.warn(warning)
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
        """
        self._stimulate(current, self.nodes, verbose=verbose)

    def _stimulate(self, current, view, verbose: bool = True):

        current = current if current.ndim == 2 else jnp.expand_dims(current, axis=0)
        batch_size = current.shape[0]
        is_multiple = len(view) == batch_size
        current = current if is_multiple else jnp.repeat(current, len(view), axis=0)
        assert batch_size in [1, len(view)], "Number of comps and stimuli do not match."

        if self.currents is not None:
            self.currents = jnp.concatenate([self.currents, current])
        else:
            self.currents = current
        self.current_inds = pd.concat([self.current_inds, view])

        if verbose:
            num_comps = "ALL(!)" if len(view) == len(self.nodes) else len(view)
            warning = f"Added stimuli to {num_comps} compartments. If this was not intended, run `delete_stimuli`."
            if len(view) > 1:
                warnings.warn(warning)
            print(f"Added {len(view)} stimuli. See `.currents` for details.")

    def data_stimulate(
        self,
        current,
        data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
        verbose: bool = False,
    ):
        """Insert a stimulus into the module within jit (or grad).

        Args:
            verbose: Whether or not to print the number of inserted stimuli. `False`
                by default because this method is meant to be jitted."""
        return self._data_stimulate(current, data_stimuli, self.nodes, verbose=verbose)

    def _data_stimulate(
        self,
        current,
        data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]],
        view,
        verbose: bool = False,
    ):
        current = current if current.ndim == 2 else jnp.expand_dims(current, axis=0)
        batch_size = current.shape[0]
        is_multiple = len(view) == batch_size
        current = current if is_multiple else jnp.repeat(current, len(view), axis=0)
        assert batch_size in [1, len(view)], "Number of comps and stimuli do not match."

        if data_stimuli is not None:
            currents = data_stimuli[0]
            inds = data_stimuli[1]
        else:
            currents = None
            inds = pd.DataFrame().from_dict({})

        # Same as in `.stimulate()`.
        if currents is not None:
            currents = jnp.concatenate([currents, current])
        else:
            currents = current
        inds = pd.concat([inds, view])

        if verbose:
            num_comps = "ALL(!)" if len(view) == len(self.nodes) else len(view)
            warning = f"Added stimuli to {num_comps} compartments."
            if len(view) > 1:
                warnings.warn(warning)
            print(f"Added {len(view)} stimuli.")

        return (currents, inds)

    def delete_stimuli(self):
        """Removes all stimuli from the module."""
        self.currents = None
        self.current_inds = pd.DataFrame().from_dict({})

    def insert(self, channel):
        """Insert a channel."""
        self._insert(channel, self.nodes)

    def _insert(self, channel, view):
        self._append_channel_to_nodes(view, channel)

    def init_syns(self, connectivities):
        self.initialized_syns = True

    def init_morph(self):
        self.initialized_morph = True

    def step(
        self,
        u,
        delta_t,
        i_inds,
        i_current,
        params: Dict[str, jnp.ndarray],
        solver: str = "bwd_euler",
        tridiag_solver: str = "stone",
    ):
        """One step of solving the Ordinary Differential Equation."""
        voltages = u["v"]

        # Parameters have to go in here.
        u, (v_terms, const_terms) = self._step_channels(
            u, delta_t, self.channels, self.nodes, params
        )

        # External input.
        i_ext = self._get_external_input(
            voltages, i_inds, i_current, params["radius"], params["length"]
        )

        # Step of the synapse.
        u, (syn_v_terms, syn_const_terms) = self._step_synapse(
            u,
            self.synapses,
            params,
            delta_t,
            self.edges,
        )

        # Voltage steps.
        cm = params["capacitance"]  # Abbreviation.
        if solver == "bwd_euler":
            new_voltages = step_voltage_implicit(
                voltages=voltages,
                voltage_terms=(v_terms + syn_v_terms) / cm,
                constant_terms=(const_terms + i_ext + syn_const_terms) / cm,
                coupling_conds_bwd=params["coupling_conds_bwd"],
                coupling_conds_fwd=params["coupling_conds_fwd"],
                summed_coupling_conds=params["summed_coupling_conds"],
                branch_cond_fwd=params["branch_conds_fwd"],
                branch_cond_bwd=params["branch_conds_bwd"],
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                branches_in_each_level=self.comb_branches_in_each_level,
                tridiag_solver=tridiag_solver,
                delta_t=delta_t,
            )
        else:
            new_voltages = step_voltage_explicit(
                voltages,
                (v_terms + syn_v_terms) / cm,
                (const_terms + i_ext + syn_const_terms) / cm,
                coupling_conds_bwd=params["coupling_conds_bwd"],
                coupling_conds_fwd=params["coupling_conds_fwd"],
                branch_cond_fwd=params["branch_conds_fwd"],
                branch_cond_bwd=params["branch_conds_bwd"],
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                delta_t=delta_t,
            )

        u["v"] = new_voltages.ravel(order="C")

        return u

    def _step_channels(
        self,
        states,
        delta_t,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ):
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
    ):
        """One integration step of the channels."""
        voltages = states["v"]

        # Update states of the channels.
        for channel in channels:
            name = channel._name
            channel_param_names = list(channel.channel_params.keys())
            channel_state_names = list(channel.channel_states.keys())
            indices = channel_nodes.loc[channel_nodes[name]]["comp_index"].to_numpy()
            indices = flip_comp_indices(indices, self.nseg)  # See #305

            channel_params = {}
            for p in channel_param_names:
                channel_params[p] = params[p][indices]
            channel_params["radius"] = params["radius"][indices]
            channel_params["length"] = params["length"][indices]
            channel_params["axial_resistivity"] = params["axial_resistivity"][indices]

            channel_states = {}
            for s in channel_state_names:
                channel_states[s] = states[s][indices]

            for channel_for_current in channels:
                name_for_current = channel_for_current._name
                channel_states[f"{name_for_current}_current"] = states[
                    f"{name_for_current}_current"
                ]

            states_updated = channel.update_states(
                channel_states, delta_t, voltages[indices], channel_params
            )
            # Rebuild state. This has to be done within the loop over channels to allow
            # multiple channels which modify the same state.
            for key, val in states_updated.items():
                states[key] = states[key].at[indices].set(val)

        return states

    def _channel_currents(
        self,
        states,
        delta_t,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ):
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
        for channel in channels:
            name = channel._name
            channel_param_names = list(channel.channel_params.keys())
            channel_state_names = list(channel.channel_states.keys())
            indices = channel_nodes.loc[channel_nodes[name]]["comp_index"].to_numpy()
            indices = flip_comp_indices(indices, self.nseg)  # See #305

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

            # Same the current (for the unperturbed voltage) as a state that will
            # also be passed to the state update.
            states[f"{name}_current"] = membrane_currents[0]

        return states, (voltage_terms, constant_terms)

    def _step_synapse(
        self,
        u,
        syn_channels,
        params,
        delta_t,
        edges,
    ):
        """One step of integration of the channels.

        `Network` overrides this method (because it actually has synapses), whereas
        `Compartment`, `Branch`, and `Cell` do not override this.
        """
        voltages = u["v"]
        return u, (jnp.zeros_like(voltages), jnp.zeros_like(voltages))

    def _synapse_currents(
        self, states, syn_channels, params, delta_t, edges: pd.DataFrame
    ):
        return states, (None, None)

    @staticmethod
    def _get_external_input(
        voltages: jnp.ndarray,
        i_inds: jnp.ndarray,
        i_stim: jnp.ndarray,
        radius: float,
        length_single_compartment: float,
    ):
        """
        Return external input to each compartment in uA / cm^2.
        """
        zero_vec = jnp.zeros_like(voltages)
        # `radius`: um
        # `length_single_compartment`: um
        # `i_stim`: nA
        current = (
            i_stim / 2 / pi / radius[i_inds] / length_single_compartment[i_inds]
        )  # nA / um^2
        current *= 100_000  # Convert (nA / um^2) to (uA / cm^2)

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
    ) -> None:
        """Visualize the module.

        Args:
            ax: An axis into which to plot.
            col: The color for all branches.
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
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

    def _vis(self, ax, col, dims, view, type, morph_plot_kwargs):
        branches_inds = view["branch_index"].to_numpy()
        coords = []
        for branch_ind in branches_inds:
            assert not np.any(
                np.isnan(self.xyzr[branch_ind][:, dims])
            ), "No coordinates available. Use `vis(detail='point')` or run `.compute_xyz()` before running `.vis()`."
            coords.append(self.xyzr[branch_ind])

        ax = plot_morph(
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

        comp_fraction = loc_of_index(comp_ind, self.nseg)
        coords = self.xyzr[branch_ind]
        interpolated_xyz = interpolate_xyz(comp_fraction, coords)

        ax = plot_morph(
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

    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Move cells or networks in the (x, y, z) plane."""
        self._move(x, y, z, self.nodes)

    def _move(self, x: float, y: float, z: float, view):
        # Need to cast to set because this will return one columnn per compartment,
        # not one column per branch.
        indizes = set(view["branch_index"].to_numpy().tolist())
        for i in indizes:
            self.xyzr[i][:, 0] += x
            self.xyzr[i][:, 1] += y
            self.xyzr[i][:, 2] += z

    def rotate(self, degrees: float, rotation_axis: str = "xy"):
        """Rotate jaxley modules clockwise. Used only for visualization.

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
    def shape(self):
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

    def _childview(self, index: Union[int, str, list, range, slice]):
        """Return the child view of the current module.

        network.cell(index) at network level.
        cell.branch(index) at cell level.
        branch.comp(index) at branch level."""
        views = ["net", "cell", "branch", "comp", "/"]
        parent_name = self.__class__.__name__.lower()  # name of current view
        child_idx = (
            np.where([v in parent_name for v in views])[0][0] + 1
        )  # idx of child view
        child_view = views[child_idx]  # name of child view
        if child_view != "/":
            return self.__getattr__(child_view)(index)
        raise AttributeError("Compartment does not support indexing")

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._childview(index[0])[index[1:]]
        return self._childview(index)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def _local_inds_to_global(
        self, cell_inds: np.ndarray, branch_inds: np.ndarray, comp_inds: np.ndarray
    ):
        """Given local inds of cell, branch, and comp, return the global comp index."""
        global_ind = (
            self.cumsum_nbranches[cell_inds] + branch_inds
        ) * self.nseg + comp_inds
        return global_ind.astype(int)


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
    ):
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

    def set_global_index_and_index(self, nodes):
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

    def insert(self, channel):
        """Insert a channel."""
        assert not inspect.isclass(
            channel
        ), """
            Channel is a class, but it was not initialized. Use `.insert(Channel())`
            instead of `.insert(Channel)`.
            """
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._insert(channel, nodes)

    def record(self, state: str = "v", verbose: bool = True):
        """Record a state."""
        nodes = self.set_global_index_and_index(self.view)
        view = deepcopy(nodes)
        view["state"] = state
        recording_view = view[["comp_index", "state"]]
        recording_view = recording_view.rename(columns={"comp_index": "rec_index"})
        self.pointer._record(recording_view, verbose=verbose)

    def stimulate(self, current: Optional[jnp.ndarray] = None, verbose: bool = True):
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._stimulate(current, nodes, verbose=verbose)

    def data_stimulate(
        self,
        current,
        data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]],
        verbose: bool = False,
    ):
        """Insert a stimulus into the module within jit (or grad)."""
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._data_stimulate(
            current, data_stimuli, nodes, verbose=verbose
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
    ):
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._vis(
            ax=ax,
            col=col,
            dims=dims,
            view=nodes,
            type=type,
            morph_plot_kwargs=morph_plot_kwargs,
        )

    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._move(x, y, z, nodes)

    def adjust_view(self, key: str, index: Union[int, str, list, range, slice]):
        """Update view."""
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

    def _get_local_indices(self):
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
        index_A_by_B = (
            lambda x, A, B: x.groupby(B)[A].rank(method="dense").astype(int) - 1
        )  # reindexes A by B, i.e. A=[0,0,1,1], B=[0,1,2,4] -> B=[0,1,0,1]
        idcs = self.view[["cell_index", "branch_index", "comp_index"]]
        idcs.loc[:, "branch_index"] = index_A_by_B(idcs, "branch_index", "cell_index")
        idcs.loc[:, "comp_index"] = index_A_by_B(
            idcs, "comp_index", ["cell_index", "branch_index"]
        )
        return idcs

    def _childview(self, index: Union[int, str, list, range, slice]):
        """Return the child view of the current view.

        cell(0).branch(index) at cell level.
        branch(0).comp(index) at branch level."""
        views = np.array(["net", "cell", "branch", "comp", "/"])
        parent_name = self.__class__.__name__.lower()
        child_idx = np.roll([v in parent_name for v in views], 1)
        child_view = views[child_idx][0]
        if child_view != "/":
            return self.__getattr__(child_view)(index)
        raise AttributeError("Compartment does not support indexing")

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if len(index) > 1:
                return self._childview(index[0])[index[1:]]
            return self._childview(index[0])
        return self._childview(index)

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
    def shape(self):
        local_idcs = self._get_local_indices()
        return tuple(local_idcs.nunique())

    def _append_multiple_synapses(
        self, pre_rows: pd.DataFrame, post_rows: pd.DataFrame, synapse_type: Synapse
    ) -> None:
        """Append multiple rows to the `self.edges` table.

        This is used, e.g. by `fully_connect` and `connect`.
        """
        synapse_name = type(synapse_type).__name__
        type_ind, is_new_type = self._infer_synapse_type_ind(synapse_name)
        index = len(self.pointer.edges)

        post_loc = loc_of_index(post_rows["comp_index"].to_numpy(), self.pointer.nseg)
        pre_loc = loc_of_index(pre_rows["comp_index"].to_numpy(), self.pointer.nseg)

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
        self.pointer.edges = pd.concat(
            [self.pointer.edges, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

        indices = list(range(index, index + len(pre_loc)))
        self._add_params_to_edges(synapse_type, indices)

        if is_new_type:
            self._update_synapse_state_names(synapse_type)

    def _infer_synapse_type_ind(self, synapse_name: str) -> Tuple[int, bool]:
        """Return the unique identifier for every synapse type.

        Also returns a boolean indicating whether the synapse is already in the
        `module`.

        Used during `self._append_multiple_synapses`.
        """
        is_new_type = True if synapse_name not in self.pointer.synapse_names else False

        if is_new_type:
            # New type: index for the synapse type is one more than the currently
            # highest index.
            max_ind = self.pointer.edges["type_ind"].max() + 1
            type_ind = 0 if jnp.isnan(max_ind) else max_ind
        else:
            # Not a new type: search for the index that this type has previously had.
            type_ind = self.pointer.edges.query(f"type == '{synapse_name}'")[
                "type_ind"
            ].to_numpy()[0]
        return type_ind, is_new_type

    def _add_params_to_edges(self, synapse_type: Synapse, indices: list) -> None:
        """Fills parameter and state columns of new synapses in the `edges` table.

        This method does not create new rows in the `.edges` table. It only fills
        columns of already existing rows.

        Used during `self._append_multiple_synapses`.
        """
        # Add parameters and states to the `.edges` table.
        for key in synapse_type.synapse_params:
            param_val = synapse_type.synapse_params[key]
            self.pointer.edges.loc[indices, key] = param_val

        # Update synaptic state array.
        for key in synapse_type.synapse_states:
            state_val = synapse_type.synapse_states[key]
            self.pointer.edges.loc[indices, key] = state_val

    def _update_synapse_state_names(self, synapse_type: Synapse) -> None:
        """Update attributes with information about the synapses.

        Used during `self._append_multiple_synapses`.
        """
        # (Potentially) update variables that track meta information about synapses.
        self.pointer.synapse_names.append(type(synapse_type).__name__)
        self.pointer.synapse_param_names += list(synapse_type.synapse_params.keys())
        self.pointer.synapse_state_names += list(synapse_type.synapse_states.keys())
        self.pointer.synapses.append(synapse_type)


class GroupView(View):
    """GroupView (aka sectionlist).

    The only difference to a standard `View` is that it sets `controlled_by_param` to
    0 for all compartments. This means that a group will always be controlled by a
    single parameter.
    """

    def __init__(self, pointer, view):
        view["controlled_by_param"] = 0
        super().__init__(pointer, view)
