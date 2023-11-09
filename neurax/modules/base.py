import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from math import pi
from typing import Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.lax import ScatterDimensionNumbers, scatter_add

from neurax.channels import Channel
from neurax.solver_voltage import step_voltage_explicit, step_voltage_implicit
from neurax.synapses import Synapse


class Module(ABC):
    def __init__(self):
        self.nseg: int = None
        self.total_nbranches: int = 0
        self.nbranches_per_cell: List[int] = None

        self.channels: List[Channel] = []
        self.params_per_channel: List[List[str]] = []
        self.states_per_channel: List[List[str]] = []
        self.conns: List[Synapse] = None
        self.group_views = {}

        self.nodes: Optional[pd.DataFrame] = None
        self.syn_edges: Optional[pd.DataFrame] = None
        self.branch_edges: Optional[pd.DataFrame] = None

        self.cumsum_nbranches: Optional[jnp.ndarray] = None

        self.comb_parents: jnp.ndarray = jnp.asarray([-1])
        self.comb_branches_in_each_level: List[jnp.ndarray] = [jnp.asarray([0])]

        self.initialized_morph: bool = False
        self.initialized_syns: bool = False

        self.params: Dict[str, jnp.ndarray] = {}
        self.states: Dict[str, jnp.ndarray] = {}

        self.syn_params: Dict[str, jnp.ndarray] = {}
        self.syn_states: Dict[str, jnp.ndarray] = {}

        # Channel indices, parameters, and states.
        self.channel_nodes: Dict[str, pd.DataFrame] = {}
        self.channel_params: Dict[str, Dict[str, jnp.ndarray]] = {}
        self.channel_states: Dict[str, Dict[str, jnp.ndarray]] = {}

        # For trainable parameters.
        self.indices_set_by_trainables: List[jnp.ndarray] = []
        self.trainable_params: List[Dict[str, jnp.ndarray]] = []
        self.allow_make_trainable: bool = True

        # For recordings.
        self.recordings: pd.DataFrame = pd.DataFrame().from_dict({})

        # For stimuli.
        self.currents: Optional[jnp.ndarray] = None
        self.current_inds: pd.DataFrame = pd.DataFrame().from_dict({})

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channel_nodes)} different channels. Use `.show()` for details."

    def __str__(self):
        return f"nx.{type(self).__name__}"

    def show(
        self,
        channel_name: Optional[str] = None,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        """Print detailed information about the Module."""
        if channel_name is None:
            return self._show_base(self.nodes, indices, params, states)
        else:
            return self._show_channel(
                self.nodes,
                channel_name,
                indices,
                params,
                states,
            )

    def _show_base(
        self,
        view,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        inds = view.index.values
        printable_nodes = deepcopy(view)

        if not indices:
            for key in printable_nodes:
                printable_nodes = printable_nodes.drop(key, axis=1)

        if params:
            for key, val in self.params.items():
                printable_nodes[key] = val[inds]

        if states:
            for key, val in self.states.items():
                printable_nodes[key] = val[inds]

        return printable_nodes

    def _show_channel(self, view, channel_name, indices, params, states):
        ind_of_params = self.channel_inds(view.index.values, channel_name)
        nodes = deepcopy(self.channel_nodes[channel_name].loc[ind_of_params])
        nodes["comp_index"] -= nodes["comp_index"].iloc[0]
        nodes["branch_index"] -= nodes["branch_index"].iloc[0]
        nodes["cell_index"] -= nodes["cell_index"].iloc[0]

        if not indices:
            for key in nodes:
                nodes = nodes.drop(key, axis=1)

        if params:
            for key, val in self.channel_params.items():
                nodes[key] = val[ind_of_params]

        if states:
            for key, val in self.channel_states.items():
                nodes[key] = val[ind_of_params]

        return nodes

    def _init_params_and_state(
        self, own_params: Dict[str, List], own_states: Dict[str, List]
    ) -> None:
        """Sets parameters and state of the module at initialization.

        Args:
            own_params: Parameters of the current module, excluding parameters of its
                constituents.
            own_states: States of the current module, excluding states of its
                constituents.
        """
        self.params = {}
        for key in own_params:
            self.params[key] = jnp.asarray([own_params[key]])  # should be atleast1d

        self.states = {}
        for key in own_states:
            self.states[key] = jnp.asarray([own_states[key]])  # should be atleast1d

    @abstractmethod
    def init_conds(self, params):
        """Initialize coupling conductances.

        Args:
            params: Conductances and morphology parameters, not yet including
                coupling conductances.
        """
        raise NotImplementedError

    def _append_to_params_and_state(self, constituents: List["Module"]):
        for key in constituents[0].params:
            param_vals = jnp.concatenate([b.params[key] for b in constituents])
            self.params[key] = param_vals

        for key in constituents[0].states:
            states_vals = jnp.concatenate([b.states[key] for b in constituents])
            self.states[key] = states_vals

    def _append_to_channel_params_and_state(
        self, channel: Union[Channel, "Module"], repeats: int = 1
    ):
        for key in channel.channel_params:
            new_params = jnp.tile(jnp.atleast_1d(channel.channel_params[key]), repeats)
            if key in self.channel_params:
                self.channel_params[key] = jnp.concatenate(
                    [self.channel_params[key], new_params]
                )
            else:
                self.channel_params[key] = new_params

        for key in channel.channel_states:
            new_states = jnp.tile(jnp.atleast_1d(channel.channel_states[key]), repeats)
            if key in self.channel_states:
                self.channel_states[key] = jnp.concatenate(
                    [self.channel_states[key], new_states]
                )
            else:
                self.channel_states[key] = new_states

    def _append_to_channel_nodes(self, index, channel):
        """Adds channel nodes from constituents to `self.channel_nodes`."""
        name = type(channel).__name__
        if name in self.channel_nodes:
            self.channel_nodes[name] = pd.concat(
                [self.channel_nodes[name], index]
            ).reset_index(drop=True)
        else:
            self.channel_nodes[name] = index
            self.channels.append(channel)
            self.params_per_channel.append(list(channel.channel_params.keys()))
            self.states_per_channel.append(list(channel.channel_states.keys()))

    def identify_channel_based_on_param_name(self, name):
        for i, param_names in enumerate(self.params_per_channel):
            if name in param_names:
                return type(self.channels[i]).__name__
        raise KeyError("parameter name was not found in any channel")

    def identify_channel_based_on_state_name(self, name):
        for i, state_names in enumerate(self.states_per_channel):
            if name in state_names:
                return type(self.channels[i]).__name__
        raise KeyError("state name was not found in any channel")

    def channel_inds(self, ind_of_comps_to_be_set, channel_name: str):
        """Not all compartments might have all channels. Thus, we have to do some
        reindexing to find the associated index of a paramter of a channel given the
        index of a compartment.

        Args:
            channel_name: For example, `HHChannel`.
        """
        frame = self.channel_nodes[channel_name]
        channel_param_or_state_ind = frame.loc[
            frame["comp_index"].isin(ind_of_comps_to_be_set)
        ].index.values
        return channel_param_or_state_ind

    def set_params(self, key, val):
        """Set parameter."""
        self._set_params(key, val, self.nodes)

    def _set_params(self, key, val, view):
        if key in self.params:
            self.params[key] = self.params[key].at[view.index.values].set(val)
        elif key in self.channel_params:
            channel_name = self.identify_channel_based_on_param_name(key)
            ind_of_params = self.channel_inds(view.index.values, channel_name)
            self.channel_params[key] = (
                self.channel_params[key].at[ind_of_params].set(val)
            )
        elif key in self.syn_params:
            self.syn_params[key] = self.syn_params[key].at[view.index.values].set(val)
        else:
            raise KeyError("Key not recognized.")

    def set_states(self, key, val):
        """Set parameters."""
        self._set_states(key, val, self.nodes)

    def _set_states(self, key: str, val: float, view):
        if key in self.states:
            self.states[key] = self.states[key].at[view.index.values].set(val)
        elif key in self.channel_states:
            channel_name = self.identify_channel_based_on_state_name(key)
            ind_of_params = self.channel_inds(view.index.values, channel_name)
            self.channel_states[key] = (
                self.channel_states[key].at[ind_of_params].set(val)
            )
        elif key in self.syn_states:
            self.syn_states[key] = self.syn_states[key].at[view.index.values].set(val)
        else:
            raise KeyError("Key not recognized.")

    def get_params(self, key: str):
        """Return parameters."""
        return self._get_params(key, self.nodes)

    def _get_params(self, key: str, view):
        if key in self.params:
            return self.params[key][view.index.values]
        elif key in self.channel_params:
            channel_name = self.identify_channel_based_on_param_name(key)
            ind_of_params = self.channel_inds(view.index.values, channel_name)
            return self.channel_params[key][ind_of_params]
        elif key in self.syn_params:
            return self.syn_params[key][view.index.values]
        else:
            raise KeyError("Key not recognized.")

    def get_states(self, key: str):
        """Return states."""
        return self._get_states(key, self.nodes)

    def _get_states(self, key: str, view):
        if key in self.states:
            return self.states[key][view.index.values]
        elif key in self.channel_states:
            channel_name = self.identify_channel_based_on_state_name(key)
            ind_of_states = self.channel_inds(view.index.values, channel_name)
            return self.channel_states[key][ind_of_states]
        elif key in self.syn_states:
            return self.syn_states[key][view.index.values]
        else:
            raise KeyError("Key not recognized.")

    def make_trainable(self, key: str, init_val: Optional[Union[float, list]] = None):
        """Make a parameter trainable.

        Args:
            key: Name of the parameter to make trainable.
            init_val: Initial value of the parameter. If `float`, the same value is
                used for every created parameter. If `list`, the length of the list has
                to match the number of created parameters. If `None`, the current
                parameter value is used and if parameter sharing is performed that the
                current parameter value is averaged over all shared parameters.
        """
        view = deepcopy(self.nodes.assign(controlled_by_param=0))
        self._make_trainable(view, key, init_val)

    def _make_trainable(
        self, view, key: str, init_val: Optional[Union[float, list]] = None
    ):
        assert (
            self.allow_make_trainable
        ), "network.cell('all') is not supported. Use a for-loop over cells."

        grouped_view = view.groupby("controlled_by_param")
        inds_of_comps = list(grouped_view.apply(lambda x: x.index.values))

        if key in self.params:
            indices_per_param = jnp.stack(inds_of_comps)
            param_vals = self.params[key][indices_per_param]
        elif key in self.channel_params:
            name = self.identify_channel_based_on_param_name(key)
            indices_per_param = jnp.stack(
                [self.channel_inds(ind, name) for ind in inds_of_comps]
            )
            param_vals = self.channel_params[key][indices_per_param]
        elif key in self.syn_params:
            indices_per_param = jnp.stack(inds_of_comps)
            param_vals = self.syn_params[key][indices_per_param]
        else:
            raise KeyError(f"Parameter {key} not recognized.")

        self.indices_set_by_trainables.append(indices_per_param)

        num_created_parameters = len(indices_per_param)
        if init_val is not None:
            if isinstance(init_val, float):
                new_params = jnp.asarray([[init_val]] * num_created_parameters)
            elif isinstance(init_val, list):
                assert (
                    len(init_val) == num_created_parameters
                ), f"len(init_val)={len(init_val)}, but trying to create {num_created_parameters} parameters."
                new_params = jnp.asarray(init_val)[:, None]
            else:
                raise ValueError(
                    f"init_val must a float, list, or None, but it is a {type(init_val).__name__}."
                )
        else:
            new_params = jnp.mean(param_vals, axis=1, keepdims=True)

        self.trainable_params.append({key: new_params})

    def add_to_group(self, group_name):
        raise ValueError("`add_to_group()` makes no sense for an entire module.")

    def _add_to_group(self, group_name, view):
        if group_name in self.group_views:
            view = pd.concat([self.group_views[group_name].view, view])
        self.group_views[group_name] = GroupView(self, view)

    def get_parameters(self):
        """Get all trainable parameters."""
        return self.trainable_params

    def get_all_parameters(self, trainable_params):
        """Return all parameters (and coupling conductances) needed to simulate."""
        params = {}
        for key, val in self.params.items():
            params[key] = val

        for key, val in self.syn_params.items():
            params[key] = val

        for key, val in self.channel_params.items():
            params[key] = val

        for inds, set_param in zip(self.indices_set_by_trainables, trainable_params):
            for key in set_param.keys():
                params[key] = params[key].at[inds].set(set_param[key])

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
        self.init_syns()
        return self

    def record(self):
        """Insert a recording into the compartment."""
        self._record(self.nodes)

    def _record(self, view):
        assert (
            len(view) == 1
        ), "Can only record from compartments, not branches, cells, or networks."
        self.recordings = pd.concat([self.recordings, view])

    def stimulate(self, current):
        """Insert a stimulus into the compartment."""
        self._stimulate(current, self.nodes)

    def _stimulate(self, current, view):
        assert (
            len(view) == 1
        ), "Can only stimulate compartments, not branches, cells, or networks."
        if self.currents is not None:
            self.currents = jnp.concatenate(
                [self.currents, jnp.expand_dims(current, axis=0)]
            )
        else:
            self.currents = jnp.expand_dims(current, axis=0)
        self.current_inds = pd.concat([self.current_inds, view])

    def insert(self, channel):
        """Insert a channel."""
        self._insert(channel, self.nodes)

    def _insert(self, channel, view):
        self._append_to_channel_nodes(view, channel)
        self._append_to_channel_params_and_state(channel, repeats=len(view))

    def init_syns(self):
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
        """One step of integration."""
        voltages = u["voltages"]

        # Parameters have to go in here.
        new_channel_states, (v_terms, const_terms) = self._step_channels(
            u, delta_t, self.channels, self.channel_nodes, params
        )

        # External input.
        i_ext = self.get_external_input(
            voltages, i_inds, i_current, params["radius"], params["length"]
        )

        # Step of the synapse.
        new_syn_states, syn_voltage_terms, syn_constant_terms = self._step_synapse(
            u,
            self.conns,
            params,
            delta_t,
            self.syn_edges,
        )

        # Voltage steps.
        if solver == "bwd_euler":
            new_voltages = step_voltage_implicit(
                voltages=voltages,
                voltage_terms=v_terms + syn_voltage_terms,
                constant_terms=const_terms + i_ext + syn_constant_terms,
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
                v_terms + syn_voltage_terms,
                const_terms + i_ext + syn_constant_terms,
                coupling_conds_bwd=params["coupling_conds_bwd"],
                coupling_conds_fwd=params["coupling_conds_fwd"],
                branch_cond_fwd=params["branch_conds_fwd"],
                branch_cond_bwd=params["branch_conds_bwd"],
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                delta_t=delta_t,
            )

        # Rebuild state.
        final_state = {}
        for channel in new_channel_states:
            for key, val in channel.items():
                final_state[key] = val

        for s in new_syn_states:
            for key, val in s.items():
                final_state[key] = val

        final_state["voltages"] = new_voltages.flatten(order="C")

        return final_state

    @staticmethod
    def _step_channels(
        states,
        delta_t,
        channels: List[Channel],
        channel_nodes: List[pd.DataFrame],
        params: Dict[str, jnp.ndarray],
    ):
        """One step of integration of the channels."""
        voltages = states["voltages"]
        voltage_terms = jnp.zeros_like(voltages)  # mV
        constant_terms = jnp.zeros_like(voltages)
        new_channel_states = []
        for channel in channels:
            name = type(channel).__name__
            indices = channel_nodes[name]["comp_index"].to_numpy()
            states_updated, membrane_current_terms = channel.step(
                states, delta_t, voltages[indices], params
            )
            voltage_terms = voltage_terms.at[indices].add(membrane_current_terms[0])
            constant_terms = constant_terms.at[indices].add(membrane_current_terms[1])
            new_channel_states.append(states_updated)

        return new_channel_states, (voltage_terms, constant_terms)

    @staticmethod
    def _step_synapse(
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
        voltages = u["voltages"]
        return [{}], jnp.zeros_like(voltages), jnp.zeros_like(voltages)

    @staticmethod
    def get_external_input(
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
        channel_name: Optional[str] = None,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        if channel_name is None:
            myview = self.view.drop("global_comp_index", axis=1)
            myview = myview.drop("global_branch_index", axis=1)
            myview = myview.drop("global_cell_index", axis=1)
            return self.pointer._show_base(myview, indices, params, states)
        else:
            return self.pointer._show_channel(
                self.view, channel_name, indices, params, states
            )

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

    def record(self):
        """Insert a channel."""
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._record(nodes)

    def stimulate(self, current):
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._stimulate(current, nodes)

    def set_params(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.pointer._set_params(key, val, self.view)

    def set_states(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.pointer._set_states(key, val, self.view)

    def get_params(self, key: str):
        """Return parameters."""
        return self.pointer._get_params(key, self.view)

    def get_states(self, key: str):
        """Return states."""
        return self.pointer._get_states(key, self.view)

    def make_trainable(self, key: str, init_val: Optional[Union[float, list]] = None):
        """Make a parameter trainable."""
        self.pointer._make_trainable(self.view, key, init_val)

    def add_to_group(self, group_name: str):
        self.pointer._add_to_group(group_name, self.view)

    def adjust_view(self, key: str, index: float):
        """Update view."""
        if isinstance(index, int) or isinstance(index, np.int64):
            self.view = self.view[self.view[key] == index]
        elif isinstance(index, list):
            self.view = self.view[self.view[key].isin(index)]
        else:
            assert index == "all"
        self.view["comp_index"] -= self.view["comp_index"].iloc[0]
        self.view["branch_index"] -= self.view["branch_index"].iloc[0]
        self.view["cell_index"] -= self.view["cell_index"].iloc[0]
        self.view["controlled_by_param"] -= self.view["controlled_by_param"].iloc[0]
        return self


class GroupView(View):
    """GroupView (aka sectionlist).

    The only difference to a standard `View` is that it sets `controlled_by_param` to
    0 for all compartments. This means that a group will always be controlled by a
    single parameter.
    """

    def __init__(self, pointer, view):
        view["controlled_by_param"] = 0
        super().__init__(pointer, view)
