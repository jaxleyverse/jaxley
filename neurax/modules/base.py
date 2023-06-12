from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
import pandas as pd
import numpy as np
from copy import deepcopy
import inspect

from neurax.channels import Channel
from neurax.solver_voltage import step_voltage_explicit, step_voltage_implicit
from neurax.stimulus import get_external_input
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

        self.nodes: pd.DataFrame = None
        self.syn_edges: pd.DataFrame = None
        self.branch_edges: pd.DataFrame = None

        self.cumsum_nbranches: jnp.ndarray = None

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
            printable_nodes = deepcopy(self.nodes)

            if not indices:
                for key in printable_nodes:
                    printable_nodes = printable_nodes.drop(key, axis=1)

            if params:
                for key, val in self.params.items():
                    printable_nodes[key] = val

            if states:
                for key, val in self.states.items():
                    printable_nodes[key] = val

        else:
            printable_nodes = deepcopy(self.channel_nodes[channel_name])

            if not indices:
                for key in printable_nodes:
                    printable_nodes = printable_nodes.drop(key, axis=1)

            if params:
                for key, val in self.channel_params.items():
                    printable_nodes[key] = val

            if states:
                for key, val in self.channel_states.items():
                    printable_nodes[key] = val

        return printable_nodes

    def _init_params_and_state(
        self, own_params: Dict[str, List], own_states: Dict[str, List]
    ) -> None:
        """Sets parameters and state of the module at initialization.

        Args:
            own_params: _description_
            own_states: _description_
            constituent: _description_
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

    def _append_to_channel_params_and_state(self, constituents: List["Module"]):
        for comp in constituents:
            for key in comp.channel_params:
                if key in self.channel_params:
                    self.channel_params[key] = jnp.concatenate(
                        [self.channel_params[key], comp.channel_params[key]]
                    )
                else:
                    self.channel_params[key] = comp.channel_params[key]

        for comp in constituents:
            for key in comp.channel_states:
                if key in self.channel_states:
                    self.channel_states[key] = jnp.concatenate(
                        [self.channel_states[key], comp.channel_states[key]]
                    )
                else:
                    self.channel_states[key] = comp.channel_states[key]

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

    def set_params(self, key, val):
        """Set parameter for entire module."""
        if key in self.params.keys():
            self.params[key] = self.params[key].at[:].set(val)
            assert (
                key not in self.syn_params.keys()
            ), "Same key for synapse and node parameter."
            assert (
                key not in self.channel_params.keys()
            ), "Same key for channel and node parameter."
        elif key in self.channel_params.keys():
            self.channel_params[key] = self.channel_params[key].at[:].set(val)
        elif key in self.syn_params.keys():
            self.syn_params[key] = self.syn_params[key].at[:].set(val)
        else:
            raise KeyError(f"{key} not recognized.")

    def set_states(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.states[key] = self.states[key].at[:].set(val)

    def make_trainable(self, key: str, init_val: float):
        """Make a parameter trainable."""
        if key in self.params.keys():
            self.indices_set_by_trainables.append(self.nodes.index.to_numpy())
            self.trainable_params.append({key: jnp.asarray(init_val)})
            assert (
                key not in self.syn_params.keys()
            ), "Same key for synapse and node parameter."
        elif key in self.syn_params.keys():
            self.indices_set_by_trainables.append(self.syn_edges.index.to_numpy())
            self.trainable_params.append({key: jnp.asarray(init_val)})
        else:
            raise KeyError(f"Parameter {key} not recognized.")

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

    def insert(self, channel):
        """Insert a channel."""
        assert not inspect.isclass(
            channel
        ), """
            Channel is a class, but it was not initialized. Use `.insert(Channel())` 
            instead of `.insert(Channel)`.
            """
        new_nodes = self.nodes
        new_p = {}
        new_s = {}
        for key in channel.channel_params:
            new_p[key] = jnp.asarray(
                [channel.channel_params[key]] * self.total_nbranches * self.nseg
            )
        for key in channel.channel_states:
            new_s[key] = jnp.asarray(
                [channel.channel_states[key]] * self.total_nbranches * self.nseg
            )

        name = type(channel).__name__
        if name in self.channel_nodes:
            self.channel_nodes[name] = pd.concat(
                [self.channel_nodes[name], new_nodes]
            ).reset_index(drop=True)
            for key in channel.channel_params:
                self.channel_params[key] = jnp.concatenate(
                    [self.channel_params[key], new_p[key]]
                )
            for key in channel.channel_states:
                self.channel_states[key] = jnp.concatenate(
                    [self.channel_states[key], new_s[key]]
                )
        else:
            self.channel_nodes[name] = new_nodes
            for key in channel.channel_params:
                self.channel_params[key] = new_p[key]
            for key in channel.channel_states:
                self.channel_states[key] = new_s[key]
            self.channels.append(channel)
            self.params_per_channel.append(list(channel.channel_params.keys()))
            self.states_per_channel.append(list(channel.channel_states.keys()))

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
        i_ext = get_external_input(
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
        final_state = new_channel_states[0]
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
            states, membrane_current_terms = channel.step(
                states, delta_t, voltages[indices], params
            )
            voltage_terms = voltage_terms.at[indices].add(membrane_current_terms[0])
            constant_terms = constant_terms.at[indices].add(membrane_current_terms[1])
            new_channel_states.append(states)

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

    def show(self, indices: bool = True, params: bool = True, states: bool = True):
        inds = self.view.index.values
        printable_nodes = deepcopy(self.view)

        if not indices:
            for key in printable_nodes:
                printable_nodes = printable_nodes.drop(key, axis=1)

        if params:
            for key in self.pointer.params:
                printable_nodes[key] = self.pointer.params[key][inds]

        if states:
            for key in self.pointer.states:
                printable_nodes[key] = self.pointer.states[key][inds]

        return printable_nodes

    def set_params(self, key: str, val: float):
        """Set parameters of the pointer."""
        if key in self.pointer.params:
            self.pointer.params[key] = (
                self.pointer.params[key].at[self.view.index.values].set(val)
            )
        elif key in self.pointer.channel_params:
            channel_name = self.pointer.identify_channel_based_on_param_name(key)
            ind_of_params = self.channel_inds(self.view.index.values, channel_name)
            self.pointer.channel_params[key] = (
                self.pointer.channel_params[key].at[ind_of_params].set(val)
            )
        else:
            raise KeyError("Key not recognized.")

    def set_states(self, key: str, val: float):
        """Set parameters of the pointer."""
        if key in self.pointer.states:
            self.pointer.states[key] = (
                self.pointer.states[key].at[self.view.index.values].set(val)
            )
        elif key in self.pointer.channel_states:
            channel_name = self.pointer.identify_channel_based_on_state_name(key)
            ind_of_params = self.channel_inds(self.view.index.values, channel_name)
            self.pointer.channel_states[key] = (
                self.pointer.channel_states[key].at[ind_of_params].set(val)
            )
        else:
            raise KeyError("Key not recognized.")

    def get_params(self, key: str):
        """Return parameters."""
        if key in self.pointer.params:
            return self.pointer.params[key][self.view.index.values]
        elif key in self.pointer.channel_params:
            channel_name = self.pointer.identify_channel_based_on_param_name(key)
            ind_of_params = self.channel_inds(self.view.index.values, channel_name)
            return self.pointer.channel_params[key][ind_of_params]
        else:
            raise KeyError("Key not recognized.")

    def get_states(self, key: str):
        """Return states."""
        if key in self.pointer.states:
            return self.pointer.states[key][self.view.index.values]
        elif key in self.pointer.channel_states:
            channel_name = self.pointer.identify_channel_based_on_state_name(key)
            ind_of_states = self.channel_inds(self.view.index.values, channel_name)
            return self.pointer.channel_states[key][ind_of_states]
        else:
            raise KeyError("Key not recognized.")

    def make_trainable(self, key: str, init_val: float):
        """Make a parameter trainable."""
        assert (
            self.allow_make_trainable
        ), "network.cell('all') is not supported. Use a for-loop over cells."

        grouped_view = self.view.groupby("controlled_by_param")
        inds_of_comps = list(grouped_view.apply(lambda x: x.index.values))

        if key in self.pointer.params:
            indices_per_param = inds_of_comps
        elif key in self.pointer.channel_params:
            name = self.pointer.identify_channel_based_on_param_name(key)
            indices_per_param = [self.channel_inds(ind, name) for ind in inds_of_comps]
        else:
            raise KeyError(f"Parameter {key} not recognized.")

        self.pointer.indices_set_by_trainables.append(jnp.stack(indices_per_param))
        num_created_parameters = len(indices_per_param)
        self.pointer.trainable_params.append(
            {key: jnp.asarray([[init_val]] * num_created_parameters)}
        )

    def adjust_view(self, key: str, index: float):
        """Update view."""
        if index != "all":
            self.view = self.view[self.view[key] == index]
            self.view -= self.view.iloc[0]
        return self

    def channel_inds(self, ind_of_comps_to_be_set, channel_name: str):
        """Not all compartments might have all channels. Thus, we have to do some
        reindexing to find the associated index of a paramter of a channel given the
        index of a compartment.

        Args:
            channel_name: For example, `HHChannel`.
        """
        frame = self.pointer.channel_nodes[channel_name]
        channel_param_or_state_ind = frame.loc[
            frame["comp_index"].isin(ind_of_comps_to_be_set)
        ].index.values
        return channel_param_or_state_ind
