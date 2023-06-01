from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod

import jax.numpy as jnp
import pandas as pd

from neurax.channels import Channel
from neurax.synapses import Synapse
from neurax.solver_voltage import step_voltage_implicit, step_voltage_explicit
from neurax.stimulus import get_external_input


class Module(ABC):
    def __init__(self):
        self.nseg: int = None
        self.total_nbranches: int = 0
        self.nbranches_per_cell: List[int] = None

        self.channels: List[Channel] = None
        self.conns: List[Synapse] = None

        self.nodes: pd.DataFrame = None
        self.syn_edges: pd.DataFrame = None
        self.branch_edges: pd.DataFrame = None

        self.cumsum_nbranches: jnp.ndarray = None
        self.post_grouped_inds: jnp.ndarray = None
        self.post_grouped_syns: jnp.ndarray = None

        self.coupling_conds_bwd: jnp.ndarray = jnp.asarray([[]])
        self.coupling_conds_fwd: jnp.ndarray = jnp.asarray([[]])
        self.summed_coupling_conds: jnp.ndarray = jnp.asarray([[0.0]])
        self.branch_conds_fwd: jnp.ndarray = jnp.asarray([[]])
        self.branch_conds_bwd: jnp.ndarray = jnp.asarray([[]])
        self.comb_parents: jnp.ndarray = jnp.asarray([-1])
        self.comb_cum_child_inds_in_each_level: jnp.ndarray = jnp.asarray([0])
        self.max_num_children: int = 1
        self.comb_parents_in_each_level: List[jnp.ndarray] = [jnp.asarray([-1])]
        self.comb_branches_in_each_level: List[jnp.ndarray] = [jnp.asarray([0])]

        self.comb_cum_child_inds: jnp.ndarray = jnp.asarray([0])  # only for fwd-Euler.

        self.initialized_morph: bool = False
        self.initialized_conds: bool = False
        self.initialized_syns: bool = False

        self.params: Dict[str, jnp.ndarray] = {}
        self.states: Dict[str, jnp.ndarray] = {}

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

    def _append_to_params_and_state(self, constituents: List["Module"]):
        for key in constituents[0].params:
            param_vals = jnp.concatenate([b.params[key] for b in constituents])
            self.params[key] = param_vals

        for key in constituents[0].states:
            states_vals = jnp.concatenate([b.states[key] for b in constituents])
            self.states[key] = states_vals

    def set_params(self, key, val):
        """Set parameter for entire module."""
        self.params[key] = self.params[key].at[:].set(val)
        self.initialized_conds = False

    @property
    def initialized(self):
        """Whether the `Module` is ready to be solved or not."""
        return (
            self.initialized_morph and self.initialized_conds and self.initialized_syns
        )

    def initialize(self):
        """Initialize the module."""
        self.init_morph()
        self.init_syns()
        self.init_conds()
        return self

    def init_syns(self):
        self.initialized_syns = True

    def init_conds(self):
        self.initialized_conds = True

    def init_morph(self):
        self.initialized_morph = True

    def step(
        self,
        u,
        delta_t,
        i_inds,
        i_current,
        solver: str = "bwd_euler",
        tridiag_solver: str = "stone",
    ):
        """One step of integration."""
        voltages = u["voltages"]

        # Parameters have to go in here.
        new_channel_states, (v_terms, const_terms) = self._step_channels(
            u, delta_t, self.channels, self.params
        )

        # External input.
        i_ext = get_external_input(
            voltages, i_inds, i_current, self.params["radius"], self.params["length"]
        )

        # Step of the synapse.
        new_syn_states, syn_voltage_terms, syn_constant_terms = self._step_synapse(
            u,
            self.conns,
            self.params,
            delta_t,
            self.cumsum_nbranches,
            self.syn_edges["pre_comp_index"].to_numpy(),
            self.post_grouped_inds,
            self.post_grouped_syns,
            self.nseg,
        )

        # Voltage steps.
        if solver == "bwd_euler":
            new_voltages = step_voltage_implicit(
                voltages=voltages,
                voltage_terms=v_terms + syn_voltage_terms,
                constant_terms=const_terms + i_ext + syn_constant_terms,
                coupling_conds_bwd=self.coupling_conds_bwd,
                coupling_conds_fwd=self.coupling_conds_fwd,
                summed_coupling_conds=self.summed_coupling_conds,
                branch_cond_fwd=self.branch_conds_fwd,
                branch_cond_bwd=self.branch_conds_bwd,
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                child_inds_in_each_level=self.comb_cum_child_inds_in_each_level,
                max_num_children=self.max_num_children,
                parents_in_each_level=self.comb_parents_in_each_level,
                branches_in_each_level=self.comb_branches_in_each_level,
                tridiag_solver=tridiag_solver,
                delta_t=delta_t,
            )
        else:
            new_voltages = step_voltage_explicit(
                voltages,
                v_terms + syn_voltage_terms,
                const_terms + i_ext + syn_constant_terms,
                coupling_conds_bwd=self.coupling_conds_bwd,
                coupling_conds_fwd=self.coupling_conds_fwd,
                branch_cond_fwd=self.branch_conds_fwd,
                branch_cond_bwd=self.branch_conds_bwd,
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                child_inds=self.comb_cum_child_inds,
                max_num_children=self.max_num_children,
                delta_t=delta_t,
            )

        # Rebuild state.
        final_state = new_channel_states[0]
        for s in new_syn_states:
            for key in s:
                final_state[key] = s[key]
        final_state["voltages"] = new_voltages.flatten(order="C")

        return final_state

    @staticmethod
    def _step_channels(
        states, delta_t, channels: List[Channel], params: Dict[str, jnp.ndarray]
    ):
        """One step of integration of the channels."""
        voltages = states["voltages"]
        voltage_terms = jnp.zeros_like(voltages)  # mV
        constant_terms = jnp.zeros_like(voltages)
        new_channel_states = []
        for channel in channels:
            states, membrane_current_terms = channel.step(
                states, delta_t, voltages, params
            )
            voltage_terms += membrane_current_terms[0]
            constant_terms += membrane_current_terms[1]
            new_channel_states.append(states)

        return new_channel_states, (voltage_terms, constant_terms)

    @staticmethod
    def _step_synapse(
        u,
        syn_channels,
        params,
        delta_t,
        cumsum_num_branches,
        syn_inds,
        grouped_post_syn_inds,
        grouped_post_syns,
        nseg,
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

    def set_params(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.pointer.params[key] = (
            self.pointer.params[key].at[self.view.index.values].set(val)
        )
        self.pointer.initialized_conds = False

    def adjust_view(self, key: str, index: float):
        """Update view."""
        self.view = self.view[self.view[key] == index]
        self.view -= self.view.iloc[0]
        return self
