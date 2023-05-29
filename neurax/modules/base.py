from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod

import jax.numpy as jnp
from neurax.channels import Channel
from neurax.solver_voltage import step_voltage_implicit


class Module(ABC):
    def __init__(self):
        pass

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
            self.params[key] = jnp.asarray(own_params[key])

        self.states = {}
        for key in own_states:
            self.states[key] = jnp.asarray(own_states[key])

    def _append_to_params_and_state(self, constituents: List["Module"]):
        for key in constituents[0].params:
            param_vals = jnp.asarray([b.params[key] for b in constituents])
            self.params[key] = param_vals

        for key in constituents[0].states:
            states_vals = jnp.asarray([b.states[key] for b in constituents])
            self.states[key] = states_vals

    @abstractmethod
    def step(self, u, dt, *args):
        raise NotImplementedError

    @staticmethod
    def step_channels(
        states, dt, channels: List[Channel], params: Dict[str, jnp.ndarray]
    ):
        voltages = states["voltages"]
        voltage_terms = jnp.zeros_like(voltages)  # mV
        constant_terms = jnp.zeros_like(voltages)
        new_channel_states = []
        for channel in channels:
            # TODO need to pass params.
            states, membrane_current_terms = channel.step(states, dt, voltages, params)
            voltage_terms += membrane_current_terms[0]
            constant_terms += membrane_current_terms[1]
            new_channel_states.append(states)

        return new_channel_states, (voltage_terms, constant_terms)

    @staticmethod
    def step_voltages(
        voltages,
        voltage_terms,
        constant_terms,
        coupling_conds_bwd,
        coupling_conds_fwd,
        summed_coupling_conds,
        branch_cond_fwd,
        branch_cond_bwd,
        num_branches,
        parents,
        kid_inds_in_each_level,
        max_num_kids,
        parents_in_each_level,
        branches_in_each_level,
        tridiag_solver,
        delta_t,
    ):
        """Perform a voltage update with forward Euler."""
        return step_voltage_implicit(
            voltages,
            voltage_terms,
            constant_terms,
            coupling_conds_bwd,
            coupling_conds_fwd,
            summed_coupling_conds,
            branch_cond_fwd,
            branch_cond_bwd,
            num_branches,
            parents,
            kid_inds_in_each_level,
            max_num_kids,
            parents_in_each_level,
            branches_in_each_level,
            tridiag_solver,
            delta_t,
        )


class View:
    def __init__(self, pointer, view):
        self.pointer = pointer
        self.view = view

    def set_params(self, key, val):
        self.pointer.params[key][self.view.index.values] = val

    def adjust_view(self, key, index):
        self.view = self.view[self.view[key] == index]
        self.view -= self.view.iloc[0]
        return self
