from typing import Dict, List, Optional, Callable

import jax.numpy as jnp

from neurax.modules.base import Module, View
from neurax.modules.compartment import Compartment, CompartmentView


class Branch(Module):
    branch_params: Dict = {}
    branch_states: Dict = {}

    def __init__(self, compartments: List[Compartment]):
        self.compartments = compartments
        self.params = [comp.params for comp in compartments]

        voltage_states = [comp.states["voltages"] for comp in compartments]
        mem_states = [comp.states["mem_states"] for comp in compartments]
        self.states = {
            "voltages": voltage_states,
            "channels": mem_states,
        }
        self.branch_conds = None

    def set_params(self, key, val):
        self.params[key][:] = val

    def __getattr__(self, key):
        assert key == "cell"
        return CompartmentView(self, self.nodes)

    def init_branch_conds(self, compartments):
        pass

    def step(self, u: Dict[str, jnp.ndarray], dt: float, ext_input=0):
        """Step for a single compartment.

        Args:
            u: The full state vector, including states of all channels and the voltage.
            dt: Time step.

        Returns:
            Next state. Same shape as `u`.
        """
        if self.branch_conds is None:
            self.init_branch_conds()

        voltages = u["voltages"]
        channel_states = u["channels"]

        # This will be a vmap.
        new_channel_states, (v_terms, const_terms) = Compartment.step_channels(
            voltages, channel_states, dt, self.compartments[0].channels
        )
        new_voltages = Branch.step_voltages(
            voltages, v_terms, const_terms, dt, ext_input=ext_input
        )
        return {"voltages": new_voltages, "channels": new_channel_states}

    @staticmethod
    def step_voltages(voltages, branch_conds, voltage_terms, const_terms, dt):
        solved_voltage = solve_thomas()
        return solved_voltage


class BranchView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("branch_index", index)

    def __getattr__(self, key):
        assert key == "comp"
        return CompartmentView(self.pointer, self.view)
