from typing import Dict, List, Optional, Callable
import jax.numpy as jnp

from neurax.modules.base import Module, View
from neurax.channels import Channel  # , ChannelView


class Compartment(Module):
    compartment_params: Dict = {
        "length": 10.0,
        "radius": 1.0,
        "axial_resistivity": 1_000.0,
    }
    compartment_states: Dict = {"voltages": 70.0}

    def __init__(self, channels: List[Channel]):
        self.channels = channels

        # This is where I will want to build the graph.
        # I should append to params here (the ones of the compartment).
        self.params = [channel.params for channel in channels]

        # Gather initial states from gates. This is not used in `step`, but will be
        # used in `solve`.
        mem_states = [channel.states for channel in self.channels]
        self.states = {
            "voltages": self.compartment_states["voltages"],
            "channels": mem_states,
        }

    def set_params(self, key, val):
        self.params[key][:] = val

    def __getattr__(self, key):
        assert key == "cell"
        return ChannelView(self, self.nodes)

    def step(self, u: Dict[str, jnp.ndarray], dt: float, ext_input=0):
        """Step for a single compartment.

        Args:
            u: The full state vector, including states of all channels and the voltage.
            dt: Time step.

        Returns:
            Next state. Same shape as `u`.
        """
        voltages = u["voltages"]
        channel_states = u["channels"]

        new_channel_states, (v_terms, const_terms) = self.step_channels(
            voltages, channel_states, dt, self.channels
        )
        new_voltages = self.step_voltages(
            jnp.asarray([[voltages]]),
            jnp.asarray([[v_terms]]),
            jnp.asarray([[const_terms]]),
            coupling_conds_bwd=jnp.asarray([[]]),
            coupling_conds_fwd=jnp.asarray([[]]),
            summed_coupling_conds=jnp.asarray([[0.0]]),
            branch_cond_fwd=jnp.asarray([[]]),
            branch_cond_bwd=jnp.asarray([[]]),
            num_branches=1,
            parents=jnp.asarray([-1]),
            kid_inds_in_each_level=jnp.asarray([0]),
            max_num_kids=1,
            parents_in_each_level=[jnp.asarray([-1,])],
            branches_in_each_level=[jnp.asarray([0,])],
            tridiag_solver="thomas",
            delta_t=dt,
        )
        return {"voltages": new_voltages, "channels": new_channel_states}


class CompartmentView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("comp_index", index)

    def __getattr__(self, key):
        assert key == "channel"
        return ChannelView(self.pointer, self.view)
