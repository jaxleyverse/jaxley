from typing import Dict, List, Optional
import jax.numpy as jnp

from neurax.modules.base import Module, View
from neurax.modules.channel import Channel, ChannelView


class Compartment(Module):
    def __init__(
        self, channels: List[Channel], params: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        self.channels = channels
        self.params = params

    def set_params(self, key, val):
        self.params[key][:] = val

    def __getattr__(self, key):
        assert key == "cell"
        return ChannelView(self, self.nodes)

    def step(self, u):
        voltage_terms = jnp.zeros_like(u)  # mV
        constant_terms = jnp.zeros_like(u)
        new_mem_states = []
        for i, update_fn in enumerate(mem_channels):
            membrane_current_terms, states = update_fn(
                u, mem_states[i], mem_params[i], delta_t
            )
            voltage_terms += membrane_current_terms[0]
            constant_terms += membrane_current_terms[1]
            new_mem_states.append(states)
        return new_mem_states, voltage_terms, constant_terms


class CompartmentView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("comp_index", index)

    def __getattr__(self, key):
        assert key == "channel"
        return ChannelView(self.pointer, self.view)
