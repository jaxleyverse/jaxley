from typing import Dict, List, Optional, Callable
import jax.numpy as jnp

from neurax.modules.base import Module, View
from neurax.modules.channel import Channel, ChannelView


class Compartment(Module):
    compartment_params: Dict = {}
    compartment_states: Dict = {"voltages": 70.0}

    def __init__(self, channels: List[Channel] = None):
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

        new_channel_states, (v_terms, const_terms) = Compartment.step_channels(
            voltages, channel_states, dt, self.channels
        )
        new_voltages = Compartment.step_voltages(
            voltages, v_terms, const_terms, dt, ext_input=ext_input
        )
        return {"voltages": new_voltages, "channels": new_channel_states}

    @staticmethod
    def step_channels(voltages, channel_states, dt, channels: List[Channel]):
        voltage_terms = jnp.zeros_like(voltages)  # mV
        constant_terms = jnp.zeros_like(voltages)
        new_channel_states = []
        for i, channel in enumerate(channels):
            # TODO need to pass params.
            states, membrane_current_terms = channel.step(
                channel_states[i], dt, voltages
            )
            voltage_terms += membrane_current_terms[0]
            constant_terms += membrane_current_terms[1]
            new_channel_states.append(states)

        return new_channel_states, (voltage_terms, constant_terms)

    @staticmethod
    def step_voltages(voltages, voltage_terms, constant_terms, dt, ext_input=0):
        """Perform a voltage update with forward Euler."""
        voltage_vectorfield = -voltage_terms * voltages + constant_terms + ext_input
        new_voltages = voltages + dt * voltage_vectorfield
        return new_voltages


class CompartmentView(View):
    def __init__(self, pointer, view):
        super().__init__(pointer, view)

    def __call__(self, index):
        return super().adjust_view("comp_index", index)

    def __getattr__(self, key):
        assert key == "channel"
        return ChannelView(self.pointer, self.view)
