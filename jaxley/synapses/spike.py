# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from jax import Array

from jaxley.synapses.synapse import Synapse


class SpikeSynapse(Synapse):
    """
    Compute synaptic current for a synapse within a network of LIF neurons.

    Note that the presynaptic neurons to this synapse must have Fire channels,
    as this is the mechanism which allows this synapse to detect presynaptic spikes.
    If the presynaptic neuron has no Fire channels, there will always be zero
    synaptic current.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_s_decay": 10.0,  # unitless
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(
        self,
        states: dict[str, Array],
        all_states: dict,
        pre_indicies: Array,
        post_indicies: Array,
        params: dict[str, Array],
        delta_t: float,
    ) -> dict:
        """Return updated synapse state and current."""
        prefix = self._name
        s = states[f"{prefix}_s"]
        s_decay = params[f"{prefix}_s_decay"]

        if "Fire_spikes" in all_states.keys():
            spike_states = all_states["Fire_spikes"][pre_indicies]
            spike = spike_states
        else:
            spike = 0.0

        new_s = (spike * 1.0) + ((1 - spike) * (s - (s_decay * s * delta_t)))

        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: dict, pre_voltage: float, post_voltage: float, params: dict
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        g_syn = params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn * (post_voltage - params[f"{prefix}_e_syn"])
