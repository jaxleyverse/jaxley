# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from jax import Array

from jaxley.synapses.synapse import Synapse


class ExpDecaySynapse(Synapse):
    """Synapse which decays exponentially after a pre-synaptic spike.

    This synapse requires that the pre-synaptic neuron has a binary voltage of either
    zero or one. After a pre-synaptic one (a spike), the state of the synapse gets
    increased immediately and then decays exponentially.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_decay_tau": 10.0,  # ms
        }
        self.synapse_states = {f"{prefix}_s": 0.0}

    def update_states(
        self,
        states: dict[str, Array],
        all_states: dict,
        pre_index: Array,
        post_index: Array,
        params: dict[str, Array],
        delta_t: float,
    ) -> dict:
        """Return updated synapse state and current."""
        prefix = self._name
        s = states[f"{prefix}_s"]
        tau = params[f"{prefix}_decay_tau"]

        spike = all_states["v"][pre_index]
        term1 = spike * 1.0
        term2 = (1 - spike) * (s - (s * delta_t / tau))
        new_s = term1 + term2

        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: dict, pre_voltage: float, post_voltage: float, params: dict
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        g_syn = -params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn
