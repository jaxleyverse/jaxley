from typing import Dict, Tuple, Optional

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class TanhRateSynapse(Synapse):
    """
    Compute synaptic current for tanh synapse (no state).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {f"{prefix}_gS": 0.5, f"{prefix}_x_offset": -70.0}
        self.synapse_states = {}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        return {}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        prefix = self._name
        current = -1 * params[f"{prefix}_gS"] * jnp.tanh(pre_voltage - params[f"{prefix}_x_offset"])
        return current
