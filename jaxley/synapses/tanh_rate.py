from typing import Dict, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class TanhRateSynapse(Synapse):
    """
    Compute synaptic current for tanh synapse (no state).
    """

    synapse_params = {"gS": 0.5, "x_offset": -70.0}
    synapse_states = {}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        return {}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        current = params["gS"] * jnp.tanh(pre_voltage - params["x_offset"])
        return current
