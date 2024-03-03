from typing import Dict, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class GapJunction(Synapse):
    """
    Compute synaptic current for a gap junction.

    The synaptic parameters are:
        - gE: the conductance across the gap junction
    """

    synapse_params = {"gE": 0.5}
    synapse_states = {}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state."""
        return {}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        """Return updated current."""
        return params["gE"] * (pre_voltage - post_voltage)
