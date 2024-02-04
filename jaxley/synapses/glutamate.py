from typing import Dict, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class GlutamateSynapse(Synapse):
    """
    Compute syanptic current and update synapse state for Glutamate receptor.
    """

    synapse_params = {"gS": 0.5, "e_syn": 0.0}
    synapse_states = {"s": 0.2}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        v_th = -35.0
        delta = 10.0
        k_minus = 1.0 / 40.0

        s_bar = 1.0 / (1.0 + jnp.exp((v_th - pre_voltage) / delta))
        tau_s = (1.0 - s_bar) / k_minus

        s_inf = s_bar
        slope = -1.0 / tau_s
        exp_term = jnp.exp(slope * delta_t)
        new_s = u["s"] * exp_term + s_inf * (1.0 - exp_term)
        return {"s": new_s}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        g_syn = params["gS"] * u["s"]
        return g_syn * (post_voltage - params["e_syn"])
