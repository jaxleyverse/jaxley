from typing import Dict, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class TestSynapse(Synapse):
    """
    Compute syanptic current and update synapse state for a test synapse.
    """

    synapse_params = {"gC": 0.5}
    synapse_states = {"c": 0.2}

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
        new_s = u["c"] * exp_term + s_inf * (1.0 - exp_term)
        return {"c": new_s}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        e_syn = 0.0
        g_syn = params["gC"] * u["c"]
        return g_syn * (post_voltage - e_syn)
