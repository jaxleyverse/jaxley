from typing import Dict

import jax.numpy as jnp

from neurax.synapses.synapse import Synapse


class GlutamateSynapse(Synapse):
    """
    Compute syanptic current and update syanpse state for Glutamate receptor.
    """

    synapse_params = {"gS": 0.5}
    synapse_states = {"s": 0.2}

    @staticmethod
    def step(
        u: Dict[str, jnp.ndarray],
        dt,
        voltages,
        params: Dict[str, jnp.ndarray],
        pre_inds: jnp.ndarray,
    ):
        """Return updated synapse state and current."""
        e_syn = 0.0
        v_th = -35.0
        delta = 10.0
        k_minus = 1.0 / 40.0

        s_bar = 1.0 / (1.0 + jnp.exp((v_th - voltages[pre_inds]) / delta))
        tau_s = (1.0 - s_bar) / k_minus

        s_inf = s_bar
        slope = -1.0 / tau_s
        exp_term = jnp.exp(slope * dt)
        new_s = u["s"] * exp_term + s_inf * (1.0 - exp_term)

        non_zero_voltage_term = params["gS"] * u["s"]
        non_zero_constant_term = params["gS"] * u["s"] * e_syn

        return {"s": new_s}, (non_zero_voltage_term, non_zero_constant_term)
