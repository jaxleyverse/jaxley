from typing import Dict, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class TestSynapse(Synapse):
    """
    Compute syanptic current and update syanpse state for a test synapse.
    """

    synapse_params = {"gC": 0.5}
    synapse_states = {"c": 0.2}

    @staticmethod
    def step(
        u: Dict[str, jnp.ndarray],
        delta_t: float,
        voltages: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        pre_inds: jnp.ndarray,
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return updated synapse state and current."""
        e_syn = 0.0
        v_th = -35.0
        delta = 10.0
        k_minus = 1.0 / 40.0

        s_bar = 1.0 / (1.0 + jnp.exp((v_th - voltages[pre_inds]) / delta))
        tau_s = (1.0 - s_bar) / k_minus

        s_inf = s_bar
        slope = -1.0 / tau_s
        exp_term = jnp.exp(slope * delta_t)
        new_s = u["s"] * exp_term + s_inf * (1.0 - exp_term)

        non_zero_voltage_term = params["gC"] * u["c"]
        non_zero_constant_term = params["gC"] * u["c"] * e_syn

        return {"s": new_s}, (non_zero_voltage_term, non_zero_constant_term)
