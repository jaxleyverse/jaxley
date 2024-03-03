from typing import Dict, Tuple

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class IonotropicSynapse(Synapse):
    """
    Compute synaptic current and update synapse state for a generic ionotropic synapse.

    The synapse state "s" is the probability that a postsynaptic receptor channel is
    open, and this depends on the amount of neurotransmitter released, which is in turn
    dependent on the presynaptic voltage.

    The synaptic parameters are:
        - gS: the maximal conductance across the postsynaptic membrane
        - e_syn: the reversal potential across the postsynaptic membrane
        - k_minus: the rate constant of neurotransmitter unbinding from the postsynaptic
            receptor

    Details of this implementation can be found in the following book chapter:
        L. F. Abbott and E. Marder, "Modeling Small Networks," in Methods in Neuronal
        Modeling, C. Koch and I. Sergev, Eds. Cambridge: MIT Press, 1998.

    """

    synapse_params = {"gS": 0.5, "e_syn": 0.0, "k_minus": 0.025}
    synapse_states = {"s": 0.2}

    def update_states(self, u, delta_t, pre_voltage, post_voltage, params):
        """Return updated synapse state and current."""
        v_th = -35.0
        delta = 10.0

        s_inf = 1.0 / (1.0 + jnp.exp((v_th - pre_voltage) / delta))
        tau_s = (1.0 - s_inf) / params["k_minus"]

        slope = -1.0 / tau_s
        exp_term = jnp.exp(slope * delta_t)
        new_s = u["s"] * exp_term + s_inf * (1.0 - exp_term)
        return {"s": new_s}

    def compute_current(self, u, pre_voltage, post_voltage, params):
        g_syn = params["gS"] * u["s"]
        return g_syn * (post_voltage - params["e_syn"])
