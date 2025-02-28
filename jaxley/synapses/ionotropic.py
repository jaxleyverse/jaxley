# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp

from jaxley.solver_gate import save_exp
from jaxley.synapses.synapse import Synapse


class IonotropicSynapse(Synapse):
    """
    Compute synaptic current and update synapse state for a generic ionotropic synapse.

    The synapse state "s" is the probability that a postsynaptic receptor channel is
    open, and this depends on the amount of neurotransmitter released, which is in turn
    dependent on the presynaptic voltage.

    The synaptic parameters are:
        - gS: the maximal conductance across the postsynaptic membrane (uS)
        - e_syn: the reversal potential across the postsynaptic membrane (mV)
        - k_minus: the rate constant of neurotransmitter unbinding from the postsynaptic
            receptor (s^-1)

    Details of this implementation can be found in the following book chapter:
        L. F. Abbott and E. Marder, "Modeling Small Networks," in Methods in Neuronal
        Modeling, C. Koch and I. Sergev, Eds. Cambridge: MIT Press, 1998.

    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_k_minus": 0.025,
            f"{prefix}_v_th": -35.0,  # mV
            f"{prefix}_delta": 10.0,  # mV
        }
        self.synapse_states = {f"{prefix}_s": 0.2}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        """Return updated synapse state and current."""
        prefix = self._name
        v_th = params[f"{prefix}_v_th"]
        delta = params[f"{prefix}_delta"]

        s_inf = 1.0 / (1.0 + save_exp((v_th - pre_voltage) / delta))
        tau_s = (1.0 - s_inf) / params[f"{prefix}_k_minus"]

        slope = -1.0 / tau_s
        exp_term = save_exp(slope * delta_t)
        new_s = states[f"{prefix}_s"] * exp_term + s_inf * (1.0 - exp_term)
        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        g_syn = params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn * (post_voltage - params[f"{prefix}_e_syn"])
