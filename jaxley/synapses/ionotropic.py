# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

from jax.nn import sigmoid

import jax.numpy as jnp

from jaxley.solver_gate import save_exp, exponential_euler
from jaxley.synapses.synapse import Synapse


class IonotropicSynapse(Synapse):
    r"""An ionotropic (state-based) synapse.

    This synapse is similar to the ``DynamicSynapse``, but its time constant is
    voltage dependent. In addition, this synapse only supports a sigmoidal activation
    function.

    This synapse implements the following equations:

    .. math::

        I = \overline{g}\, \cdot s\, \cdot (E - V_{\text{post}})

    .. math::

        \tau (V_{\text{pre}}) \, \cdot s = s_{\infty}(V_{\text{pre}}) - s

    .. math::

        s_{\infty}(V_{\text{pre}}) = \sigma\!\left(\frac{V_{\text{pre}} - V_{\text{thr}}}{\Delta}\right)

    .. math::

        \tau(V_{\text{pre}})\, = \frac{1 - s_{\infty}(V_{\text{pre}})}{k_{-}},

    The synapse state "s" is the probability that a postsynaptic receptor channel is
    open, and this depends on the amount of neurotransmitter released, which is in turn
    dependent on the presynaptic voltage. This synapse has a time constant which is
    voltage dependent.

    The synaptic parameters are:
        - ``gS``: the maximal conductance :math:`\overline{g}` (uS).
        - ``e_syn``: the reversal potential :math:`E` (mV).
        - ``k_minus``: the rate constant :math:`1/\tau` (:math:`s^{-1}`).
        - ``v_th``: the threshold at which the synapse becomes active
          :math:`V_{\text{thr}}` (mV).
        - ``delta``: The inverse of the slope of the activation :math:`\Delta` (mV).

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

        s_inf = sigmoid((v_th - pre_voltage) / delta)
        s_tau = (1.0 - s_inf) / params[f"{prefix}_k_minus"]

        new_s = exponential_euler(
            states[f"{prefix}_s"],
            delta_t,
            s_inf,
            s_tau
        )
        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        g_syn = params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn * (post_voltage - params[f"{prefix}_e_syn"])
