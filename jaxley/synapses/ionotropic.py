# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid

from jaxley.solver_gate import exponential_euler
from jaxley.synapses.synapse import Synapse


class IonotropicSynapse(Synapse):
    r"""A state-based synapse with voltage dependent time constant.

    This synapse is similar to the ``DynamicSynapse``, but its time constant is
    voltage dependent. In addition, this synapse only supports a sigmoidal activation
    function.

    This synapse implements the following equations:

    .. math::

        I = \overline{g}\, \cdot s\, \cdot (E - V_{\text{post}})

    .. math::

        \tau (V_{\text{pre}}) \frac{\text{d}s}{\text{d}t} = s_{\infty}(V_{\text{pre}}) - s

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
        - ``k_minus``: the rate constant :math:`1/\tau` (:math:`ms^{-1}`).
        - ``v_th``: the threshold at which the synapse becomes active
          :math:`V_{\text{thr}}` (mV).
        - ``delta``: The inverse of the slope of the activation :math:`\Delta` (mV).

    The synaptic state is:
        - ``s``: the activity level of the synapse :math:`\in [0, 1]`.

    Details of this implementation can be found in the following book chapter:
        L. F. Abbott and E. Marder, "Modeling Small Networks," in Methods in Neuronal
        Modeling, C. Koch and I. Sergev, Eds. Cambridge: MIT Press, 1998.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_k_minus": 0.025,  # 1/ms
            f"{prefix}_v_th": -35.0,  # mV
            f"{prefix}_delta": 10.0,  # mV
        }
        self.synapse_states = {f"{prefix}_s": 0.2}
        self.node_params = {f"{prefix}_e_syn": 0.0}
        self.node_states = {}

    def update_states(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> Dict:
        """Return updated synapse state and current."""
        prefix = self._name
        v_th = synapse_params[f"{prefix}_v_th"]
        delta = synapse_params[f"{prefix}_delta"]

        s_inf = sigmoid((pre_voltage - v_th) / delta)
        s_tau = (1.0 - s_inf) / synapse_params[f"{prefix}_k_minus"]

        new_s = exponential_euler(
            synapse_states[f"{prefix}_s"],
            delta_t,
            s_inf,
            s_tau,
        )
        return {f"{prefix}_s": new_s}

    def compute_current(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> float:
        prefix = self._name
        g_syn = synapse_params[f"{prefix}_gS"] * synapse_states[f"{prefix}_s"]
        return g_syn * (post_voltage - post_params[f"{prefix}_e_syn"])
