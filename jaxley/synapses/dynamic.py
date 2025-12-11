# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from jax.nn import sigmoid

from jaxley.solver_gate import exponential_euler, save_exp
from jaxley.synapses.synapse import Synapse


class DynamicSynapse(Synapse):
    r"""A dynamic (state-based) synapse.

    Unlike the ``ConductanceSynapse``, this synapse contains a synaptic state. However,
    unlike in the ``IonotropicSynapse``, the synaptic state approaches its steady-state
    with a constant (i.e., not voltage dependent) time constant.

    This synapse implements the following equations:

    .. math::

        I = \overline{g}\, \cdot s\, \cdot (E - V_{\text{post}})

    .. math::

        \tau\, \cdot s = \sigma\!\left(\frac{V_{\text{pre}} - V_{\text{thr}}}{\Delta}\right) - s,

    where :math:`\mathrm{\sigma}(\cdot)` is a nonlinearity such as a ReLU, Sigmoid,
    or TanH. By default, it is a sigmoid, but it can be modified by the user.

    More informally:
    This synapse has a state which defines its conductance. The state approaches
    a nonlinear map of the voltage with a time constant which is not voltage-dependent.
    The current is conductance-based, i.e., it depends on a reversal potential.

    The synaptic parameters are:
        - ``gS``: the maximal conductance :math:`\overline{g}` (uS).
        - ``e_syn``: the reversal potential :math:`E` (mV).
        - ``k_minus``: the rate constant :math:`1/\tau` (:math:`ms^{-1}`).
        - ``v_th``: the threshold at which the synapse becomes active
          :math:`V_{\text{thr}}` (mV).
        - ``delta``: The inverse of the slope of the activation :math:`\Delta` (mV).

    The synaptic state is:
        - ``s``: the activity level of the synapse :math:`\in [0, 1]`.

    .. rubric:: Example usage

    Insert a synapse with a sigmoid nonlinearity (the default) and change parameters
    and initial state.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import DynamicSynapse

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        # Connect neurons with the `DynamicSynapse`.
        connect(net.cell(0), net.cell(1), DynamicSynapse())

        # Set parameters.
        net.set("DynamicSynapse_gS", 0.0001)  # Maximal conductance.
        net.set("DynamicSynapse_e_syn", 10.0)  # Reversal potential.
        net.set("DynamicSynapse_k_minus", 0.1)  # tau = 10.0 ms
        net.set("DynamicSynapse_v_th", -40.0)  # Threshold.
        net.set("DynamicSynapse_delta", 10.0)  # 1 / slope of activation.

        # Set the initial state.
        net.set("DynamicSynapse_s", 0.1)

    Insert a synapse with a ReLU nonlinearity.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import DynamicSynapse
        from jax.nn import relu

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        # Connect neurons with the `DynamicSynapse`.
        connect(net.cell(0), net.cell(1), DynamicSynapse(relu))

    Insert a synapse with a custom nonlinearity.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import DynamicSynapse

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        def nonlinearity(x):
            return x ** 2

        # Connect neurons with the `DynamicSynapse`.
        connect(net.cell(0), net.cell(1), DynamicSynapse(nonlinearity))
    """

    def __init__(self, nonlinearity: Callable = sigmoid, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_k_minus": 0.025,  # 1/ms
            f"{prefix}_v_th": -35.0,  # mV
            f"{prefix}_delta": 10.0,  # mV
        }
        self.synapse_states = {f"{prefix}_s": 0.0}
        self.nonlinearity = nonlinearity

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
        s = states[f"{prefix}_s"]

        s_inf = self.nonlinearity((pre_voltage - v_th) / delta)
        s_tau = 1.0 / params[f"{prefix}_k_minus"]

        new_s = exponential_euler(s, delta_t, s_inf, s_tau)
        return {f"{prefix}_s": new_s}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        g_syn = params[f"{prefix}_gS"] * states[f"{prefix}_s"]
        return g_syn * (post_voltage - params[f"{prefix}_e_syn"])
