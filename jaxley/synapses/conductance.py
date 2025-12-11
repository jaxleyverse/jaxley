# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, Optional

import jax.numpy as jnp
from jax.nn import sigmoid

from jaxley.synapses.synapse import Synapse


class ConductanceSynapse(Synapse):
    r"""A conductance based synapse.

    Unlike the ``CurrentSynapse``, the current of this synapse is conductance-based
    and thus also depends on the post-synaptic voltage. However, unlike the
    ``DynamicSynapse``, this synapse does not have a state.

    This synapse implements the following equations:

    .. math::

        I = \overline{g}\, \cdot \sigma\!\left( \frac{V_{\text{pre}} - V_{\text{thr}}}{\Delta} \right) \cdot (E - V_{\text{post}})

    where :math:`\mathrm{\sigma}(\cdot)` is a nonlinearity such as a ReLU, Sigmoid,
    or TanH. By default, it is a sigmoid, but it can be modified by the user.

    More informally:
    This synaptic conductance nonlinearly depends on the pre-synaptic voltage.
    The current is conductance-based, i.e., it depends on a reversal potential.

    The synaptic parameters are:
        - ``gS``: the maximal conductance :math:`\overline{g}` (uS).
        - ``e_syn``: the reversal potential :math:`E` (mV).
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
        from jaxley.synapses import ConductanceSynapse

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        # Connect neurons with the `ConductanceSynapse`.
        connect(net.cell(0), net.cell(1), ConductanceSynapse())

        # Set parameters.
        net.set("ConductanceSynapse_gS", 0.0001)  # Maximal conductance.
        net.set("ConductanceSynapse_e_syn", 10.0)  # Reversal potential.
        net.set("ConductanceSynapse_v_th", -40.0)  # Threshold.
        net.set("ConductanceSynapse_delta", 10.0)  # 1 / slope of activation.

    Insert a synapse with a ReLU nonlinearity.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import ConductanceSynapse
        from jax.nn import relu

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        # Connect neurons with the `ConductanceSynapse`.
        connect(net.cell(0), net.cell(1), ConductanceSynapse(relu))

    Insert a synapse with a custom nonlinearity.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import ConductanceSynapse

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        def nonlinearity(x):
            return x ** 2

        # Connect neurons with the `ConductanceSynapse`.
        connect(net.cell(0), net.cell(1), ConductanceSynapse(nonlinearity))
    """

    def __init__(self, nonlinearity: Callable = sigmoid, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_v_th": -35.0,  # mV
            f"{prefix}_delta": 10.0,  # mV
        }
        self.synapse_states = {}
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
        return {}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        prefix = self._name
        activation = self.nonlinearity(
            (pre_voltage - params[f"{prefix}_v_th"]) / params[f"{prefix}_delta"]
        )
        g_syn = params[f"{prefix}_gS"] * activation
        return g_syn * (post_voltage - params[f"{prefix}_e_syn"])
