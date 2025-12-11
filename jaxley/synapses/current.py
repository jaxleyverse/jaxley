# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Dict, Optional
from jax.nn import sigmoid

import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class CurrentSynapse(Synapse):
    r"""A current-based synapse.

    The current of this synapse depends only on the pre-synaptic voltage.
    
    This synapse implements the following equations:

    .. math::

        I = \overline{g}\, \cdot \sigma\!\left( \frac{V_{\text{pre}} - V_{\text{thr}}}{\Delta} \right)

    where :math:`\mathrm{\sigma}(\cdot)` is a nonlinearity such as a ReLU, Sigmoid,
    or TanH. By default, it is a sigmoid, but it can be modified by the user.

    More informally:
    This synaptic current nonlinearly depends on the pre-synaptic voltage.

    The synaptic parameters are:
        - ``gS``: the maximal conductance :math:`\overline{g}` (uS).
        - ``v_th``: the threshold at which the synapse becomes active
          :math:`V_{\text{thr}}` (mV).
        - ``delta``: The inverse of the slope of the activation :math:`\Delta` (mV).

    .. rubric:: Example usage

    Insert a synapse with a sigmoid nonlinearity (the default) and change parameters
    and initial state.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import CurrentSynapse

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        # Connect neurons with the `CurrentSynapse`.
        connect(net.cell(0), net.cell(1), CurrentSynapse())

        # Set parameters.
        net.set("CurrentSynapse_gS", 0.0001)  # Maximal conductance.
        net.set("CurrentSynapse_v_th", -40.0)  # Threshold.
        net.set("CurrentSynapse_delta", 10.0)  # 1 / slope of activation.

    Insert a synapse with a ReLU nonlinearity.

    ::

        import jaxley as jx
        from jaxley.connect import connect
        from jaxley.synapses import CurrentSynapse
        from jax.nn import relu

        cell = jx.Cell()
        net = jx.Network([cell for _ in range(2)])

        # Connect neurons with the `CurrentSynapse`.
        connect(net.cell(0), net.cell(1), CurrentSynapse(relu))
    """

    def __init__(self, nonlinearity: Callable = sigmoid, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,  # uS
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
        current = params[f"{prefix}_gS"] * activation
        return current
