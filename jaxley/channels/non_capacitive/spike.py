# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.solver_gate import heaviside


class Fire(Channel):
    """Mechanism to reset the voltage when it crosses a threshold.

    When combined with a ``Leak`` channel, this can be used to implement
    leaky-integrate-and-fire neurons.

    Note that, after the voltage is reset by this channel, other channels (or external
    currents), can still modify the membrane voltage `within the same time step`.

    Note as well that this function implements a surrogate gradient through the
    use of the ``heaviside`` function in ``update_states()``. This allows the user
    to perform gradient descent on networks using this channel despite the ``Fire``
    mechanism being non-differentiable.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``Fire_vth``
         - -50.0
         - Threshold for firing.
         - mV
       * - ``Fire_vreset``
         - -70.0
         - The reset for the voltage after a spike.
         - mV

    The following states are registered in ``channel_states``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``Fire_spikes``
         - False
         - Whether or not a spike occured.
         - 1
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {f"{self.name}_vth": -50, f"{self.name}_vreset": -70}
        self.channel_states = {f"{self.name}_spikes": False}
        self.current_name = f"{self.name}_fire"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Reset the voltage when a spike occurs and log the spike"""
        prefix = self._name
        vreset = params[f"{prefix}_vreset"]
        vth = params[f"{prefix}_vth"]

        spike_occurred = heaviside(voltage - vth)
        voltage = (voltage * (1 - heaviside(voltage - vth))) + (
            vreset * heaviside(voltage - vth)
        )

        return {"v": voltage, f"{self.name}_spikes": spike_occurred}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        return {}
