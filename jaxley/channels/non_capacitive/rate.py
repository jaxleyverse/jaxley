# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler


class Rate(Channel):
    """Rate-based, unit-less, neuron model.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``tau``
         - 1
         - Time constant of the neuron (unitless).
         - 1

    The channel has no internal states.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {f"{self.name}_tau": 1.0}
        self.channel_states = {}
        self.current_name = f"{self.name}_rate"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Voltages get pulled towards zero."""
        tau = params[f"{self.name}_tau"]
        return {"v": exponential_euler(voltage, delta_t, 0.0, tau)}

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
