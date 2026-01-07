# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler


class Izhikevich(Channel):
    """Izhikevich neuron model.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``a``
         - 0.02
         - Time scale of the recovery variable ``u``.
         - 1/ms
       * - ``b``
         - 0.2
         - Sensitivity of the recovery variable ``u`` to the membrane potential ``v``.
         - 1/ms
       * - ``c``
         - -65.0
         - After-spike reset value of the membrane potential ``v``.
         - mV
       * - ``d``
         - 8
         - After-spike increment of the recovery variable ``u``.
         - mV/ms

    The following states are registered in ``channel_states``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``u``
         - 0.02
         - Recovery variable.
         - mV/ms
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self.name}_a": 0.02,
            f"{self.name}_b": 0.2,
            f"{self.name}_c": -65.0,
            f"{self.name}_d": 8,
        }
        self.channel_states = {f"{self.name}_u": 0.0}
        self.current_name = f"{self.name}_izhikevich"
        warn(
            "The `Izhikevich` channel does not support surrogate gradients. Its "
            "gradient will be zero after every spike."
        )

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Reset the voltage when a spike occurs and log the spike"""
        a = params[f"{self.name}_a"]
        b = params[f"{self.name}_b"]
        c = params[f"{self.name}_c"]
        d = params[f"{self.name}_d"]
        u = states[f"{self.name}_u"]

        # Update the recovery variable u with exponential Euler.
        u = exponential_euler(u, delta_t, b * voltage, 1 / a)

        # Update voltages with Forward Euler because the vectorfield is nonlinear in v.
        dv = (0.04 * voltage**2) + (5 * voltage) + 140 - u
        voltage = voltage + delta_t * dv

        condition = voltage >= 30.0
        voltage = jax.lax.select(condition, c, voltage)
        u = jax.lax.select(condition, u + d, u)
        return {f"{self.name}_u": u, "v": voltage}

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
        prefix = self.name
        return {f"{self.name}_u": params[f"{prefix}_b"] * voltage}
