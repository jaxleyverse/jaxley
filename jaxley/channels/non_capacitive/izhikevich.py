# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler


class Izhikevich(Channel):
    """Izhikevich neuron model."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
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

    def update_states(self, states, dt, v, params):
        """Reset the voltage when a spike occurs and log the spike"""
        a = params[f"{self.name}_a"]
        b = params[f"{self.name}_b"]
        c = params[f"{self.name}_c"]
        d = params[f"{self.name}_d"]
        u = states[f"{self.name}_u"]

        # Update the recovery variable u with exponential Euler.
        u = exponential_euler(u, dt, b * v, 1 / a)

        # Update voltages with Forward Euler because the vectorfield is nonlinear in v.
        dv = (0.04 * v**2) + (5 * v) + 140 - u
        v = v + dt * dv

        condition = v >= 30.0
        v = jax.lax.select(condition, c, v)
        u = jax.lax.select(condition, u + d, u)
        return {f"{self.name}_u": u, "v": v}

    def compute_current(self, states, v, params):
        return jnp.zeros((1,))

    def init_state(self, states, v, params, delta_t):
        prefix = self.name
        return {f"{self.name}_u": params[f"{prefix}_b"] * v}
