# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler


class Rate(Channel):
    """Rate-based, unit-less, neuron model."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {f"{self.name}_tau": 1.0}
        self.channel_states = {}
        self.current_name = f"{self.name}_rate"

    def update_states(self, states, dt, v, params):
        """Voltages get pulled towards zero."""
        tau = params[f"{self.name}_tau"]
        return {"v": exponential_euler(v, dt, 0.0, tau)}

    def compute_current(self, states, v, params):
        return jnp.zeros((1,))

    def init_state(self, states, v, params, delta_t):
        return {}
