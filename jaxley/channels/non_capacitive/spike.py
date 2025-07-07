# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp

from jaxley.channels import Channel


class Fire(Channel):
    """Mechanism to reset the voltage when it crosses a threshold.

    When combined with a ``Leak`` channel, this can be used to implement
    leaky-integrate-and-fire neurons.

    Note that, after the voltage is reset by this channel, other channels (or external
    currents), can still modify the membrane voltage `within the same time step`.
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {f"{self.name}_vth": -50, f"{self.name}_vreset": -70}
        self.channel_states = {f"{self.name}_spikes": False}
        self.current_name = f"{self.name}_fire"
        warn(
            "The `Fire` channel does not support surrogate gradients. Its gradient "
            "will be zero after every spike."
        )

    def update_states(self, states, dt, v, params):
        """Reset the voltage when a spike occurs and log the spike"""
        prefix = self._name
        vreset = params[f"{prefix}_vreset"]
        vth = params[f"{prefix}_vth"]

        spike_occurred = v > vth
        v = jax.lax.select(spike_occurred, vreset, v)
        return {"v": v, f"{self.name}_spikes": spike_occurred}

    def compute_current(self, states, v, params):
        return jnp.zeros((1,))

    def init_state(self, states, v, params, delta_t):
        return {}
