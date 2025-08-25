# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import jax
import jax.numpy as jnp

from jaxley.channels import Channel


class LIF(Channel):
    """Leaky integrate and fire neuron with surrogate gradients.

    This model by default implements a surrogate gradient for the voltage
    threshold, meaning that despite the voltage threshold having a 0 gradient
    over the voltage updating function, it derives a gradient through an
    imagined update function that implements the voltage threshold as
    differentiable.
    """

    def __init__(self, name: str | None = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.name}_g": 1e-1,
            f"{self.name}_vth": -20.0,
            f"{self.name}_vreset": -70.0,
        }
        self.current_name = f"{self.name}_current"
        self.channel_states = {"v": -70.0}

    def update_states(self, states, dt, v, params):
        """Reset the voltage when a spike occurs and log the spike"""
        g = params[f"{self.name}_g"]
        vth = params[f"{self.name}_vth"]
        vreset = params[f"{self.name}_vreset"]

        @jax.custom_gradient
        def over_thresh(v, g, vth, vreset):
            return vreset, lambda gr: (0.0, 0.0, 0.0, gr)

        @jax.custom_gradient
        def under_thresh(v, g, vth, vreset):
            dv = (-g * (v - vreset)) * dt
            return v + dv, lambda gr: (
                gr * (1.0 - (g * dt)),
                gr * -1.0 * (v - vreset) * dt,
                gr * -jnp.exp(v - vth) * (1.0 + (vreset - vth)),
                gr * g * dt,
            )

        v = jax.lax.select(
            (v >= vth)[0],
            over_thresh(v[0], g[0], vth[0], vreset[0]),
            under_thresh(v[0], g[0], vth[0], vreset[0]),
        )
        return {"v": v}

    def compute_current(self, states, v, params):
        return jnp.zeros((1,))

    def init_state(self, states, v, params, delta_t):
        return {}
