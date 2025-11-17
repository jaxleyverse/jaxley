# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax.numpy as jnp

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
    """

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {f"{self.name}_vth": -50, f"{self.name}_vreset": -70}
        self.channel_states = {f"{self.name}_spikes": False}
        self.current_name = f"{self.name}_fire"

    def update_states(self, states, dt, v, params):
        """Reset the voltage when a spike occurs and log the spike"""
        prefix = self._name
        vreset = params[f"{prefix}_vreset"]
        vth = params[f"{prefix}_vth"]

        spike_occurred = heaviside(v - vth)
        v = (v * (1 - heaviside(v - vth))) + (vreset * heaviside(v - vth))

        return {"v": v, f"{self.name}_spikes": spike_occurred}

    def compute_current(self, states, v, params):
        return 0

    def init_state(self, states, v, params, delta_t):
        return {}
