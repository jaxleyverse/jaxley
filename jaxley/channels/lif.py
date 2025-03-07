# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional
from warnings import warn

import jax
import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import save_exp


class LIF(Channel):
    """Leaky integrate-and-fire channel."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.name}_gLeak": 0.0003,
            f"{self.name}_eLeak": -55,
            f"{self.name}_vth": -50,
            f"{self.name}_vreset": -55,
            f"{self.name}_Tref": 2.0,  # refractory period in ms
        }
        self.channel_states = {
            f"{self.name}_rem_tref": 0.0,  # time remaining in refractory period
        }
        self.current_name = f"{self.name}_i"

        warn("This channel is not differentiable. Consider using SmoothLIF instead.")

    def update_states(self, states, dt, v, params):
        prefix = self._name
        # Decrease refractory time
        rem_tref = states[f"{prefix}_rem_tref"]
        # Reset refractory timer when spike occurs
        spike_occurred = v >= params[f"{prefix}_vth"]
        new_rem_tref = jax.lax.select(
            spike_occurred,
            params[f"{prefix}_Tref"],  # Reset to full refractory period
            jnp.maximum(0.0, rem_tref - dt),  # Otherwise decrease by dt
        )
        return {f"{prefix}_rem_tref": new_rem_tref}

    def compute_current(self, states, v, params):
        prefix = self._name
        eLeak = params[f"{prefix}_eLeak"]
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2
        vreset = params[f"{prefix}_vreset"]
        vth = params[f"{prefix}_vth"]

        # Check if in refractory period
        is_refractory = states[f"{prefix}_rem_tref"] > 0

        # First check if in refractory period
        v_reset = jax.lax.select(
            is_refractory,
            vreset,
            # If not in refractory, check if above threshold
            jax.lax.select(v > vth, vreset, v),
        )

        return -(v_reset - v) + gLeak * (v_reset - eLeak)

    def init_state(self, states, v, params, delta_t):
        return {f"{self.name}_rem_tref": jnp.zeros_like(v)}
