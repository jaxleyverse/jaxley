# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import save_exp


class LIF(Channel):
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

    def update_states(self, states, dt, v, params):
        prefix = self._name
        # Decrease refractory time
        rem_tref = states[f"{prefix}_rem_tref"]
        # Reset refractory timer when spike occurs
        new_rem_tref = jnp.where(
            v >= params[f"{prefix}_vth"],
            params[f"{prefix}_Tref"],  # Reset to full refractory period
            jnp.maximum(0.0, rem_tref - dt),  # Otherwise decrease by dt
        )
        return {f"{prefix}_rem_tref": new_rem_tref}

    def compute_current(self, states, v, params):
        prefix = self._name
        eLeak = params[f"{prefix}_eLeak"]
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2

        # Check if in refractory period
        is_refractory = states[f"{prefix}_rem_tref"] > 0

        # If in refractory period, force voltage to reset
        # If not in refractory and above threshold, reset voltage
        # Otherwise keep current voltage
        v_reset = jnp.where(
            is_refractory,
            params[f"{prefix}_vreset"],
            jnp.where(v > params[f"{prefix}_vth"], params[f"{prefix}_vreset"], v),
        )

        return -(v_reset - v) + gLeak * (v_reset - eLeak)

    def init_state(self, states, v, params, delta_t):
        return {f"{self.name}_rem_tref": jnp.zeros_like(v)}


class SmoothLIF(Channel):
    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self.name}_gLeak": 0.0003,
            f"{self.name}_eLeak": -55,
            f"{self.name}_vth": -50,
            f"{self.name}_vreset": -55,
            f"{self.name}_Tref": 2.0,  # refractory period in ms
            f"{self.name}_beta": 1.0e8,  # smoothing parameter
        }
        self.channel_states = {
            f"{self.name}_rem_tref": 0.0,  # time remaining in refractory period
            f"{self.name}_spike_prob": 0.0,  # continuous spike probability
        }
        self.current_name = f"{self.name}_i"

    def sigmoid(self, x, beta):
        """Smooth approximation of the Heaviside step function"""
        return 1 / (1 + save_exp(-beta * x))

    def update_states(self, states, dt, v, params):
        prefix = self._name
        beta = params[f"{prefix}_beta"]
        vth = params[f"{prefix}_vth"]
        Tref = params[f"{prefix}_Tref"]
        rem_tref = states[f"{prefix}_rem_tref"]

        # Compute smooth spike probability
        spike_prob = self.sigmoid(v - vth, beta)

        new_rem_tref = spike_prob * Tref + (1 - spike_prob) * jnp.maximum(
            0.0, rem_tref - dt
        )

        return {f"{prefix}_rem_tref": new_rem_tref, f"{prefix}_spike_prob": spike_prob}

    def compute_current(self, states, v, params):
        prefix = self._name
        eLeak = params[f"{prefix}_eLeak"]
        gLeak = params[f"{prefix}_gLeak"]
        vreset = params[f"{prefix}_vreset"]
        rem_tref = states[f"{prefix}_rem_tref"]
        Tref = params[f"{prefix}_Tref"]
        spike_prob = states[f"{prefix}_spike_prob"]

        # Smooth transition between normal voltage and reset voltage
        v_effective = (1 - spike_prob) * v + spike_prob * vreset

        # Blend v_effective towards v_reset based on remaining refractory time
        ref_decay = save_exp(-rem_tref / Tref * 1e3)
        v_effective = (1 - ref_decay) * vreset + ref_decay * v_effective

        # Compute current with smooth voltage
        return -(v_effective - v) + gLeak * (v_effective - eLeak)

    def init_state(self, states, v, params, delta_t):
        return {
            f"{self.name}_rem_tref": jnp.zeros_like(v),
            f"{self.name}_spike_prob": jnp.zeros_like(v),
        }
