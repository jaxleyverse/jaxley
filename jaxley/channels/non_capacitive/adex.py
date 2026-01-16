# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""
Unified AdEx (Adaptive Exponential Integrate-and-Fire) neuron model.

References:
    Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model
    as an effective description of neuronal activity.
"""

from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp

__all__ = ["AdEx"]


class AdEx(Channel):
    """
    Unified Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.

    This channel implements the full AdEx dynamics:
    - Leak current
    - Exponential spike mechanism
    - Adaptation current
    - Spike detection and reset

    The membrane equation is:
        C_m * dV/dt = -g_L * (V - E_L) + g_L * delta_T * exp((V - V_T) / delta_T) - w + I_ext
        tau_w * dw/dt = a * (V - E_L) - w

    When V > v_threshold:
        V -> v_reset
        w -> w + b

    Parameters (set via channel_params):
        g_L: Leak conductance (default: 10.0 nS)
        E_L: Leak reversal potential (default: -70.0 mV)
        v_T: Spike threshold potential (default: -50.0 mV)
        delta_T: Spike slope factor (default: 2.0 mV)
        v_threshold: Spike detection threshold (default: 20.0 mV)
        v_reset: Reset potential after spike (default: -70.0 mV)
        tau_w: Adaptation time constant (default: 30.0 ms)
        a: Subthreshold adaptation (default: 2.0 nS)
        b: Spike-triggered adaptation (default: 0.0 pA)
    """

    def __init__(self, name: Optional[str] = None, surrogate_warning=True):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)

        prefix = self._name

        # All AdEx parameters
        # Note: Conductances are in S/cm^2, currents in mA/cm^2
        # These defaults give reasonable tonic spiking behavior
        self.channel_params = {
            # Leak parameters
            # f"capacitance": 200.,
            f"{prefix}_g_L": 10,  # S/cm^2 (leak conductance density)
            f"{prefix}_E_L": -70.0,  # mV
            # Exponential spike parameters
            f"{prefix}_v_T": -50.0,  # mV (spike threshold)
            f"{prefix}_delta_T": 2.0,  # mV (spike slope factor)
            # Spike reset parameters
            f"{prefix}_v_threshold": 0.0,  # mV (detection threshold)
            f"{prefix}_v_reset": -58.0,  # mV (reset potential)
            # Adaptation parameters
            f"{prefix}_tau_w": 30.0,  # ms
            f"{prefix}_a": 2,  # S/cm^2 (sub-threshold adaptation)
            f"{prefix}_b": 0,  # mA/cm^2 (spike-triggered adaptation)
        }

        # AdEx state variables
        self.channel_states = {
            f"{prefix}_w": 0.0,  # Adaptation current
            f"{prefix}_spikes": False,  # Spike indicator
        }

        self.current_name = f"i_{prefix}"

        if surrogate_warning:
            warn(
                f"The {self.name} channel does not support surrogate gradients. "
                "Its gradient will be zero after every spike. "
                "Use AdExSurrogate for differentiable spiking."
            )

    def update_states(
        self, states: dict[str, Array], dt: float, v: Array, params: dict[str, Array]
    ) -> dict[str, Array]:
        """
        Update adaptation variable, integrate voltage, and handle spike reset.

        Like Izhikevich, all dynamics are handled here. This allows proper clamping
        of the exponential spike mechanism.

        Args:
            states: Current state dictionary
            dt: Time step (ms)
            v: Membrane potential (mV)
            params: Parameter dictionary

        Returns:
            Dictionary with updated states: v, w, spikes
        """
        prefix = self._name

        # Get parameters
        C_m = params[f"capacitance"]
        g_L = params[f"{prefix}_g_L"]
        E_L = params[f"{prefix}_E_L"]
        v_T = params[f"{prefix}_v_T"]
        v_reset = params[f"{prefix}_v_reset"]
        v_threshold = params[f"{prefix}_v_threshold"]
        delta_T = params[f"{prefix}_delta_T"]

        tau_w = params[f"{prefix}_tau_w"]
        a = params[f"{prefix}_a"]
        b = params[f"{prefix}_b"]

        # Get adaptation current
        w = states[f"{prefix}_w"]

        # Update adaptation variable with exponential Euler
        w = exponential_euler(w, dt, a * (v - E_L), tau_w)

        # Leak current
        i_leak = g_L * (v - E_L)

        # Exponential spike current (clamped to prevent overflow)
        exp_arg = (v - v_T) / delta_T
        exp_arg = jnp.minimum(exp_arg, 10.0)  # Clamp at exp(10) ≈ 22000
        i_exp = g_L * delta_T * save_exp(exp_arg)

        # Total derivative
        dv = (-i_leak + i_exp - w) / C_m
        # Forward Euler
        v = v + dt * dv

        # Check for spike and reset
        spike_occurred = (
            v >= v_threshold
        )  # ToDo: replace this with heaviside function in v1.0.0
        v = jax.lax.select(spike_occurred, v_reset, v)
        w = jax.lax.select(spike_occurred, w + b, w)

        return {
            "v": v,
            f"{prefix}_w": w,
            f"{prefix}_spikes": spike_occurred.astype(jnp.float32),
        }

    def compute_current(
        self, states: dict[str, Array], v: Array, params: dict[str, Array]
    ) -> Array:
        """
        Return zero current since all dynamics are handled in update_states.

        Like Izhikevich, AdEx integrates voltage directly in update_states,
        so compute_current returns zero to avoid double-integration.
        """
        return jnp.zeros((1,))

    def init_state(
        self,
        states: dict[str, Array],
        v: Array,
        params: dict[str, Array],
        delta_t: float,
    ) -> dict[str, Array]:
        """ """
        return {}
