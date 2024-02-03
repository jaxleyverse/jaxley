from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential


class HH(Channel):
    """Hodgkin-Huxley channel."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 0.12,
            f"{prefix}_gK": 0.036,
            f"{prefix}_gLeak": 0.0003,
            f"{prefix}_eNa": 50.0,
            f"{prefix}_eK": -77.0,
            f"{prefix}_eLeak": -54.3,
        }
        self.channel_states = {
            f"{prefix}_m": 0.2,
            f"{prefix}_h": 0.2,
            f"{prefix}_n": 0.2,
        }

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return updated HH channel state."""
        prefix = self._name
        ms, hs, ns = u[f"{prefix}_m"], u[f"{prefix}_h"], u[f"{prefix}_n"]
        new_m = solve_gate_exponential(ms, dt, *self.m_gate(voltages))
        new_h = solve_gate_exponential(hs, dt, *self.h_gate(voltages))
        new_n = solve_gate_exponential(ns, dt, *self.n_gate(voltages))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h, f"{prefix}_n": new_n}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current through HH channels."""
        prefix = self._name
        ms, hs, ns = u[f"{prefix}_m"], u[f"{prefix}_h"], u[f"{prefix}_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params[f"{prefix}_gNa"] * (ms**3) * hs * 1000  # mS/cm^2
        kd_conds = params[f"{prefix}_gK"] * ns**4 * 1000  # mS/cm^2
        leak_conds = params[f"{prefix}_gLeak"] * 1000  # mS/cm^2

        return (
            na_conds * (voltages - params[f"{prefix}_eNa"])
            + kd_conds * (voltages - params[f"{prefix}_eK"])
            + leak_conds * (voltages - params[f"{prefix}_eLeak"])
        )

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltages)
        alpha_h, beta_h = self.h_gate(voltages)
        alpha_n, beta_n = self.n_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.1 * _vtrap(-(v + 40), 10)
        beta = 4.0 * jnp.exp(-(v + 65) / 18)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 0.07 * jnp.exp(-(v + 65) / 20)
        beta = 1.0 / (jnp.exp(-(v + 35) / 10) + 1)
        return alpha, beta

    @staticmethod
    def n_gate(v):
        alpha = 0.01 * _vtrap(-(v + 55), 10)
        beta = 0.125 * jnp.exp(-(v + 65) / 80)
        return alpha, beta


def _vtrap(x, y):
    return x / (jnp.exp(x / y) - 1.0)
