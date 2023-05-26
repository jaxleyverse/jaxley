from typing import Dict, Optional

import jax.numpy as jnp
from neurax.modules.channel import Channel
from neurax.solver_gate import solve_gate_exponential


class HHChannel(Channel):
    """Hodgkin-Huxley channel."""

    def step(self, u, dt, voltages):
        """Return updated HH channel state and current."""
        ms, hs, ns = (u[0], u[1], u[2])
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages))
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = self.params[0] * (ms**3) * hs * 1000  # mS/cm^2
        kd_conds = self.params[1] * ns**4 * 1000  # mS/cm^2
        leak_conds = self.params[2] * 1000  # mS/cm^2

        voltage_term = na_conds + kd_conds + leak_conds

        e_na = 50.0
        e_kd = -77.0
        e_leak = -54.3
        constant_term = na_conds * e_na + kd_conds * e_kd + leak_conds * e_leak

        return jnp.stack([new_m, new_h, new_n]), (voltage_term, constant_term)


def _m_gate(v):
    alpha = 0.1 * _vtrap(-(v + 40), 10)
    beta = 4.0 * jnp.exp(-(v + 65) / 18)
    return alpha, beta


def _h_gate(v):
    alpha = 0.07 * jnp.exp(-(v + 65) / 20)
    beta = 1.0 / (jnp.exp(-(v + 35) / 10) + 1)
    return alpha, beta


def _n_gate(v):
    alpha = 0.01 * _vtrap(-(v + 55), 10)
    beta = 0.125 * jnp.exp(-(v + 65) / 80)
    return alpha, beta


def _vtrap(x, y):
    return x / (jnp.exp(x / y) - 1.0)
