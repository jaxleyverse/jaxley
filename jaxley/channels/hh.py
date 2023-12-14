from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential


class HH(Channel):
    """Hodgkin-Huxley channel."""

    channel_params = {
        "HH_gNa": 0.12,
        "HH_gK": 0.036,
        "HH_gLeak": 0.0003,
        "HH_eNa": 50.0,
        "HH_eK": -77.0,
        "HH_eLeak": -54.3,
    }
    channel_states = {"HH_m": 0.2, "HH_h": 0.2, "HH_n": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return updated HH channel state."""
        ms, hs, ns = u["HH_m"], u["HH_h"], u["HH_n"]
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages))
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages))
        return {"HH_m": new_m, "HH_h": new_h, "HH_n": new_n}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current through HH channels."""
        ms, hs, ns = u["HH_m"], u["HH_h"], u["HH_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params["HH_gNa"] * (ms**3) * hs * 1000  # mS/cm^2
        kd_conds = params["HH_gK"] * ns**4 * 1000  # mS/cm^2
        leak_conds = params["HH_gLeak"] * 1000  # mS/cm^2

        return (
            na_conds * (voltages - params["HH_eNa"])
            + kd_conds * (voltages - params["HH_eK"])
            + leak_conds * (voltages - params["HH_eLeak"])
        )


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
