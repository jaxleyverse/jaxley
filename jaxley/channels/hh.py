from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential


class HH(Channel):
    """Hodgkin-Huxley channel."""

    channel_params = {
        "hh_gNa": 0.12,
        "hh_gK": 0.036,
        "hh_gLeak": 0.0003,
        "hh_eNa": 50.0,
        "hh_eK": -77.0,
        "hh_eLeak": -54.3,
    }
    channel_states = {"hh_m": 0.2, "hh_h": 0.2, "hh_n": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return updated HH channel state."""
        ms, hs, ns = u["hh_m"], u["hh_h"], u["hh_n"]
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages))
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages))
        return {"hh_m": new_m, "hh_h": new_h, "hh_n": new_n}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current through HH channels."""
        ms, hs, ns = u["hh_m"], u["hh_h"], u["hh_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params["hh_gNa"] * (ms**3) * hs * 1000  # mS/cm^2
        kd_conds = params["hh_gK"] * ns**4 * 1000  # mS/cm^2
        leak_conds = params["hh_gLeak"] * 1000  # mS/cm^2

        return (
            na_conds * (voltages - params["hh_eNa"])
            + kd_conds * (voltages - params["hh_eK"])
            + leak_conds * (voltages - params["hh_eLeak"])
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
