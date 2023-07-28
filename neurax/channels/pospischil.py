from typing import Dict, Optional

import jax.numpy as jnp

from neurax.channels import Channel
from neurax.solver_gate import solve_gate_exponential, solve_inf_gate_exponential

# This is an implementation of Pospischil channels:
# Leak, Na, K, Km, CaT, CaL
# [Pospischil et al. Biological Cybernetics (2008)]


# Helper function
def efun(x):
    """x/[exp(x)-1]

    Args:
        x (float): _description_

    Returns:
        float: x/[exp(x)-1]
    """

    return x / (jnp.exp(x) - 1.0)


class Leak(Channel):
    """Leak current"""

    channel_params = {
        "gl": 1e-4,
        "el": -70.0,
    }
    channel_states = {}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""

        # Multiply with 1000 to convert Siemens to milli Siemens.
        leak_conds = params["gl"] * 1000  # mS/cm^2

        current = leak_conds * (params["el"] - voltages)

        return {}, (jnp.zeros_like(current), current)


class NaChannelPospi(Channel):
    """Sodium channel"""

    channel_params = {"gNa": 50e-3, "eNa": 50.0, "vt": -60.0}
    channel_states = {"m": 0.2, "h": 0.2}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""
        ms, hs = u["m"], u["h"]
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages, params["vt"]))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages, params["vt"]))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params["gNa"] * (new_m**3) * new_h * 1000  # mS/cm^2

        current = na_conds * (params["eNa"] - voltages)

        return {"m": new_m, "h": new_h}, (jnp.zeros_like(current), current)


def _m_gate(v, vt):
    v_alpha = v - vt - 13.0
    alpha = 0.32 * efun(-0.25 * v_alpha) / 0.25

    v_beta = v - vt - 40.0
    beta = 0.28 * efun(0.2 * v_beta) / 0.2
    return alpha, beta


def _h_gate(v, vt):
    v_alpha = v - vt - 17.0
    alpha = 0.128 * jnp.exp(-v_alpha / 18.0)

    v_beta = v - vt - 40.0
    beta = 4.0 / (jnp.exp(-v_beta / 5.0) + 1.0)
    return alpha, beta


class KChannelPospi(Channel):
    """Potassium channel"""

    # KChannelPospi.vt_ should be set to the same value as NaChannelPospi.vt
    # if the Na channel is also present

    channel_params = {"gK": 5e-3, "eK": -90.0, "vt_": -60.0}
    channel_states = {"n": 0.2}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""
        ns = u["n"]
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages, params["vt_"]))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        k_conds = params["gK"] * (new_n**4) * 1000  # mS/cm^2

        current = k_conds * (params["eK"] - voltages)

        return {"n": new_n}, (jnp.zeros_like(current), current)


def _n_gate(v, vt):
    v_alpha = v - vt - 15.0
    alpha = 0.032 * efun(-0.2 * v_alpha) / 0.2

    v_beta = v - vt - 10.0
    beta = 0.5 * jnp.exp(-v_beta / 40.0)
    return alpha, beta


class KmChannelPospi(Channel):
    """Slow M Potassium channel"""

    channel_params = {"gM": 0.004e-3, "taumax": 4000.0, "eM": -90.0}  # ms
    # eM is the reversal potential of K, should be set to eK if another K channel is present

    channel_states = {"p": 0.2}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""
        ps = u["p"]
        new_p = solve_inf_gate_exponential(ps, dt, *_p_gate(voltages, params["taumax"]))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        m_conds = params["gM"] * new_p * 1000  # mS/cm^2
        current = m_conds * (params["eM"] - voltages)

        return {"p": new_p}, (jnp.zeros_like(current), current)


def _p_gate(v, taumax):
    v_p = v + 35.0
    p_inf = 1.0 / (1.0 + jnp.exp(-0.1 * v_p))

    tau_p = taumax / (3.3 * jnp.exp(0.05 * v_p) + jnp.exp(-0.05 * v_p))

    return p_inf, tau_p


class NaKChannelsPospi(Channel):
    """Sodium and Potassium channel"""

    channel_params = {"gNa": 0.05, "eNa": 50.0, "gK": 0.005, "eK": -90.0, "vt": -60}  # 0.005,

    channel_states = {"m": 0.2, "h": 0.2, "n": 0.2}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""
        ms, hs, ns = u["m"], u["h"], u["n"]
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages, params["vt"]))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages, params["vt"]))
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages, params["vt"]))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params["gNa"] * (new_m**3) * new_h * 1000  # mS/cm^2
        k_conds = params["gK"] * (new_n**4) * 1000  # mS/cm^2

        current = na_conds * (params["eNa"] - voltages) + k_conds * (params["eK"] - voltages)

        return {"m": new_m, "h": new_h, "n": new_n}, (jnp.zeros_like(current), current)


class CaLChannelPospi(Channel):
    """L-type Calcium channel"""

    channel_params = {"gCaL": 0.1e-3, "eCa": 120.0}  # S/cm^2
    # eCa is the reversal potential of Ca, should be set to eCa if another Ca channel is present

    channel_states = {"q": 0.2, "r": 0.2}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""
        qs, rs = u["q"], u["r"]
        new_q = solve_gate_exponential(qs, dt, *_q_gate(voltages))
        new_r = solve_gate_exponential(rs, dt, *_r_gate(voltages))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params["gCaL"] * (new_q**2) * new_r * 1000  # mS/cm^2

        current = ca_conds * (params["eCa"] - voltages)

        return {"q": new_q, "r": new_r}, (jnp.zeros_like(current), current)


def _q_gate(v):
    v_alpha = -v - 27.0
    alpha = 0.055 * efun(v_alpha / 3.8) * 3.8

    v_beta = -v - 75.0
    beta = 0.94 * jnp.exp(v_beta / 17.0)
    return alpha, beta


def _r_gate(v):
    v_alpha = -v - 13.0
    alpha = 0.000457 * jnp.exp(v_alpha / 50)

    v_beta = -v - 15.0
    beta = 0.0065 / (jnp.exp(v_beta / 28.0) + 1)
    return alpha, beta


class CaTChannelPospi(Channel):
    """T-type Calcium channel"""

    channel_params = {"gCaT": 0.4e-4, "eCa_": 120.0, "vx": 2.0}  # S/cm^2
    # eCa_ is the reversal potential of Ca, should be set to eCa if another Ca channel is present

    channel_states = {"u": 0.2}

    @staticmethod
    def step(u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]):
        """Return updated channel states and current."""
        us = u["u"]
        new_u = solve_inf_gate_exponential(us, dt, *_u_gate(voltages, params["vx"]))
        s_inf = 1.0 / (1.0 + jnp.exp(-(voltages + params["vx"] + 57.0) / 6.2))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params["gCaT"] * (s_inf**2) * new_u * 1000  # mS/cm^2

        current = ca_conds * (params["eCa_"] - voltages)

        return {"u": new_u}, (jnp.zeros_like(current), current)


def _u_gate(v, vx):
    v_u1 = v + vx + 81.0
    u_inf = 1.0 / (1.0 + jnp.exp(v_u1 / 4))

    tau_u = (30.8 + (211.4 + jnp.exp((v + vx + 113.2) / 5.0))) / (
        3.7 * (1 + jnp.exp((v + vx + 84.0) / 3.2))
    )

    return u_inf, tau_u
