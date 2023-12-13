from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential, solve_inf_gate_exponential

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


class LeakPospi(Channel):
    """Leak current"""

    channel_params = {
        "leakpospi_gl": 1e-4,
        "leakpospi_el": -70.0,
    }
    channel_states = {}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """No state to update."""
        return {}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        # Multiply with 1000 to convert Siemens to milli Siemens.
        leak_conds = params["leakpospi_gl"] * 1000  # mS/cm^2
        return leak_conds * (voltages - params["leakpospi_el"])


class NaPospi(Channel):
    """Sodium channel"""

    channel_params = {"napospi_gNa": 50e-3, "napospi_eNa": 50.0, "pospi_vt": -60.0}
    channel_states = {"napospi_m": 0.2, "napospi_h": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        ms, hs = u["napospi_m"], u["napospi_h"]
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages, params["pospi_vt"]))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages, params["pospi_vt"]))
        return {"napospi_m": new_m, "napospi_h": new_h}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        ms, hs = u["m"], u["h"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params["gNa"] * (ms**3) * hs * 1000  # mS/cm^2

        current = na_conds * (voltages - params["eNa"])
        return current


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


class KPospi(Channel):
    """Potassium channel"""

    channel_params = {"kpospi_gK": 5e-3, "kpospi_eK": -90.0, "pospi_vt": -60.0}
    channel_states = {"kpospi_n": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        ns = u["kpospi_n"]
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages, params["vt_"]))
        return {"kpospi_n": new_n}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        ns = u["kpospi_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        k_conds = params["kpospi_gK"] * (ns**4) * 1000  # mS/cm^2

        return k_conds * (voltages - params["kpospi_eK"])


def _n_gate(v, vt):
    v_alpha = v - vt - 15.0
    alpha = 0.032 * efun(-0.2 * v_alpha) / 0.2

    v_beta = v - vt - 10.0
    beta = 0.5 * jnp.exp(-v_beta / 40.0)
    return alpha, beta


class KmPospi(Channel):
    """Slow M Potassium channel"""

    channel_params = {
        "kmpospi_gM": 0.004e-3,
        "kmpospi_taumax": 4000.0,
        "pospi_eM": -90.0,
    }
    channel_states = {"kmpospi_p": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        ps = u["kmpospi_p"]
        new_p = solve_inf_gate_exponential(
            ps, dt, *_p_gate(voltages, params["kmpospi_taumax"])
        )
        return {"kmpospi_p": new_p}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        ps = u["kmpospi_p"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        m_conds = params["kmpospi_gM"] * ps * 1000  # mS/cm^2
        return m_conds * (voltages - params["kmpospi_eM"])


def _p_gate(v, taumax):
    v_p = v + 35.0
    p_inf = 1.0 / (1.0 + jnp.exp(-0.1 * v_p))

    tau_p = taumax / (3.3 * jnp.exp(0.05 * v_p) + jnp.exp(-0.05 * v_p))

    return p_inf, tau_p


class NaKPospi(Channel):
    """Sodium and Potassium channel"""

    channel_params = {
        "nakpospi_gNa": 0.05,
        "nakpospi_eNa": 50.0,
        "nakpospi_gK": 0.005,
        "nakpospi_eK": -90.0,
        "pospi_vt": -60,
    }

    channel_states = {"nakpospi_m": 0.2, "nakpospi_h": 0.2, "nakpospi_n": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        ms, hs, ns = u["nakpospi_m"], u["nakpospi_h"], u["nakpospi_n"]
        new_m = solve_gate_exponential(ms, dt, *_m_gate(voltages, params["pospi_vt"]))
        new_h = solve_gate_exponential(hs, dt, *_h_gate(voltages, params["pospi_vt"]))
        new_n = solve_gate_exponential(ns, dt, *_n_gate(voltages, params["pospi_vt"]))
        return {"nakpospi_m": new_m, "nakpospi_h": new_h, "nakpospi_n": new_n}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        ms, hs, ns = u["nakpospi_m"], u["nakpospi_h"], u["nakpospi_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params["nakpospi_gNa"] * (ms**3) * hs * 1000  # mS/cm^2
        k_conds = params["nakpospi_gK"] * (ns**4) * 1000  # mS/cm^2

        return na_conds * (voltages - params["nakpospi_eNa"]) + k_conds * (
            voltages - params["nakpospi_eK"]
        )


class CaLPospi(Channel):
    """L-type Calcium channel"""

    channel_params = {"calpospi_gCaL": 0.1e-3, "pospi_eCa": 120.0}
    channel_states = {"calpospi_q": 0.2, "calpospi_r": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        qs, rs = u["calpospi_q"], u["calpospi_r"]
        new_q = solve_gate_exponential(qs, dt, *_q_gate(voltages))
        new_r = solve_gate_exponential(rs, dt, *_r_gate(voltages))
        return {"calpospi_q": new_q, "calpospi_r": new_r}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        qs, rs = u["calpospi_q"], u["calpospi_r"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params["calpospi_gCaL"] * (qs**2) * rs * 1000  # mS/cm^2

        return ca_conds * (voltages - params["pospi_eCa"])


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


class CaTPospi(Channel):
    """T-type Calcium channel"""

    channel_params = {"catpospi_gCaT": 0.4e-4, "pospi_eCa": 120.0, "catpospi_vx": 2.0}
    channel_states = {"catpospi_u": 0.2}

    @staticmethod
    def update_states(
        u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        us = u["catpospi_u"]
        new_u = solve_inf_gate_exponential(
            us, dt, *_u_gate(voltages, params["catpospi_vx"])
        )
        return {"catpospi_u": new_u}

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        us = u["catpospi_u"]
        s_inf = 1.0 / (1.0 + jnp.exp(-(voltages + params["catpospi_vx"] + 57.0) / 6.2))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params["catpospi_gCaT"] * (s_inf**2) * us * 1000  # mS/cm^2

        return ca_conds * (voltages - params["pospi_eCa"])


def _u_gate(v, vx):
    v_u1 = v + vx + 81.0
    u_inf = 1.0 / (1.0 + jnp.exp(v_u1 / 4))

    tau_u = (30.8 + (211.4 + jnp.exp((v + vx + 113.2) / 5.0))) / (
        3.7 * (1 + jnp.exp((v + vx + 84.0) / 3.2))
    )

    return u_inf, tau_u
