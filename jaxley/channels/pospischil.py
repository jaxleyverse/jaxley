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


class Leak(Channel):
    """Leak current"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gl": 1e-4,
            f"{prefix}_el": -70.0,
        }
        self.channel_states = {}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """No state to update."""
        return {}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        # Multiply with 1000 to convert Siemens to milli Siemens.
        leak_conds = params[f"{prefix}_gl"] * 1000  # mS/cm^2
        return leak_conds * (voltages - params[f"{prefix}_el"])


class Na(Channel):
    """Sodium channel"""

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 50e-3,
            f"{prefix}_eNa": 50.0,
            "vt": -60.0,  # Global parameter, not prefixed with `Na`.
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        new_m = solve_gate_exponential(ms, dt, *self.m_gate(voltages, params["vt"]))
        new_h = solve_gate_exponential(hs, dt, *self.h_gate(voltages, params["vt"]))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params[f"{prefix}_gNa"] * (ms**3) * hs * 1000  # mS/cm^2

        current = na_conds * (voltages - params[f"{prefix}_eNa"])
        return current

    @staticmethod
    def m_gate(v, vt):
        v_alpha = v - vt - 13.0
        alpha = 0.32 * efun(-0.25 * v_alpha) / 0.25

        v_beta = v - vt - 40.0
        beta = 0.28 * efun(0.2 * v_beta) / 0.2
        return alpha, beta

    @staticmethod
    def h_gate(v, vt):
        v_alpha = v - vt - 17.0
        alpha = 0.128 * jnp.exp(-v_alpha / 18.0)

        v_beta = v - vt - 40.0
        beta = 4.0 / (jnp.exp(-v_beta / 5.0) + 1.0)
        return alpha, beta


class K(Channel):
    """Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 5e-3,
            f"{prefix}_eK": -90.0,
            "vt": -60.0,  # Global parameter, not prefixed with `Na`.
        }
        self.channel_states = {f"{prefix}_n": 0.2}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        ns = u[f"{prefix}_n"]
        new_n = solve_gate_exponential(ns, dt, self.n_gate(voltages, params["vt"]))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        ns = u[f"{prefix}_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        k_conds = params[f"{prefix}_gK"] * (ns**4) * 1000  # mS/cm^2

        return k_conds * (voltages - params[f"{prefix}_eK"])

    @staticmethod
    def n_gate(v, vt):
        v_alpha = v - vt - 15.0
        alpha = 0.032 * efun(-0.2 * v_alpha) / 0.2

        v_beta = v - vt - 10.0
        beta = 0.5 * jnp.exp(-v_beta / 40.0)
        return alpha, beta


class Km(Channel):
    """Slow M Potassium channel"""

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gM": 0.004e-3,
            f"{prefix}_taumax": 4000.0,
            f"eM": -90.0,
        }
        self.channel_states = {f"{prefix}_p": 0.2}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        ps = u[f"{prefix}_p"]
        new_p = solve_inf_gate_exponential(
            ps, dt, *self.p_gate(voltages, params[f"{prefix}_taumax"])
        )
        return {f"{prefix}_p": new_p}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        ps = u[f"{prefix}_p"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        m_conds = params[f"{prefix}_gM"] * ps * 1000  # mS/cm^2
        return m_conds * (voltages - params["eM"])

    @staticmethod
    def p_gate(v, taumax):
        v_p = v + 35.0
        p_inf = 1.0 / (1.0 + jnp.exp(-0.1 * v_p))

        tau_p = taumax / (3.3 * jnp.exp(0.05 * v_p) + jnp.exp(-0.05 * v_p))

        return p_inf, tau_p


class NaK(Channel):
    """Sodium and Potassium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 0.05,
            f"{prefix}_eNa": 50.0,
            f"{prefix}_gK": 0.005,
            f"{prefix}_eK": -90.0,
            "vt": -60,  # Global parameter, not prefixed with `NaK`.
        }
        self.channel_states = {
            f"{prefix}_m": 0.2,
            f"{prefix}_h": 0.2,
            f"{prefix}_n": 0.2,
        }

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""

        prefix = self._name
        ms, hs, ns = u[f"{prefix}_m"], u[f"{prefix}_h"], u[f"{prefix}_n"]
        new_m = solve_gate_exponential(ms, dt, *Na.m_gate(voltages, params["vt"]))
        new_h = solve_gate_exponential(hs, dt, *Na.h_gate(voltages, params["vt"]))
        new_n = solve_gate_exponential(ns, dt, *K.n_gate(voltages, params["vt"]))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h, f"{prefix}_n": new_n}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        ms, hs, ns = u[f"{prefix}_m"], u[f"{prefix}_h"], u[f"{prefix}_n"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        na_conds = params[f"{prefix}_gNa"] * (ms**3) * hs * 1000  # mS/cm^2
        k_conds = params[f"{prefix}_gK"] * (ns**4) * 1000  # mS/cm^2

        return na_conds * (voltages - params[f"{prefix}_eNa"]) + k_conds * (
            voltages - params[f"{prefix}_eK"]
        )


class CaL(Channel):
    """L-type Calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaL": 0.1e-3,
            "eCa": 120.0,
        }
        self.channel_states = {f"{prefix}_q": 0.2, f"{prefix}_r": 0.2}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        qs, rs = u[f"{prefix}_q"], u[f"{prefix}_r"]
        new_q = solve_gate_exponential(qs, dt, *self.q_gate(voltages))
        new_r = solve_gate_exponential(rs, dt, *self.r_gate(voltages))
        return {f"{prefix}_q": new_q, f"{prefix}_r": new_r}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        qs, rs = u[f"{prefix}_q"], u[f"{prefix}_r"]

        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params[f"{prefix}_gCaL"] * (qs**2) * rs * 1000  # mS/cm^2

        return ca_conds * (voltages - params["eCa"])

    @staticmethod
    def q_gate(v):
        v_alpha = -v - 27.0
        alpha = 0.055 * efun(v_alpha / 3.8) * 3.8

        v_beta = -v - 75.0
        beta = 0.94 * jnp.exp(v_beta / 17.0)
        return alpha, beta

    @staticmethod
    def r_gate(v):
        v_alpha = -v - 13.0
        alpha = 0.000457 * jnp.exp(v_alpha / 50)

        v_beta = -v - 15.0
        beta = 0.0065 / (jnp.exp(v_beta / 28.0) + 1)
        return alpha, beta


class CaT(Channel):
    """T-type Calcium channel"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaT": 0.4e-4,
            f"{prefix}_vx": 2.0,
            "eCa": 120.0,  # Global parameter, not prefixed with `CaT`.
        }
        self.channel_states = {f"{prefix}_u": 0.2}

    def update_states(
        self, u: Dict[str, jnp.ndarray], dt, voltages, params: Dict[str, jnp.ndarray]
    ):
        """Update state."""
        prefix = self._name
        us = u[f"{prefix}_u"]
        new_u = solve_inf_gate_exponential(
            us, dt, *self.u_gate(voltages, params[f"{prefix}_vx"])
        )
        return {f"{prefix}_u": new_u}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        us = u[f"{prefix}_u"]
        s_inf = 1.0 / (1.0 + jnp.exp(-(voltages + params[f"{prefix}_vx"] + 57.0) / 6.2))

        # Multiply with 1000 to convert Siemens to milli Siemens.
        ca_conds = params[f"{prefix}_gCaT"] * (s_inf**2) * us * 1000  # mS/cm^2

        return ca_conds * (voltages - params["eCa"])

    @staticmethod
    def u_gate(v, vx):
        v_u1 = v + vx + 81.0
        u_inf = 1.0 / (1.0 + jnp.exp(v_u1 / 4))

        tau_u = (30.8 + (211.4 + jnp.exp((v + vx + 113.2) / 5.0))) / (
            3.7 * (1 + jnp.exp((v + vx + 84.0) / 3.2))
        )

        return u_inf, tau_u
