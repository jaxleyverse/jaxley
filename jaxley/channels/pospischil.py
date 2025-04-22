# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import (
    save_exp,
    solve_gate_exponential,
    solve_inf_gate_exponential,
)

# This is an implementation of Pospischil channels:
# Leak, Na, K, Km, CaT, CaL
# [Pospischil et al. Biological Cybernetics (2008)]

__all__ = ["Leak", "Na", "K", "Km", "CaT", "CaL"]


# Helper function
def efun(x):
    """x/[exp(x)-1]

    Args:
        x (float): _description_

    Returns:
        float: x/[exp(x)-1]
    """
    return x / (save_exp(x) - 1.0)


class Leak(Channel):
    """Leak current based on Pospischil et al., 2008."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 1e-4,
            f"{prefix}_eLeak": -70.0,
        }
        self.channel_states = {}
        self.current_name = f"i_{prefix}"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """No state to update."""
        return {}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2
        return gLeak * (v - params[f"{prefix}_eLeak"])

    def init_state(self, states, v, params, delta_t):
        return {}


class Na(Channel):
    """Sodium channel based on Pospischil et al., 2008."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 50e-3,
            "eNa": 50.0,
            "vt": -60.0,  # Global parameter, not prefixed with `Na`.
        }
        self.channel_states = {f"{prefix}_m": 0.2, f"{prefix}_h": 0.2}
        self.current_name = f"i_Na"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *self.m_gate(v, params["vt"]))
        new_h = solve_gate_exponential(h, dt, *self.h_gate(v, params["vt"]))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]

        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2

        current = gNa * (v - params["eNa"])
        return current

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v, params["vt"])
        alpha_h, beta_h = self.h_gate(v, params["vt"])
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

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
        alpha = 0.128 * save_exp(-v_alpha / 18.0)

        v_beta = v - vt - 40.0
        beta = 4.0 / (save_exp(-v_beta / 5.0) + 1.0)
        return alpha, beta


class K(Channel):
    """Potassium channel based on Pospischil et al., 2008."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 5e-3,
            "eK": -90.0,
            "vt": -60.0,  # Global parameter, not prefixed with `Na`.
        }
        self.channel_states = {f"{prefix}_n": 0.2}
        self.current_name = f"i_K"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(n, dt, *self.n_gate(v, params["vt"]))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        n = states[f"{prefix}_n"]

        gK = params[f"{prefix}_gK"] * (n**4)  # S/cm^2

        return gK * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = self.n_gate(v, params["vt"])
        return {f"{prefix}_n": alpha_n / (alpha_n + beta_n)}

    @staticmethod
    def n_gate(v, vt):
        v_alpha = v - vt - 15.0
        alpha = 0.032 * efun(-0.2 * v_alpha) / 0.2

        v_beta = v - vt - 10.0
        beta = 0.5 * save_exp(-v_beta / 40.0)
        return alpha, beta


class Km(Channel):
    """Slow M Potassium channel based on Pospischil et al., 2008."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gKm": 0.004e-3,
            f"{prefix}_taumax": 4000.0,
            f"eK": -90.0,
        }
        self.channel_states = {f"{prefix}_p": 0.2}
        self.current_name = f"i_K"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        p = states[f"{prefix}_p"]
        new_p = solve_inf_gate_exponential(
            p, dt, *self.p_gate(v, params[f"{prefix}_taumax"])
        )
        return {f"{prefix}_p": new_p}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        p = states[f"{prefix}_p"]

        gKm = params[f"{prefix}_gKm"] * p  # S/cm^2
        return gKm * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_p, beta_p = self.p_gate(v, params[f"{prefix}_taumax"])
        return {f"{prefix}_p": alpha_p / (alpha_p + beta_p)}

    @staticmethod
    def p_gate(v, taumax):
        v_p = v + 35.0
        p_inf = 1.0 / (1.0 + save_exp(-0.1 * v_p))

        tau_p = taumax / (3.3 * save_exp(0.05 * v_p) + save_exp(-0.05 * v_p))

        return p_inf, tau_p


class CaL(Channel):
    """L-type Calcium channel based on Pospischil et al., 2008."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaL": 0.1e-3,
            "eCa": 120.0,
        }
        self.channel_states = {f"{prefix}_q": 0.2, f"{prefix}_r": 0.2}
        self.current_name = f"i_Ca"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        q, r = states[f"{prefix}_q"], states[f"{prefix}_r"]
        new_q = solve_gate_exponential(q, dt, *self.q_gate(v))
        new_r = solve_gate_exponential(r, dt, *self.r_gate(v))
        return {f"{prefix}_q": new_q, f"{prefix}_r": new_r}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        q, r = states[f"{prefix}_q"], states[f"{prefix}_r"]
        gCaL = params[f"{prefix}_gCaL"] * (q**2) * r  # S/cm^2

        return gCaL * (v - params["eCa"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_q, beta_q = self.q_gate(v)
        alpha_r, beta_r = self.r_gate(v)
        return {
            f"{prefix}_q": alpha_q / (alpha_q + beta_q),
            f"{prefix}_r": alpha_r / (alpha_r + beta_r),
        }

    @staticmethod
    def q_gate(v):
        v_alpha = -v - 27.0
        alpha = 0.055 * efun(v_alpha / 3.8) * 3.8

        v_beta = -v - 75.0
        beta = 0.94 * save_exp(v_beta / 17.0)
        return alpha, beta

    @staticmethod
    def r_gate(v):
        v_alpha = -v - 13.0
        alpha = 0.000457 * save_exp(v_alpha / 50)

        v_beta = -v - 15.0
        beta = 0.0065 / (save_exp(v_beta / 28.0) + 1)
        return alpha, beta


class CaT(Channel):
    """T-type Calcium channel based on Pospischil et al., 2008."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gCaT": 0.4e-4,
            f"{prefix}_vx": 2.0,
            "eCa": 120.0,  # Global parameter, not prefixed with `CaT`.
        }
        self.channel_states = {f"{prefix}_u": 0.2}
        self.current_name = f"i_Ca"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        u = states[f"{prefix}_u"]
        new_u = solve_inf_gate_exponential(
            u, dt, *self.u_gate(v, params[f"{prefix}_vx"])
        )
        return {f"{prefix}_u": new_u}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        u = states[f"{prefix}_u"]
        s_inf = 1.0 / (1.0 + save_exp(-(v + params[f"{prefix}_vx"] + 57.0) / 6.2))

        gCaT = params[f"{prefix}_gCaT"] * (s_inf**2) * u  # S/cm^2

        return gCaT * (v - params["eCa"])

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_u, beta_u = self.u_gate(v, params[f"{prefix}_vx"])
        return {f"{prefix}_u": alpha_u / (alpha_u + beta_u)}

    @staticmethod
    def u_gate(v, vx):
        v_u1 = v + vx + 81.0
        u_inf = 1.0 / (1.0 + save_exp(v_u1 / 4))

        tau_u = (30.8 + (211.4 + save_exp((v + vx + 113.2) / 5.0))) / (
            3.7 * (1 + save_exp((v + vx + 84.0) / 3.2))
        )

        return u_inf, tau_u
