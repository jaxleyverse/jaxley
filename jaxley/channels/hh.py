# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential


class Na(Channel):
    """Hodgkin-Huxley Sodium channel."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.params = {
            f"{prefix}_gNa": 0.12,
            f"{prefix}_eNa": 50.0,
        }
        self.states = {
            f"{prefix}_m": 0.2,
            f"{prefix}_h": 0.2,
        }
        self.current_name = f"i_Na"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return updated HH channel state."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, dt, *self.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *self.h_gate(v))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through HH channels."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]

        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2

        return gNa * (v - params[f"{prefix}_eNa"])

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        dt: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        alpha = 0.1 * _vtrap(-(v + 40), 10)
        beta = 4.0 * save_exp(-(v + 65) / 18)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 0.07 * save_exp(-(v + 65) / 20)
        beta = 1.0 / (save_exp(-(v + 35) / 10) + 1)
        return alpha, beta


class K(Channel):
    """Hodgkin-Huxley Potassium channel."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.params = {
            f"{prefix}_gK": 0.036,
            f"{prefix}_eK": -77.0,
        }
        self.states = {
            f"{prefix}_n": 0.2,
        }
        self.current_name = f"i_K"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return updated HH channel state."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(n, dt, *self.n_gate(v))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through HH channels."""
        prefix = self._name
        n = states[f"{prefix}_n"]

        gK = params[f"{prefix}_gK"] * n**4  # S/cm^2

        return gK * (v - params[f"{prefix}_eK"])

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        dt: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = self.n_gate(v)
        return {f"{prefix}_n": alpha_n / (alpha_n + beta_n)}

    @staticmethod
    def n_gate(v):
        alpha = 0.01 * _vtrap(-(v + 55), 10)
        beta = 0.125 * save_exp(-(v + 65) / 80)
        return alpha, beta


class Leak(Channel):
    """Hodgkin-Huxley Leak channel."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.params = {
            f"{prefix}_gLeak": 0.0003,
            f"{prefix}_eLeak": -54.3,
        }
        self.states = {}
        self.current_name = f"i_Leak"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return updated HH channel state."""
        return {}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through HH channels."""
        prefix = self._name
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2

        return gLeak * (v - params[f"{prefix}_eLeak"])

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        dt: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize the state such at fixed point of gate dynamics."""
        return {}


class HH(Channel):
    """Hodgkin-Huxley channel."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        self.Na = Na(self._name)
        self.K = K(self._name)
        self.Leak = Leak(self._name)
        self.channels = [self.Na, self.K, self.Leak]

        self.params = {
            **self.Na.params,
            **self.K.params,
            **self.Leak.params,
        }

        self.states = {
            **self.Na.states,
            **self.K.states,
            **self.Leak.states,
        }

        self.current_name = f"i_HH"

    def change_name(self, new_name: str):
        self._name = new_name
        for channel in self.channels:
            channel.change_name(new_name)
        return self

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return updated HH channel state."""
        new_states = {}
        for channel in self.channels:
            new_states.update(channel.update_states(states, dt, v, params))
        return new_states

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through HH channels."""
        current = 0
        for channel in self.channels:
            current += channel.compute_current(states, v, params)
        return current

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        dt: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize the state such at fixed point of gate dynamics."""
        init_states = {}
        for channel in self.channels:
            init_states.update(channel.init_state(states, v, params, dt))
        return init_states


def _vtrap(x, y):
    return x / (save_exp(x / y) - 1.0)
