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
        self.params = {"gNa": 0.12, "eNa": 50.0}
        self.states = {"m": 0.2, "h": 0.2}

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return updated HH channel state."""
        m, h = states["m"], states["h"]

        new_m = solve_gate_exponential(m, dt, *self.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *self.h_gate(v))
        return {"m": new_m, "h": new_h}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through HH channels."""
        m, h = states["m"], states["h"]
        gNa, eNa = params["gNa"], params["eNa"]

        gNa = gNa * (m**3) * h  # S/cm^2
        return gNa * (v - eNa)

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        dt: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize the state such at fixed point of gate dynamics."""
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        return {"m": alpha_m / (alpha_m + beta_m), "h": alpha_h / (alpha_h + beta_h)}

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
        self.params = {"gK": 0.036, "eK": -77.0}
        self.states = {"n": 0.2}

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return updated HH channel state."""
        n = states["n"]

        new_n = solve_gate_exponential(n, dt, *self.n_gate(v))
        return {"n": new_n}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through HH channels."""
        n = states["n"]
        gK, eK = params["gK"], params["eK"]

        gK = gK * n**4  # S/cm^2
        return gK * (v - eK)

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        dt: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize the state such at fixed point of gate dynamics."""
        alpha_n, beta_n = self.n_gate(v)
        return {"n": alpha_n / (alpha_n + beta_n)}

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
        self.params = {"gLeak": 0.0003, "eLeak": -54.3}
        self.states = {}

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
        gLeak, eLeak = params["gLeak"], params["eLeak"]

        return gLeak * (v - eLeak)

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
