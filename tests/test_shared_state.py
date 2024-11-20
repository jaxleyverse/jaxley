# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import Channel
from jaxley.solver_gate import exponential_euler, save_exp, solve_gate_exponential


class Dummy1(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy2)."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"Dummy_s": 0.0}
        self.current_name = f"i_Dummy1"

    @staticmethod
    def update_states(u, dt, voltages, params):
        """Update state."""
        dummy_state = u["Dummy_s"]
        print("dummy_state1", dummy_state)
        return {"Dummy_s": dummy_state + 0.01}

    @staticmethod
    def compute_current(u, voltages, params):
        """Return current."""
        return jnp.zeros_like(voltages)


class Dummy2(Channel):
    """A dummy channel which simply accumulates a state (same state as dummy1)."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"Dummy_s": 0.0}
        self.current_name = f"i_Dummy2"

    @staticmethod
    def update_states(u, dt, voltages, params):
        """Update state."""
        dummy_state = u["Dummy_s"]
        print("dummy_state2", dummy_state)
        return {"Dummy_s": dummy_state + 0.01}

    @staticmethod
    def compute_current(u, voltages, params):
        """Return current."""
        return jnp.zeros_like(voltages)


class CaHVA(Channel):
    """High-Voltage-Activated (HVA) Ca2+ channel"""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gCaHVA": 0.00001,  # S/cm^2
        }
        self.channel_states = {
            f"{self._name}_m": 0.1,  # Initial value for m gating variable
            f"{self._name}_h": 0.1,  # Initial value for h gating variable
            "eCa": 0.0,  # mV, assuming eca for demonstration
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993",
            "mechanism": "HVA Ca2+ channel",
        }

    def update_states(
        self,
        u: Dict[str, jnp.ndarray],
        dt: float,
        voltages: float,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state of gating variables."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        m_new = solve_gate_exponential(ms, dt, *self.m_gate(voltages))
        h_new = solve_gate_exponential(hs, dt, *self.h_gate(voltages))
        return {f"{prefix}_m": m_new, f"{prefix}_h": h_new, "eCa": u["eCa"]}

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        """Compute the current through the channel."""
        prefix = self._name
        ms, hs = u[f"{prefix}_m"], u[f"{prefix}_h"]
        ca_cond = params[f"{prefix}_gCaHVA"] * (ms**2) * hs * 1000
        current = ca_cond * (voltages - u["eCa"])
        return current

    def init_state(self, voltages, params):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltages)
        alpha_h, beta_h = self.h_gate(voltages)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    @staticmethod
    def m_gate(v):
        """Voltage-dependent dynamics for the m gating variable."""
        alpha = (0.055 * (-27 - v + 1e-6)) / (save_exp((-27.0 - v + 1e-6) / 3.8) - 1.0)
        beta = 0.94 * save_exp((-75.0 - v + 1e-6) / 17.0)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        """Voltage-dependent dynamics for the h gating variable."""
        alpha = 0.000457 * save_exp((-13.0 - v) / 50.0)
        beta = 0.0065 / (save_exp((-v - 15.0) / 28.0) + 1.0)
        return alpha, beta


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered)
            f"{self._name}_decay": 80,  # Rate of removal of calcium in ms
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_minCai": 1e-4,  # Minimum intracellular calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["i_Ca"] / 1_000.0
        cai = u["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)

        cai_tau = decay
        cai_inf = minCai + decay * drive_channel
        new_cai = exponential_euler(cai, dt, cai_inf, cai_tau)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, voltages, params):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


def test_shared_state():
    """Test whether two channels can share a state.

    This has to be copied into a notebook and executed with `jax.disable_jit():`."""
    comp1 = jx.Compartment()
    comp1.insert(Dummy1())

    comp2 = jx.Compartment()
    comp2.insert(Dummy2())

    comp3 = jx.Compartment()
    comp3.insert(Dummy1())
    comp3.insert(Dummy2())

    voltages = []
    for comp in [comp1, comp2, comp3]:
        comp.record()
        current = jx.step_current(
            i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
        )
        comp.stimulate(current)

        voltages.append(jx.integrate(comp))


def test_current_as_state_multicompartment():
    """#323 had discovered a bug when currents are only used in a few compartments."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 2)

    branch.comp(0).insert(CaHVA())  # defines current `i_Ca`
    branch.comp(0).insert(CaPump())  # uses `states["i_Ca"]`

    branch.comp(0).record()
    _ = jx.integrate(branch, t_max=1.0)
