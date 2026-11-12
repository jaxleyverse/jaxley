# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Mechanisms that are only used by tests.

Channels from `jaxley-mech` are imported from there instead of being copied here.
"""

from typing import Optional

import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.pumps import Pump
from jaxley.solver_gate import exponential_euler, save_exp


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
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

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = states["i_Ca"] / 1_000.0
        cai = states["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)

        cai_tau = decay
        cai_inf = minCai + decay * drive_channel
        new_cai = exponential_euler(cai, delta_t, cai_inf, cai_tau)

        return {f"CaCon_i": new_cai}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eCa": 0.0, "CaCon_i": 5e-05, "CaCon_e": 2.0}
        self.current_name = f"i_Ca"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = states["CaCon_i"]
        Cao = states["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vCa = C * jnp.log(Cao / Cai)
        return {"eCa": vCa, "CaCon_i": Cai, "CaCon_e": Cao}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class KCA11(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_q10_ch": 3,
            f"{prefix}_q10_ch0": 22,
            "celsius": 22,
        }
        self.channel_states = {f"{prefix}_m": 0.02, "CaCon_i": 1e-4}
        self.current_name = f"i_K"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update state."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        q10 = params[f"{prefix}_q10_ch"] ** (
            (params["celsius"] - params[f"{prefix}_q10_ch0"]) / 10
        )
        cai = states["CaCon_i"]
        new_m = exponential_euler(m, delta_t, *self.m_gate(voltage, cai, q10))
        return {f"{prefix}_m": new_m}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Return current."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        g = 0.03 * m * 1000  # mS/cm^2
        return g * (voltage + 80.0)

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        q10 = params[f"{prefix}_q10_ch"] ** (
            (params["celsius"] - params[f"{prefix}_q10_ch0"]) / 10
        )
        cai = states["CaCon_i"]
        m_inf, _ = self.m_gate(voltage, cai, q10)
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v, cai, q10):
        cai = cai * 1e3
        v_half = -66 + 137 * save_exp(-0.3044 * cai) + 30.24 * save_exp(-0.04141 * cai)
        alpha = 25.0

        beta = 0.075 / save_exp((v - v_half) / 10)
        m_inf = alpha / (alpha + beta)
        tau_m = 1.0 * q10
        return m_inf, tau_m


class NaPump(Pump):
    """A dummy pump which is only used for internal `Jaxley` tests.

    Modeled after the calcium channel in Destexhe et al. 1994.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered).
            f"{self._name}_decay": 80,  # Buffering time constant in ms.
            f"{self._name}_depth": 0.1,  # Depth of shell in um.
            f"{self._name}_minNai": 1e-4,  # Minimum intracell. ca concentration in mM.
        }
        self.channel_states = {"i_Na": 1e-8, "NaCon_i": 5e-05}
        self.ion_name = "NaCon_i"
        self.current_name = "i_NaPump"
        self.META = {
            "reference": "Made up by Jaxley developers (@michaeldeistler)",
            "mechanism": "Sodium dynamics",
        }

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        modified_state: Array,
        delta_t: float,
    ):
        """Update states if necessary (but this pump has no states to update)."""
        return {"NaCon_i": states["NaCon_i"], "i_Na": states["i_Na"]}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        modified_state: Array,
        delta_t: float,
    ):
        """Return change of calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = states["i_Na"] / 1_000.0
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minNai"]

        FARADAY = 96485  # Coulombs per mole.

        # Calculate the contribution of calcium currents to cai change.
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)
        state_decay = (modified_state - minCai) / decay
        diff = drive_channel - state_decay
        return -diff

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        modified_state: Array,
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}


class NaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eNa": 0.0, "NaCon_i": 5e-05, "NaCon_e": 3.0}
        self.current_name = f"i_Na"
        self.META = {"ion": "Na"}

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Nai = states["NaCon_i"]
        Nao = states["NaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vNa = C * jnp.log(Nao / Nai)
        return {"eNa": vNa, "NaCon_i": Nai, "NaCon_e": Nao}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the state at fixed point of gate dynamics."""
        return {}
