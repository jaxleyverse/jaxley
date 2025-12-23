# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Optional

import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel
from jaxley.solver_gate import save_exp, solve_gate_exponential


class HH(Channel):
    """Hodgkin-Huxley channel based on Sterratt, Graham, Gillies & Einevoll.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``HH_gNa``
         - 0.12
         - Maximal sodium conductance.
         - S/cm²
       * - ``HH_gK``
         - 0.036
         - Maximal potassium conductance.
         - S/cm²
       * - ``HH_gLeak``
         - 0.0003
         - Leak conductance.
         - S/cm²
       * - ``HH_eNa``
         - 50.0
         - Sodium reversal potential.
         - mV
       * - ``HH_eK``
         - -77.0
         - Potassium reversal potential.
         - mV
       * - ``HH_eLeak``
         - -54.3
         - Leak reversal potential.
         - mV
       * - ``HH_tadj``
         - 1.0
         - Temperature adjustment factor (Q10 scaling).
         - 1
       * - ``temperature``
         - 37.0
         - Absolute temperature.
         - °C

    The following dynamic gating variables are registered in ``channel_states``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Initial value
         - Description
         - Unit
       * - ``HH_m``
         - 0.2
         - Sodium activation gate.
         - 1
       * - ``HH_h``
         - 0.2
         - Sodium inactivation gate.
         - 1
       * - ``HH_n``
         - 0.2
         - Potassium activation gate.
         - 1
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 0.12,
            f"{prefix}_gK": 0.036,
            f"{prefix}_gLeak": 0.0003,
            f"{prefix}_eNa": 50.0,
            f"{prefix}_eK": -77.0,
            f"{prefix}_eLeak": -54.3,
            f"{prefix}_tadj": 1.0,
            f"temperature": 37.0,  # Celsius.
        }
        self.channel_states = {
            f"{prefix}_m": 0.2,
            f"{prefix}_h": 0.2,
            f"{prefix}_n": 0.2,
        }
        self.current_name = f"i_HH"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Return updated HH channel state."""
        prefix = self._name
        m, h, n = states[f"{prefix}_m"], states[f"{prefix}_h"], states[f"{prefix}_n"]
        new_m = solve_gate_exponential(m, delta_t, *m_gate(voltage))
        new_h = solve_gate_exponential(h, delta_t, *h_gate(voltage))
        new_n = solve_gate_exponential(n, delta_t, *n_gate(voltage))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h, f"{prefix}_n": new_n}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> float:
        """Return current through HH channels."""
        prefix = self._name
        m, h, n = states[f"{prefix}_m"], states[f"{prefix}_h"], states[f"{prefix}_n"]

        gNa = params[f"{prefix}_tadj"] * params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2
        gK = params[f"{prefix}_tadj"] * params[f"{prefix}_gK"] * n**4  # S/cm^2
        gLeak = params[f"{prefix}_tadj"] * params[f"{prefix}_gLeak"]  # S/cm^2

        return (
            gNa * (voltage - params[f"{prefix}_eNa"])
            + gK * (voltage - params[f"{prefix}_eK"])
            + gLeak * (voltage - params[f"{prefix}_eLeak"])
        )

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, float]:
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = m_gate(voltage)
        alpha_h, beta_h = h_gate(voltage)
        alpha_n, beta_n = n_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

    def init_params(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the maximal conductances given the temperature."""
        prefix = self._name
        q10 = 2.3
        t = params["temperature"]
        tadj = q10 ** ((t - 37.0) / 10.0)
        return {f"{prefix}_tadj": tadj}


class Na(Channel):
    """Sodium channel based on Sterratt, Graham, Gillies & Einevoll.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``Na_gNa``
         - 0.12
         - Maximal sodium conductance.
         - S/cm²
       * - ``Na_eNa``
         - 50.0
         - Sodium reversal potential.
         - mV
       * - ``Na_tadj``
         - 1.0
         - Temperature adjustment factor (Q10 scaling).
         - 1
       * - ``temperature``
         - 37.0
         - Absolute temperature.
         - °C

    The following dynamic gating variables are registered in ``channel_states``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Initial value
         - Description
         - Unit
       * - ``Na_m``
         - 0.2
         - Sodium activation gate.
         - 1
       * - ``Na_h``
         - 0.2
         - Sodium inactivation gate.
         - 1
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 0.12,
            f"{prefix}_eNa": 50.0,
            f"{prefix}_tadj": 1.0,
            f"temperature": 37.0,  # Celsius.
        }
        self.channel_states = {
            f"{prefix}_m": 0.2,
            f"{prefix}_h": 0.2,
        }
        self.current_name = f"i_Na"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Return updated HH channel state."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        new_m = solve_gate_exponential(m, delta_t, *m_gate(voltage))
        new_h = solve_gate_exponential(h, delta_t, *h_gate(voltage))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> float:
        """Return current through HH channels."""
        prefix = self._name
        m, h = states[f"{prefix}_m"], states[f"{prefix}_h"]
        gNa = params[f"{prefix}_tadj"] * params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2
        return gNa * (voltage - params[f"{prefix}_eNa"])

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, float]:
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(voltage)
        alpha_h, beta_h = self.h_gate(voltage)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
        }

    def init_params(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the maximal conductances given the temperature."""
        prefix = self._name
        q10 = 2.3
        t = params["temperature"]
        tadj = q10 ** ((t - 37.0) / 10.0)
        return {f"{prefix}_tadj": tadj}


class K(Channel):
    """Potassium channel based on Sterratt, Graham, Gillies & Einevoll.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``K_gK``
         - 0.036
         - Maximal potassium conductance.
         - S/cm²
       * - ``K_eK``
         - -77.0
         - Potassium reversal potential.
         - mV
       * - ``K_tadj``
         - 1.0
         - Temperature adjustment factor (Q10 scaling).
         - 1
       * - ``temperature``
         - 37.0
         - Absolute temperature.
         - °C

    The following dynamic gating variables are registered in ``channel_states``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Initial value
         - Description
         - Unit
       * - ``K_n``
         - 0.2
         - Potassium activation gate.
         - 1
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gK": 0.036,
            f"{prefix}_eK": -77.0,
            f"{prefix}_tadj": 1.0,
            f"temperature": 37.0,  # Celsius.
        }
        self.channel_states = {
            f"{prefix}_n": 0.2,
        }
        self.current_name = f"i_K"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Return updated HH channel state."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        new_n = solve_gate_exponential(n, delta_t, *n_gate(voltage))
        return {f"{prefix}_n": new_n}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> float:
        """Return current through HH channels."""
        prefix = self._name
        n = states[f"{prefix}_n"]
        gK = params[f"{prefix}_tadj"] * params[f"{prefix}_gK"] * n**4  # S/cm^2
        return gK * (voltage - params[f"{prefix}_eK"])

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, float]:
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_n, beta_n = self.n_gate(voltage)
        return {f"{prefix}_n": alpha_n / (alpha_n + beta_n)}

    def init_params(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the maximal conductances given the temperature."""
        prefix = self._name
        q10 = 2.3
        t = params["temperature"]
        tadj = q10 ** ((t - 37.0) / 10.0)
        return {f"{prefix}_tadj": tadj}


class Leak(Channel):
    """Leak channel.

    The following parameters are registered in ``channel_params``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``Leak_gLeak``
         - 0.0003
         - Leak conductance.
         - (S/cm²)
       * - ``Leak_eLeak``
         - -54.3
         - Leak reversal potential.
         - mV
       * - ``Leak_tadj``
         - 1.0
         - Temperature adjustment factor (Q10 scaling).
         - 1
       * - ``temperature``
         - 37.0
         - Absolute temperature.
         - °C

    This channel has no internal states.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gLeak": 0.0003,
            f"{prefix}_eLeak": -54.3,
            f"{prefix}_tadj": 1.0,
            f"temperature": 37.0,  # Celsius.
        }
        self.channel_states = {}
        self.current_name = f"i_Leak"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Return updated HH channel state."""
        return {}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> float:
        """Return current through HH channels."""
        prefix = self._name
        gLeak = params[f"{prefix}_tadj"] * params[f"{prefix}_gLeak"]  # S/cm^2
        return gLeak * (voltage - params[f"{prefix}_eLeak"])

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, float]:
        """Initialize the state such at fixed point of gate dynamics."""
        return {}

    def init_params(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Initialize the maximal conductances given the temperature."""
        prefix = self._name
        q10 = 2.3
        t = params["temperature"]
        tadj = q10 ** ((t - 37.0) / 10.0)
        return {f"{prefix}_tadj": tadj}


def m_gate(v):
    alpha = 0.1 * _vtrap(-(v + 40), 10)
    beta = 4.0 * save_exp(-(v + 65) / 18)
    return alpha, beta


def h_gate(v):
    alpha = 0.07 * save_exp(-(v + 65) / 20)
    beta = 1.0 / (save_exp(-(v + 35) / 10) + 1)
    return alpha, beta


def n_gate(v):
    alpha = 0.01 * _vtrap(-(v + 55), 10)
    beta = 0.125 * save_exp(-(v + 65) / 80)
    return alpha, beta


def _vtrap(x, y):
    return x / (save_exp(x / y) - 1.0)
