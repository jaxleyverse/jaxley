# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax.numpy as jnp
from jax import Array

from jaxley.pumps import Pump


class CaNernstReversal(Pump):
    """Compute Calcium reversal from inner and outer concentration of calcium.

    This Pump has no additional parameters.

    The following states are registered in ``channel_states``:

    .. list-table::
       :widths: 25 15 50 10
       :header-rows: 1

       * - Name
         - Default
         - Description
         - Unit
       * - ``eCa``
         - 0.0
         - The reversal potential of calcium.
         - mV
       * - ``CaCon_i``
         - 5e-5
         - The intracellular calcium concentration.
         - mM
       * - ``CaCon_e``
         - 2.0
         - The extracellular calcium concentration.
         - mM
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eCa": 0.0, "CaCon_i": 5e-05, "CaCon_e": 2.0}
        # Note that the `self.ion_name` does not matter here, because `compute_current`
        # returns 0.0 (and `self.ion_name` only sets which ion concentration the
        # current should be added to).
        self.ion_name = "CaCon_i"
        self.current_name = f"i_Ca"
        self.META = {"ion": "Ca"}

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
