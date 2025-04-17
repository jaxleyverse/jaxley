# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.pumps.pump import Pump


class CaPump(Pump):
    """Calcium dynamics based on Destexhe et al. 1994."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered).
            f"{self._name}_decay": 80,  # Buffering time constant in ms.
            f"{self._name}_depth": 0.1,  # Depth of shell in um.
            f"{self._name}_minCaCon_i": 1e-4,  # Minimum intracell. concentration in mM.
        }
        self.channel_states = {"i_Ca": 1e-8, "CaCon_i": 5e-05}
        self.ion_name = "CaCon_i"
        self.current_name = "i_CaPump"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update states if necessary (but this pump has no states to update)."""
        return {"CaCon_i": states["CaCon_i"], "i_Ca": states["i_Ca"]}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        modified_state,
        params: Dict[str, jnp.ndarray],
    ):
        """Return change of calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = states["i_Ca"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCaCon_i"]

        FARADAY = 96485  # Coulombs per mole.

        # Calculate the contribution of calcium currents to cai change.
        #
        # Note the similarity to the update in `CaCurrentToConcentrationChange`. The
        # main difference (apart from the multiplication with gamma) is that
        # `CaCurrentToConcentrationChange` multiplies by `surface_area / volume`,
        # whereas this channel multiplies by `1/depth`. However, If one assumes calcium
        # to be stored in a ring of width `depth` just inside the membrane, then A/V is:
        # `2r / (r^2 - (r-depth)^2)`. For any r >> depth, this equation is approximately
        # `1/depth`. It holds quite well as long as `r > 5*depth`.
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)

        # Return to steady state ("leak").
        state_decay = (modified_state - minCai) / decay

        # Total calcium update is the sum of the two.
        diff = drive_channel - state_decay
        return -diff

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}
