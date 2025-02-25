# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

import jax.numpy as jnp

from jaxley.pumps.pump import Pump


def convert_current_to_molarity_change(
    current: float, valence: float, radius: float
) -> float:
    r"""Compute the change in molarity of an ion given current returned by a channel.

    This function returns the _change_ in molarity M, i.e. :math:`\frac{dM}{dt}`.
    Consistent with the units of Jaxley (mM for concentration, ms for time), it returns
    the unit milli-mole / liter / milli-second.

    This function implements the formula

    .. math::

        dM/dt = \frac{I \cdot A}{V \cdot F \cdot z},

    where:

    * \( I \) is the current per surface area,
    * \( A \) is the surface area of a compartment,
    * \( V \) is the volume of a compartment,
    * \( F \) is the Faraday constant,
    * \( z \) is the valence of the ion.

    Args:
        current: Current through an ion channel (returned by the `compute_current`
            method of a `Channel`) in mS/cm^2.
        valence: Valence of the ion (e.g., 2 for calcium, 1 for sodium).
        radius: Radius of the compartment in micrometer.

    Returns:
        The change in molarity M, dM/dt, in unit milli-mole/liter/milli-second.
    """
    FARADAY = 96485.3329  # Coulombs per mole.

    # surface_area = 2 * pi * radius * length  # um^2
    # volume = pi * radius ** 2 * length  # um^3
    # surface_per_volume = surface_area / volume = 2 / radius
    surface_per_volume = 2 / radius

    # Where does 1e4 come from? `volume`: current * area / volume ->
    # mA / cm^2 * um^2 / um^3 = mA / cm^2 / um = mA / dm^2 * (1e1)^2 / dm * 1e5 =
    # mA / dm^3 * 1e7 = mA / liter * 1e7
    #
    # Finally, FARADAY is A * second / mol. However, our time is in units ms (not s),
    # so we need to divide by 1000 -> 1e7 / 1e3 = 1e4
    return current * surface_per_volume * 1e4 / FARADAY / valence


class CaFaradayConcentrationChange(Pump):
    r"""Update the intracellular calcium ion concentration depending on calcium current.

    This channel implements Faraday's first law of electrolysis to update the
    intracellular calcium concentration based on calcium current. Faraday's law relates
    how a current (e.g., through a channel) impacts the number of ions. Mathematically:

    .. math::

        n = \frac{I \cdot t}{F \cdot z}

    Taking the derivative with respect to time:

    .. math::

        \frac{dn}{dt} = \frac{I}{F \cdot z}

    where:

    * \( n \) is the amount of substance (number of moles),
    * \( I \) is the current,
    * \( t \) is time,
    * \( F \) is the Faraday constant,
    * \( z \) is the valence of the ion.

    To obtain concentration \( c \) from the amount of substance \( n \), we divide by
    the volume:

    .. math::

        \frac{dc}{dt} = \frac{1}{V} \cdot \frac{dn}{dt} = \frac{I}{F \cdot z \cdot V}

    In Jaxley, the current is given in mS/cm2, so we first have to multiply the current
    by the surface area of a compartment.

    The update is fully passive (i.e., there is no active pump). As such, it is even
    possible that ion concentration can become negative (because we do not enforce
    that calcium currents stop when no more ions are available).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {}
        self.channel_states = {"i_Ca": 1e-8, "CaCon_i": 5e-05}
        self.ion_name = "CaCon_i"
        self.current_name = "i_CaCurrent"
        self.META = {"mechanism": "Calcium change"}

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update states if necessary (but this mechanism has no states to update)."""
        return {"CaCon_i": states["CaCon_i"], "i_Ca": states["i_Ca"]}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        modified_state,
        params: Dict[str, jnp.ndarray],
    ):
        """Return change of calcium concentration as the calcium current."""
        # 2.0 is valence of calcium.
        return convert_current_to_molarity_change(states["i_Ca"], 2.0, params["radius"])

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}
