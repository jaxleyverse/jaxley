# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from jax import Array
from jax.typing import ArrayLike


class Pump:
    """Pump base class. All pumps inherit from this class.

    A pump in Jaxley is everything that modifies the intracellular ion concentrations.
    """

    _name: str
    channel_params: dict[str, Array]
    channel_states: dict[str, Array]
    current_name: str

    def __init__(self, name: Optional[str] = None):
        self._name = name if name else self.__class__.__name__

    @property
    def name(self) -> Optional[str]:
        """The name of the channel (by default, this is the class name)."""
        return self._name

    def change_name(self, new_name: str):
        """Change the pump name.

        Args:
            new_name: The new name of the pump.

        Returns:
            Renamed pump, such that this function is chainable.
        """
        old_prefix = self._name + "_"
        new_prefix = new_name + "_"

        self._name = new_name
        self.channel_params = {
            (
                new_prefix + key[len(old_prefix) :]
                if key.startswith(old_prefix)
                else key
            ): value
            for key, value in self.channel_params.items()
        }

        self.channel_states = {
            (
                new_prefix + key[len(old_prefix) :]
                if key.startswith(old_prefix)
                else key
            ): value
            for key, value in self.channel_states.items()
        }
        return self

    def update_states(
        self, states: dict[str, Array], v, params: dict[str, Array]
    ) -> None:
        """Update the states of the pump."""
        raise NotImplementedError

    def compute_current(
        self, states: dict[str, Array], v, params: dict[str, Array]
    ) -> None:
        """Given channel states and voltage, return the change in ion concentration.

        Args:
            states: All states of the compartment.
            v: Voltage of the compartment in mV.
            params: Parameters of the channel (conductances in `S/cm2`).

        Returns:
            Ion concentration change in `mM`.
        """
        raise NotImplementedError

    def init_state(
        self,
        states: dict[str, ArrayLike],
        v: ArrayLike,
        params: dict[str, ArrayLike],
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}
