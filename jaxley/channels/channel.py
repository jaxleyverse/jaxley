# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Channel:
    """Channel base class. All channels inherit from this class.

    As in NEURON, a `Channel` is considered a distributed process, which means that its
    conductances are to be specified in `S/cm2` and its currents are to be specified in
    `uA/cm2`."""

    _name = None
    channel_params = None
    channel_states = None
    current_name = None

    def __init__(self, name: Optional[str] = None):
        self._name = name if name else self.__class__.__name__

    @property
    def name(self) -> Optional[str]:
        """The name of the channel (by default, this is the class name)."""
        return self._name

    def change_name(self, new_name: str):
        """Change the channel name.

        Args:
            new_name: The new name of the channel.

        Returns:
            Renamed channel, such that this function is chainable.
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
        self, states, dt, v, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return the updated states."""
        raise NotImplementedError

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Given channel states and voltage, return the current through the channel.

        Args:
            states: All states of the compartment.
            v: Voltage of the compartment in mV.
            params: Parameters of the channel (conductances in `S/cm2`).

        Returns:
            Current in `uA/cm2`.
        """
        raise NotImplementedError

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}
