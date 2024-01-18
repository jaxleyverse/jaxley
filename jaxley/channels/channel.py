from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import vmap


class Channel:
    _channel_name = None
    channel_params = None
    channel_states = None

    def __init__(self, channel_name: Optional[str] = None):
        self._channel_name = channel_name if channel_name else self.__class__.__name__
        self.vmaped_update_states = vmap(self.update_states, in_axes=(0, None, 0, 0))
        self.vmapped_compute_current = vmap(
            self.compute_current, in_axes=(None, 0, None)
        )

    @property
    def channel_name(self) -> Optional[str]:
        return self._channel_name

    def change_name(self, new_name: str):
        self._channel_name = new_name
        self.channel_params = {
            new_name + key[key.find("_") :]: value
            for key, value in self.channel_params.items()
            if "_" in key
        }
        self.channel_states = {
            new_name + key[key.find("_") :]: value
            for key, value in self.channel_states.items()
            if "_" in key
        }

    def update_states(
        self, u, dt, voltages, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass

    def compute_current(
        self, u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        pass
