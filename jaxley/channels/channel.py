from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from jax import vmap

import jax.numpy as jnp


class Channel:
    channel_params = None
    channel_states = None

    def __init__(self):
        self.vmapped_compute_current = vmap(self.compute_current, in_axes=(None, 0, None))

    @staticmethod
    def update_states(
        u, dt, voltages, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass

    @staticmethod
    def compute_current(
        u: Dict[str, jnp.ndarray], voltages, params: Dict[str, jnp.ndarray]
    ):
        pass
