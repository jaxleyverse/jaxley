from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Channel:
    channel_params = None
    channel_states = None

    @abstractmethod
    def step(
        self, u, dt, voltages, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass
