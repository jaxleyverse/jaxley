from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Synapse:
    synapse_params = None
    synapse_states = None

    def __init__(self):
        pass
    
    @abstractmethod
    def step(
        self, u, dt, voltages, params
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        pass
