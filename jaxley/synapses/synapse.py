from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Synapse:
    """Base class for a synapse."""

    synapse_params = None
    synapse_states = None

    def __init__(self):
        pass

    @staticmethod
    def step(
        u: Dict[str, jnp.ndarray],
        delta_t: float,
        voltages: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        pre_inds: jnp.ndarray,
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """ODE update step."""
        raise NotImplementedError
