from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import vmap


class Synapse:
    """Base class for a synapse."""

    synapse_params = None
    synapse_states = None

    def __init__(self):
        self.vmapped_compute_current = vmap(
            self.compute_current, in_axes=(None, 0, None, None)
        )

    def update_states(
        u: Dict[str, jnp.ndarray],
        delta_t: float,
        voltages: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        pre_inds: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """ODE update step."""
        raise NotImplementedError

    def compute_current(
        u: Dict[str, jnp.ndarray],
        voltages: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        pre_inds: jnp.ndarray,
    ) -> jnp.ndarray:
        """ODE update step."""
        raise NotImplementedError
