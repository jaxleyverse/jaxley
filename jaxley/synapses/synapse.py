from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Synapse:
    """Base class for a synapse."""

    _name = None
    synapse_params = None
    synapse_states = None

    def __init__(self, name: Optional[str] = None):
        self._name = name if name else self.__class__.__name__

    @property
    def name(self) -> Optional[str]:
        return self._name

    def update_states(
        u: Dict[str, jnp.ndarray],
        delta_t: float,
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """ODE update step."""
        raise NotImplementedError

    def compute_current(
        u: Dict[str, jnp.ndarray],
        voltages: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """ODE update step."""
        raise NotImplementedError
