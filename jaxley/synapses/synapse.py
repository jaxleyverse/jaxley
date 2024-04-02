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
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """ODE update step."""
        raise NotImplementedError

    def compute_current(
        states: Dict[str, jnp.ndarray],
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through synapse in `nA`.

        Args:
            states: States of the synapse.
            pre_voltage: Voltage of the presynaptic compartment, shape `()`.
            post_voltage: Voltage of the postsynaptic compartment, shape `()`.
            params: Parameters of the synapse.

        Returns:
            Current through the synapse in `nA`, shape `()`.
        """
        raise NotImplementedError
