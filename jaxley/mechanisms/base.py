# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax.numpy as jnp


class Mechanism(ABC):
    """A Mechanism is everything that can be inserted into the membrane.

    In Jaxley, a `Mechanism` is everything that modifies the membrane voltage via its
    current returned by the `compute_current()` method.

    A `Mechanism` must implement the the following:
    - `compute_current()` method.
    - `update_states()` method.
    - `init_states()` method.

    Mechamisms can be distributed or point processes.
    """

    name: str = None
    params: dict = None
    states: dict = None
    current_name: str = None
    META: dict = None

    def __init__(self, name=None):
        self.name = self.__class__.__name__  # TODO:should we use .lower() ?
        self.params = {}
        self.states = {}
        self.current_name = f"i_{self.name}"
        self.META = {}

        if name is not None:
            self.change_name(name)

    def change_name(self, new_name: str) -> Mechanism:
        """Change the name of the mechanism.

        Renames `.name` as well as params / states prefixed with `name_`.

        Args:
            new_name: The new name of the mechanism.

        Returns:
            self
        """
        old_prefix = self.name + "_"
        new_prefix = new_name + "_"
        self.name = new_name

        self.params = {
            key.replace(old_prefix, new_prefix): value
            for key, value in self.params.items()
        }

        self.states = {
            key.replace(old_prefix, new_prefix): value
            for key, value in self.states.items()
        }
        return self

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Return the updated states."""
        raise NotImplementedError

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return the current through the mechanism."""
        raise NotImplementedError

    def init_states(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ) -> Dict[str, jnp.ndarray]:
        """Initialize states of mechanism."""
        return NotImplementedError

    def _derivative(self, states, v, params) -> Dict[str, jnp.ndarray]:
        """Return the derivative of the states.

        This method is optional and can be used within `update_states` with a solver of
        choice."""
        raise NotImplementedError
