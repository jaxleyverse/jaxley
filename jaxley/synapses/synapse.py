# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations
from abc import ABC
from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Synapse(ABC):
    """Base class for a synapse.

    As in NEURON, a `Synapse` is considered a point process, which means that its
    conductances are to be specified in `uS` and its currents are to be specified in
    `nA`.
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

    def change_name(self, new_name: str) -> Synapse:
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
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """ODE update step.

        Args:
            states: States of the synapse.
            delta_t: Time step in `ms`.
            pre_voltage: Voltage of the presynaptic compartment, shape `()`.
            post_voltage: Voltage of the postsynaptic compartment, shape `()`.
            params: Parameters of the synapse. Conductances in `uS`.

        Returns:
            Updated states."""
        raise NotImplementedError

    def compute_current(
        states: Dict[str, jnp.ndarray],
        pre_voltage: jnp.ndarray,
        post_voltage: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Return current through one synapse in `nA`.

        Internally, we use `jax.vmap` to vectorize this function across many synapses.

        Args:
            states: States of the synapse.
            pre_voltage: Voltage of the presynaptic compartment, shape `()`.
            post_voltage: Voltage of the postsynaptic compartment, shape `()`.
            params: Parameters of the synapse. Conductances in `uS`.

        Returns:
            Current through the synapse in `nA`, shape `()`.
        """
        raise NotImplementedError
