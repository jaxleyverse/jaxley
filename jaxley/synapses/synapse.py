# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp


class Synapse:
    """Base class for a synapse.

    As in NEURON, a `Synapse` is considered a point process, which means that its
    conductances are to be specified in `uS` and its currents are to be specified in
    `nA`.
    """

    _name = None
    synapse_params = None
    synapse_states = None

    def __init__(self, name: Optional[str] = None):
        self._name = name if name else self.__class__.__name__

    @property
    def name(self) -> Optional[str]:
        return self._name

    def change_name(self, new_name: str):
        """Change the synapse name.

        Args:
            new_name: The new name of the channel.

        Returns:
            Renamed channel, such that this function is chainable.
        """
        old_prefix = self._name + "_"
        new_prefix = new_name + "_"

        self._name = new_name
        self.synapse_params = {
            (
                new_prefix + key[len(old_prefix) :]
                if key.startswith(old_prefix)
                else key
            ): value
            for key, value in self.synapse_params.items()
        }

        self.synapse_states = {
            (
                new_prefix + key[len(old_prefix) :]
                if key.startswith(old_prefix)
                else key
            ): value
            for key, value in self.synapse_states.items()
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
