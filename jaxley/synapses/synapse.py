# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Optional

from jax import Array
from jax.typing import ArrayLike


class Synapse:
    """Base class for a synapse.

    As in NEURON, a `Synapse` is considered a point process, which means that its
    conductances are to be specified in `uS` and its currents are to be specified in
    `nA`.
    """

    _name = None
    synapse_params = None
    synapse_states = None
    # If True, the slope of `compute_current` w.r.t. voltage is taken with
    # `jax.jvp` instead of a finite difference (two calls `1e-3` apart). This is
    # more accurate for any current: the finite difference loses
    # ~`eps * |I| / 1e-3` to cancellation in float32, its reverse pass splits each
    # cotangent into two `1/1e-3`-scaled terms that nearly cancel, and for
    # nonlinear currents it yields a secant rather than the tangent. It is opt-in
    # because it changes results relative to the finite-difference default and
    # requires `compute_current` to be differentiable in `v` (no `custom_vjp`,
    # `stop_gradient`, or constant-valued guards around singularities).
    exact_linearization = False

    def __init__(self, name: Optional[str] = None):
        self._name = name if name else self.__class__.__name__
        self.node_params = {}
        self.node_states = {}

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
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> dict[str, Array]:
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
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> Array:
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
