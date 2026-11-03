# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from warnings import warn

from jax import Array
from jax.typing import ArrayLike


class Channel:
    """Channel base class. All channels inherit from this class.

    A channel in Jaxley is everything that modifies the membrane voltage via its
    current returned by the `compute_current()` method.

    As in NEURON, a `Channel` is considered a distributed process, which means that its
    conductances are to be specified in `S/cm2` and its currents are to be specified in
    `mA/cm2`."""

    _name = None
    channel_params = None
    channel_states = None
    current_name = None
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

    @property
    def name(self) -> Optional[str]:
        """The name of the channel (by default, this is the class name)."""
        return self._name

    def change_name(self, new_name: str):
        """Change the channel name.

        Args:
            new_name: The new name of the channel.

        Returns:
            Renamed channel, such that this function is chainable.
        """
        old_prefix = self._name + "_"
        new_prefix = new_name + "_"

        self._name = new_name
        self.channel_params = {
            (
                new_prefix + key[len(old_prefix) :]
                if key.startswith(old_prefix)
                else key
            ): value
            for key, value in self.channel_params.items()
        }

        self.channel_states = {
            (
                new_prefix + key[len(old_prefix) :]
                if key.startswith(old_prefix)
                else key
            ): value
            for key, value in self.channel_states.items()
        }
        return self

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Return the updated states.

        Args:
            states: All states of the compartment.
            params: Parameters of the channel (conductances in `S/cm2`).
            voltage: Voltage of the compartment in mV.
            delta_t: The time step in ms.

        Returns:
            A dictionary of updated state values.
        """
        raise NotImplementedError

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> Array:
        """Given channel states and voltage, return the current through the channel.

        Args:
            states: All states of the compartment.
            params: Parameters of the channel (conductances in `S/cm2`).
            voltage: Voltage of the compartment in mV.
            delta_t: The time step in ms.

        Returns:
            Current in `mA/cm2`.
        """
        raise NotImplementedError

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Initialize states of channel.

        Args:
            states: All states of the compartment.
            params: Parameters of the channel (conductances in `S/cm2`).
            voltage: Voltage of the compartment in mV.
            delta_t: The time step in ms.

        Returns:
            A initial state that is written into ``module.nodes`` when the user runs
            ``module.init_states()``.
        """
        return {}

    def init_params(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ) -> dict[str, Array]:
        """Initialize the maximal conductances given the temperature.

        Args:
            states: All states of the compartment.
            params: Parameters of the channel (conductances in `S/cm2`).
            voltage: Voltage of the compartment in mV.
            delta_t: The time step in ms.

        Returns:
            Initial parameters that are written into ``module.nodes`` when the user runs
            ``module.init_params()``.
        """
        return {}
