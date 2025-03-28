# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from warnings import warn

import jax.numpy as jnp

from jaxley.mechanisms.base import Mechanism


class Channel(Mechanism):
    """Channel base class. All channels inherit from this class.

    A channel in Jaxley is everything that modifies the membrane voltage via its
    current returned by the `compute_current()` method.

    As in NEURON, a `Channel` is considered a distributed process, which means that its
    conductances are to be specified in `S/cm2` and its currents are to be specified in
    `uA/cm2`."""

    name = None
    params = None
    states = None
    current_name = None

    def __init__(self, name: Optional[str] = None):
        contact = (
            "If you have any questions, please reach out via email to "
            "michael.deistler@uni-tuebingen.de or create an issue on Github: "
            "https://github.com/jaxleyverse/jaxley/issues. Thank you!"
        )
        if (
            not hasattr(self, "current_is_in_mA_per_cm2")
            or not self.current_is_in_mA_per_cm2
        ):
            raise ValueError(
                "The channel you are using is deprecated. "
                "In Jaxley version 0.5.0, we changed the unit of the current returned "
                "by `compute_current` of channels from `uA/cm^2` to `mA/cm^2`. Please "
                "update your channel model (by dividing the resulting current by 1000) "
                "and set `self.current_is_in_mA_per_cm2=True` as the first line "
                f"in the `__init__()` method of your channel. {contact}"
            )
        super().__init__(name)
