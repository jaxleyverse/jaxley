# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax.numpy as jnp

from jaxley.mechanisms.base import Mechanism


class Pump(Mechanism):
    """Pump base class. All pumps inherit from this class.

    A pump in Jaxley is everything that modifies the intracellular ion concentrations.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
