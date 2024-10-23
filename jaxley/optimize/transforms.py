# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, List

import jax.numpy as jnp

from jaxley.solver_gate import save_exp
from abc import ABC, abstractmethod
from jax.typing import ArrayLike
from jax import Array


class Transform(ABC):
    @abstractmethod
    def __call__(self, x: ArrayLike) -> Array:
        pass

    @abstractmethod
    def inverse(self, x: ArrayLike) -> Array:
        pass


class SigmoidTransform(Transform):
    """Sigmoid transformation."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        super().__init__()
        self.lower = lower
        self.width = upper - lower

    def __call__(self, x: ArrayLike) -> Array:
        y = 1.0 / (1.0 + save_exp(-x))
        return self.lower + self.width * y

    def inverse(self, y: ArrayLike) -> Array:
        x = (y - self.lower) / self.width
        x = -jnp.log((1.0 / x) - 1.0)  # Corrected the logic for the inverse
        return x


class SoftplusTransform(Transform):
    """Softplus transformation."""

    def __init__(self, lower: ArrayLike) -> None:
        super().__init__()
        self.lower = lower

    def __call__(self, x: ArrayLike) -> Array:
        return jnp.log1p(save_exp(x)) + self.lower

    def inverse(self, y: ArrayLike) -> Array:
        return jnp.log(jnp.exp(y - self.lower) - 1.0)


class NegSoftplusTransform(SoftplusTransform):
    def __init__(self, upper: ArrayLike) -> None:
        super().__init__(upper)

    def __call__(self, x: ArrayLike) -> Array:
        return -super().__call__(-x)

    def inverse(self, y: ArrayLike) -> Array:
        return -super().inverse(-y)


class AffineTransform(Transform):
    def __init__(self, scale: ArrayLike, shift: ArrayLike):
        if jnp.allclose(scale, 0):
            raise ValueError("a cannot be zero, must be invertible")
        self.a = scale
        self.b = shift

    def __call__(self, x: ArrayLike) -> Array:
        return self.a * x + self.b

    def inverse(self, x: ArrayLike) -> Array:
        return (x - self.b) / self.a


class ParamTransform:
    """Parameter transformation utility.

    This class is used to transform parameters from an unconstrained space to a constrained space
    and back. If the range is bounded both from above and below, we use the sigmoid function to
    transform the parameters. If the range is only bounded from below or above, we use softplus.

    Attributes:
        tf_dict: A dictionary of transforms for each parameter.

    """

    def __init__(self, tf_dict: Dict[str, Transform]) -> None:
        """Creates a new ParamTransform object.

        Args:
            transform_dict: A dictionary of transforms for each parameter.
        """

        self.tf_dict = tf_dict

    def forward(self, params: List[Dict[str, ArrayLike]]) -> Dict[str, Array]:
        """Pushes unconstrained parameters through a tf such that they fit the interval.

        Args:
            params: A list of dictionaries with unconstrained parameters.

        Returns:
            A list of dictionaries with transformed parameters.

        """

        tf_params = []
        for param in params:
            key = list(param.keys())[0]

            if key in self.tf_dict:
                tf = self.tf_dict[key](param[key])
                tf_params.append({key: tf})
            else:
                tf_params.append({key: param[key]})

        return tf_params

    def inverse(self, params: List[Dict[str, ArrayLike]]) -> Dict[str, Array]:
        """Takes parameters from within the interval and makes them unconstrained.

        Args:
            params: A list of dictionaries with transformed parameters.

        Returns:
            A list of dictionaries with unconstrained parameters.
        """

        tf_params = []
        for param in params:
            key = list(param.keys())[0]

            if key in self.tf_dict:
                tf_inv = self.tf_dict[key].inverse(param[key])
                tf_params.append({key: tf_inv})
            else:
                tf_params.append({key: param[key]})

        return tf_params
