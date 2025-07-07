# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from jaxley.solver_gate import save_exp


class Transform(ABC):
    def __call__(self, x: ArrayLike) -> Array:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Array:
        pass

    @abstractmethod
    def inverse(self, x: ArrayLike) -> Array:
        pass


class SigmoidTransform(Transform):
    """Sigmoid transformation."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, upper].

        Args:
            lower (ArrayLike): Lower bound of the interval.
            upper (ArrayLike): Upper bound of the interval.
        """
        super().__init__()
        self.lower = lower
        self.width = upper - lower

    def forward(self, x: ArrayLike) -> Array:
        y = 1.0 / (1.0 + save_exp(-x))
        return self.lower + self.width * y

    def inverse(self, y: ArrayLike) -> Array:
        x = (y - self.lower) / self.width
        x = -jnp.log((1.0 / x) - 1.0)
        return x


class SoftplusTransform(Transform):
    """Softplus transformation."""

    def __init__(self, lower: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, inf).

        Args:
            lower (ArrayLike): Lower bound of the interval.
        """
        super().__init__()
        self.lower = lower

    def forward(self, x: ArrayLike) -> Array:
        return jnp.log1p(save_exp(x)) + self.lower

    def inverse(self, y: ArrayLike) -> Array:
        return jnp.log(save_exp(y - self.lower) - 1.0)


class NegSoftplusTransform(SoftplusTransform):
    """Negative softplus transformation."""

    def __init__(self, upper: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval (-inf, upper].

        Args:
            upper (ArrayLike): Upper bound of the interval.
        """
        super().__init__(upper)

    def forward(self, x: ArrayLike) -> Array:
        return -super().forward(-x)

    def inverse(self, y: ArrayLike) -> Array:
        return -super().inverse(-y)


class AffineTransform(Transform):
    def __init__(self, scale: ArrayLike, shift: ArrayLike):
        """This transform rescales and shifts the input.

        Args:
            scale (ArrayLike): Scaling factor.
            shift (ArrayLike): Additive shift.

        Raises:
            ValueError: Scale needs to be larger than 0
        """
        if jnp.allclose(scale, 0):
            raise ValueError("a cannot be zero, must be invertible")
        self.a = scale
        self.b = shift

    def forward(self, x: ArrayLike) -> Array:
        return self.a * x + self.b

    def inverse(self, x: ArrayLike) -> Array:
        return (x - self.b) / self.a


class ChainTransform(Transform):
    """Chaining together multiple transformations"""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """A chain of transformations

        Args:
            transforms (Sequence[Transform]): Transforms to apply
        """
        super().__init__()
        self.transforms = transforms

    def forward(self, x: ArrayLike) -> Array:
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def inverse(self, y: ArrayLike) -> Array:
        for transform in reversed(self.transforms):
            y = transform.inverse(y)
        return y


class MaskedTransform(Transform):
    def __init__(self, mask: ArrayLike, transform: Transform) -> None:
        """A masked transformation

        Args:
            mask (ArrayLike): Which elements to transform
            transform (Transform): Transformation to apply
        """
        super().__init__()
        self.mask = mask
        self.transform = transform

    def forward(self, x: ArrayLike) -> Array:
        return jnp.where(self.mask, self.transform.forward(x), x)

    def inverse(self, y: ArrayLike) -> Array:
        return jnp.where(self.mask, self.transform.inverse(y), y)


class CustomTransform(Transform):
    """Custom transformation."""

    def __init__(self, forward_fn: Callable, inverse_fn: Callable) -> None:
        """A custom transformation using a user-defined froward and
        inverse function

        Args:
            forward_fn (Callable): Forward transformation
            inverse_fn (Callable): Inverse transformation
        """
        super().__init__()
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn

    def forward(self, x: ArrayLike) -> Array:
        return self.forward_fn(x)

    def inverse(self, y: ArrayLike) -> Array:
        return self.inverse_fn(y)


class ParamTransform:
    """Parameter transformation utility.

    This class is used to transform parameters usually from an unconstrained space to
    a constrained space and back (because most biophysical parameter are bounded).
    The user can specify a PyTree of transforms that are applied to the parameters.

    Attributes:
        tf_dict: A PyTree of transforms for each parameter.
    """

    def __init__(self, tf_dict: List[Dict[str, Transform]] | Transform) -> None:
        """Creates a new ParamTransform object.

        Args:
            tf_dict: A PyTree of transforms for each parameter.
        """

        self.tf_dict = tf_dict

    def forward(
        self, params: List[Dict[str, ArrayLike]] | ArrayLike
    ) -> Dict[str, Array]:
        """Pushes unconstrained parameters through a tf such that they fit the interval.

        Args:
            params: A list of dictionaries (or any PyTree) with unconstrained parameters.

        Returns:
            A list of dictionaries (or any PyTree) with transformed parameters.

        """

        return jax.tree_util.tree_map(lambda x, tf: tf.forward(x), params, self.tf_dict)

    def inverse(
        self, params: List[Dict[str, ArrayLike]] | ArrayLike
    ) -> Dict[str, Array]:
        """Takes parameters from within the interval and makes them unconstrained.

        Args:
            params: A list of dictionaries (or any PyTree) with transformed parameters.

        Returns:
            A list of dictionaries (or any PyTree) with unconstrained parameters.
        """

        return jax.tree_util.tree_map(lambda x, tf: tf.inverse(x), params, self.tf_dict)
