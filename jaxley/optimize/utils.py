# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import tree_util


def l2_norm(x: "PyTree") -> jnp.array:
    """Return the L2-norm of a pytree. Taken from GH jax/issues/3124."""
    leaves, _ = tree_util.tree_flatten(x)
    return jnp.sqrt(sum([jnp.sum(leaf**2) for leaf in leaves]))


class Uniform:
    """Uniform distribution with sample and log_prob methods."""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def sample(self, key: jnp.ndarray, shape: Tuple[int, ...] = (1,)) -> jnp.ndarray:
        """Samples from the uniform distribution.

        Args:
            key: A JAX random key.
            shape: Sample shape.

        Returns:
            Samples from the uniform distribution.
        """
        return jax.random.uniform(
            key, shape=shape, minval=self.lower, maxval=self.upper
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the log probability of the uniform distribution.

        Args:
            x: The input to compute the log probability for.

        Returns:
            The log probability of the uniform distribution.
        """
        in_bounds = (x >= self.lower) & (x <= self.upper)
        log_p = jnp.where(in_bounds, -jnp.log(self.upper - self.lower), -jnp.inf)
        if x.ndim > 1:
            return jnp.sum(log_p, axis=tuple(range(1, x.ndim)))

        return log_p
