from typing import List
import jax.numpy as jnp
import distrax


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid."""
    return 1 / (1 + jnp.exp(-x))


def expit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse sigmoid (expit)"""
    return -jnp.log(1 / x - 1)


class MembraneParamTransform:
    """Membrane parameter transformation utility.
    It makes the assumption that a particular conductance has the same lower and upper
    bound in every compartment and in every cell. With this assumption, only the lower
    and upper bound for every type of conductance have to be provided to this class.
    """

    def __init__(self, lowers: List[List[float]], uppers: List[List[float]]) -> None:
        """Build the transformation."""
        self.lowers = [jnp.stack(vec) for vec in lowers]
        self.uppers = [jnp.stack(vec) for vec in uppers]

    def forward(self, params: jnp.ndarray) -> jnp.ndarray:
        """Pushes unconstrained parameters through a tf such that they fit the interval."""
        tf_params = []
        for channel_params, lower, upper in zip(params, self.lowers, self.uppers):
            transformed = sigmoid(channel_params.T) * (upper - lower) + lower
            tf_params.append(transformed.T)
        return tf_params

    def inverse(self, params: jnp.ndarray) -> jnp.ndarray:
        """Takes parameters from within the interval and makes them unconstrained."""
        tf_params = []
        for channel_params, lower, upper in zip(params, self.lowers, self.uppers):
            transformed = expit((channel_params.T - lower) / (upper - lower))
            tf_params.append(transformed.T)
        return tf_params


class SynapseParamTransform:
    """Synapse parameter transformation utility."""

    def __init__(self, lowers: List[List[float]], uppers: List[List[float]]) -> None:
        """Build the transformation."""
        raise NotImplementedError

    def forward(self, params: jnp.ndarray) -> jnp.ndarray:
        """Pushes unconstrained parameters through a tf such that they fit the interval."""
        raise NotImplementedError

    def inverse(self, params: jnp.ndarray) -> jnp.ndarray:
        """Takes parameters from within the interval and makes them unconstrained."""
        raise NotImplementedError
