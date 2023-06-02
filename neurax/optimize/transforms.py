from typing import Dict, List
import jax.numpy as jnp


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid."""
    return 1 / (1 + jnp.exp(-x))


def expit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse sigmoid (expit)"""
    return -jnp.log(1 / x - 1)


class MembraneParamTransform:
    """Membrane parameter transformation utility."""

    def __init__(self, lowers: Dict[str, float], uppers: Dict[str, float]) -> None:
        """Initialize."""
        self.lowers = lowers
        self.uppers = uppers

    def forward(self, params: List[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
        """Pushes unconstrained parameters through a tf such that they fit the interval."""
        tf_params = []
        for param in params:
            key = list(param.keys())[0]
            tf = (
                sigmoid(param[key]) * (self.uppers[key] - self.lowers[key])
                + self.lowers[key]
            )
            tf_params.append({key: tf})
        return tf_params

    def inverse(self, params: jnp.ndarray) -> jnp.ndarray:
        """Takes parameters from within the interval and makes them unconstrained."""
        tf_params = []
        for param in params:
            key = list(param.keys())[0]
            tf = expit(
                (param[key] - self.lowers[key]) / (self.uppers[key] - self.lowers[key])
            )
            tf_params.append({key: tf})
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
