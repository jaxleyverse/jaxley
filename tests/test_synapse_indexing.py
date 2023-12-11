import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import jaxley as jx


def test_set_params_and_querying_params():
    """Test whether the correct parameters are set."""
    pass


def test_shuffling_order_of_set_params():
    """Test whether the result is the same if the order of `set_params` is changed."""
    pass