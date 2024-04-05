import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jax import jit

from jaxley.optimize.transforms import ParamTransform


def test_inverse():
    # test forward(inverse(x))=x
    lowers = {"param_array_1": 2, "param_array_2": None, "param_array_3": -2}
    uppers = {"param_array_1": -2, "param_array_2": 2, "param_array_3": None}

    params = [
        {"param_array_1": jnp.asarray(np.linspace(-1, 1, 4))},
        {"param_array_2": jnp.asarray(np.linspace(-4, 1, 4))},
        {"param_array_3": jnp.asarray(np.linspace(-1, 4, 4))},
    ]

    tf = ParamTransform(lowers, uppers)

    assert np.allclose(
        tf.forward(tf.inverse(params))[0]["param_array_1"], params[0]["param_array_1"]
    )
    assert np.allclose(
        tf.forward(tf.inverse(params))[1]["param_array_2"], params[1]["param_array_2"]
    )
    assert np.allclose(
        tf.forward(tf.inverse(params))[2]["param_array_3"], params[2]["param_array_3"]
    )


def test_bounds():
    # test if forward maps parameters into bounds
    lowers = {"param_array_1": -2, "param_array_2": None, "param_array_3": -2}
    uppers = {"param_array_1": 2, "param_array_2": 2, "param_array_3": None}

    params = [
        {"param_array_1": jnp.asarray(np.linspace(-10, 10, 4))},
        {"param_array_2": jnp.asarray(np.linspace(-10, 10, 4))},
        {"param_array_3": jnp.asarray(np.linspace(-10, 10, 4))},
    ]

    tf = ParamTransform(lowers, uppers)

    assert all(tf.forward(params)[0]["param_array_1"] > lowers["param_array_1"])
    assert all(tf.forward(params)[0]["param_array_1"] < uppers["param_array_1"])
    assert any(
        tf.forward(params)[1]["param_array_2"] < lowers["param_array_1"]
    )  # lower not constrained
    assert all(tf.forward(params)[1]["param_array_2"] < uppers["param_array_2"])
    assert all(tf.forward(params)[2]["param_array_3"] > lowers["param_array_3"])
    assert any(
        tf.forward(params)[2]["param_array_3"] > uppers["param_array_1"]
    )  # upper not constrained


def test_jit():
    # test jit-compilation:
    lowers = {"param_array_1": 2}
    uppers = {"param_array_1": -2}

    params = [{"param_array_1": jnp.asarray(np.linspace(-1, 1, 4))}]

    tf = ParamTransform(lowers, uppers)

    @jit
    def test_jit(params):
        return tf.inverse(params)

    _ = test_jit(params)
