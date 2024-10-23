# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

import jaxley.optimize.transforms as jt
from jaxley.optimize.transforms import ParamTransform


def test_joint_inverse():
    # test forward(inverse(x))=x
    tf_dict = {
        "param_array_1": jt.SigmoidTransform(-2, 2),
        "param_array_2": jt.SoftplusTransform(2),
        "param_array_3": jt.NegSoftplusTransform(-2),
    }

    params = [
        {"param_array_1": jnp.asarray(np.linspace(-1, 1, 4))},
        {"param_array_2": jnp.asarray(np.linspace(-4, 1, 4))},
        {"param_array_3": jnp.asarray(np.linspace(-1, 4, 4))},
    ]

    tf = ParamTransform(tf_dict)
    forward = tf.forward(params)
    inverse = tf.inverse(forward)

    assert np.allclose(
        inverse[0]["param_array_1"], params[0]["param_array_1"]
    ), "SigmoidTransform forward, inverse failed."
    assert np.allclose(
        inverse[1]["param_array_2"], params[1]["param_array_2"]
    ), "SoftplusTransform forward, inverse failed."
    assert np.allclose(
        inverse[2]["param_array_3"], params[2]["param_array_3"]
    ), "NegSoftplusTransform forward, inverse failed."


def test_bounds():
    # test if forward maps parameters into bounds
    lowers = {"param_array_1": -2, "param_array_2": None, "param_array_3": -2}
    uppers = {"param_array_1": 2, "param_array_2": 2, "param_array_3": None}

    tf_dict = {
        "param_array_1": jt.SigmoidTransform(-2, 2),
        "param_array_2": jt.NegSoftplusTransform(2),
        "param_array_3": jt.SoftplusTransform(-2),
    }

    params = [
        {"param_array_1": jnp.asarray(np.linspace(-10, 10, 4))},
        {"param_array_2": jnp.asarray(np.linspace(-10, 10, 4))},
        {"param_array_3": jnp.asarray(np.linspace(-10, 10, 4))},
    ]

    tf = ParamTransform(tf_dict)
    forward = tf.forward(params)

    assert all(
        forward[0]["param_array_1"] > lowers["param_array_1"]
    ), "SigmoidTransform failed to match lower bound."
    assert all(
        forward[0]["param_array_1"] < uppers["param_array_1"]
    ), "SigmoidTransform failed to match upper bound."
    assert all(
        forward[1]["param_array_2"] < uppers["param_array_2"]
    ), "SoftplusTransform failed to match upper bound."
    assert all(
        forward[2]["param_array_3"] > lowers["param_array_3"]
    ), "NegSoftplusTransform failed to match lower bound."
    assert any(
        forward[2]["param_array_3"] > uppers["param_array_1"]
    )  # upper not constrained


@pytest.mark.parametrize(
    "transform",
    [
        jt.SigmoidTransform(-2, 2),
        jt.SoftplusTransform(2),
        jt.NegSoftplusTransform(2),
        jt.AffineTransform(1.0, 1.0),
        jt.CustomTransform(lambda x: x, lambda x: x),
        jt.ChainTransform([jt.SigmoidTransform(-2, 2), jt.SoftplusTransform(2)]),
    ],
)
def test_jit(transform):
    # test jit-compilation:
    tf_dict = {"param_array_1": transform}

    params = [{"param_array_1": jnp.asarray(np.linspace(-1, 1, 4))}]

    tf = ParamTransform(tf_dict)

    @jit
    def test_jit(params):
        return tf.inverse(params)

    _ = test_jit(params)


@pytest.mark.parametrize(
    "transform",
    [
        jt.SigmoidTransform(-2, 2),
        jt.SoftplusTransform(2),
        jt.NegSoftplusTransform(2),
        jt.AffineTransform(1.0, 1.0),
        jt.CustomTransform(lambda x: x, lambda x: x),
        jt.ChainTransform([jt.SigmoidTransform(-2, 2), jt.SoftplusTransform(2)]),
    ],
)
def test_correct(transform):
    # test jit-compilation:
    tf_dict = {"param_array_1": transform}

    params = [{"param_array_1": jnp.asarray(np.linspace(-1, 1, 4))}]

    tf = ParamTransform(tf_dict)

    forward = tf.forward(params)
    inverse = tf.inverse(forward)

    assert np.allclose(
        inverse[0]["param_array_1"], params[0]["param_array_1"]
    ), f"{transform} forward, inverse failed."
