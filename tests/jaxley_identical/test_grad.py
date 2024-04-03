from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.connection import fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse


def test_network_grad():
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net = jx.Network([cell for _ in range(7)])
    net.insert(HH())

    _ = np.random.seed(0)
    pre = net.cell([0, 1, 2])
    post = net.cell([3, 4, 5])
    fully_connect(pre, post, IonotropicSynapse())
    fully_connect(pre, post, TestSynapse())

    pre = net.cell([3, 4, 5])
    post = net.cell(6)
    fully_connect(pre, post, IonotropicSynapse())
    fully_connect(pre, post, TestSynapse())

    net.set("gS", 0.44)
    net.set("gC", 0.62)
    net.IonotropicSynapse([0, 2, 4]).set("gS", 0.32)
    net.TestSynapse([0, 3, 5]).set("gC", 0.24)

    current = jx.step_current(0.5, 0.5, 0.1, 0.025, 10.0)
    for i in range(3):
        net.cell(i).branch(0).loc(0.0).stimulate(current)

    net.cell(6).branch(0).loc(0.0).record()

    def simulate(params):
        return jnp.sum(jx.integrate(net, params=params)[0, ::40])

    net.make_trainable("HH_gNa")
    net.cell([0, 1, 4]).make_trainable("HH_gK")
    net.cell("all").make_trainable("HH_gLeak")

    net.IonotropicSynapse.make_trainable("gS")
    net.TestSynapse([0, 2]).make_trainable("gC")

    params = net.get_parameters()
    grad_fn = value_and_grad(simulate)
    v, g = grad_fn(params)

    value_230224 = jnp.asarray(-580.79271734)
    max_error = np.max(np.abs(v - value_230224))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
    grad_230224 = [
        {"HH_gNa": jnp.asarray([-1330.69128594])},
        {"HH_gK": jnp.asarray([2.4996021, 13.61190008, 61.16684584])},
        {
            "HH_gLeak": jnp.asarray(
                [
                    -121.305113,
                    -606.555168,
                    -183.932419,
                    -465.945968,
                    -5668.16363,
                    -248.320435,
                    -250947.475,
                ]
            )
        },
        {"gS": jnp.asarray([-85.83902596])},
        {"gC": jnp.asarray([-0.00831085, -0.00502889])},
    ]

    for true_g, new_g in zip(grad_230224, g):
        for key in true_g:
            max_error = np.max(np.abs(true_g[key] - new_g[key]))
            tolerance = 1e-3  # Leak cond has a huge gradient...
            assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
