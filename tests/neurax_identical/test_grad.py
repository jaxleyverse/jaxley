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
from jaxley.synapses import GlutamateSynapse, TestSynapse


def test_network_grad():
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net = jx.Network([cell for _ in range(7)])
    net.insert(HH())

    _ = np.random.seed(0)
    pre = net.cell([0, 1, 2])
    post = net.cell([3, 4, 5])
    pre.connect(post, GlutamateSynapse(), p=1)
    pre.connect(post, TestSynapse(), p=1)

    pre = net.cell([3, 4, 5])
    post = net.cell(6)
    pre.connect(post, GlutamateSynapse(), p=1)
    pre.connect(post, TestSynapse(), p=1)

    net.set("gS", 0.44)
    net.set("gC", 0.62)
    net.GlutamateSynapse([0, 2, 4]).set("gS", 0.32)
    net.TestSynapse([0, 3, 5]).set("gC", 0.24)

    current = jx.step_current(0.5, 0.5, 0.1, 0.025, 10.0)
    for i in range(3):
        net.cell(i).branch(0).comp(0.0).stimulate(current)

    net.cell(6).branch(0).comp(0.0).record()

    def simulate(params):
        return jnp.sum(jx.integrate(net, params=params)[0, ::40])

    net.make_trainable("HH_gNa")
    net.cell([0, 1, 4]).make_trainable("HH_gK")
    net.cell("all").make_trainable("HH_gLeak")

    net.GlutamateSynapse.make_trainable("gS")
    net.TestSynapse([0, 2]).make_trainable("gC")

    params = net.get_parameters()
    grad_fn = value_and_grad(simulate)
    v, g = grad_fn(params)

    value_170224 = jnp.asarray(-610.59244206)
    max_error = np.max(np.abs(v - value_170224))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
    grad_040224 = [
        {"HH_gNa": jnp.asarray([[-471.60465337]])},
        {"HH_gK": jnp.asarray([[1.55258077], [0.16785627], [8.87050663]])},
        {
            "HH_gLeak": jnp.asarray(
                [
                    [-6.59547510e01],
                    [-7.20277308e00],
                    [-8.01210580e01],
                    [-5.23790208e02],
                    [-8.77460674e02],
                    [-1.62911998e02],
                    [-1.07148436e05],
                ]
            )
        },
        {"gS": jnp.asarray([[-43.87992857]])},
        {"gC": jnp.asarray([[-0.03726582], [-0.01559716]])},
    ]

    for true_g, new_g in zip(grad_040224, g):
        for key in true_g:
            max_error = np.max(np.abs(true_g[key] - new_g[key]))
            tolerance = 1e-3  # Leak cond has a huge gradient...
            assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
