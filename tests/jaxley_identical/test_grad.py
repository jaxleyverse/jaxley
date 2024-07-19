# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

from math import pi

import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import fully_connect
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

    area = 2 * pi * 10.0 * 1.0
    point_process_to_dist_factor = 100_000.0 / area

    net.set("IonotropicSynapse_gS", 0.44 / point_process_to_dist_factor)
    net.set("TestSynapse_gC", 0.62 / point_process_to_dist_factor)
    net.IonotropicSynapse([0, 2, 4]).set(
        "IonotropicSynapse_gS", 0.32 / point_process_to_dist_factor
    )
    net.TestSynapse([0, 3, 5]).set(
        "TestSynapse_gC", 0.24 / point_process_to_dist_factor
    )

    current = jx.step_current(0.5, 0.5, 0.1, 0.025, 10.0)
    for i in range(3):
        net.cell(i).branch(0).loc(0.0).stimulate(current)

    net.cell(6).branch(0).loc(0.0).record()

    def simulate(params):
        return jnp.sum(jx.integrate(net, params=params, tridiag_solver="thomas")[0, ::40])

    net.make_trainable("HH_gNa")
    net.cell([0, 1, 4]).make_trainable("HH_gK")
    net.cell("all").make_trainable("HH_gLeak")

    net.IonotropicSynapse.make_trainable("IonotropicSynapse_gS")
    net.TestSynapse([0, 2]).make_trainable("TestSynapse_gC")

    params = net.get_parameters()
    grad_fn = value_and_grad(simulate)
    v, g = grad_fn(params)

    value_080424 = jnp.asarray(-595.26727325)
    max_error = np.max(np.abs(v - value_080424))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
    grad_080424 = [
        {"HH_gNa": jnp.asarray([-1535.22449953])},
        {"HH_gK": jnp.asarray([35.48478889, 14.32112752, 121.63791342])},
        {
            "HH_gLeak": jnp.asarray(
                [
                    -1455.10679072,
                    -653.562989,
                    -295.76687787,
                    -379.81105237,
                    -10324.49675834,
                    -1344.91808099,
                    -277680.15914084,
                ]
            )
        },
        {
            "IonotropicSynapse_gS": jnp.asarray([-91.88927175883443])
            * point_process_to_dist_factor
        },
        {
            "TestSynapse_gC": jnp.asarray([-0.05350404, -0.11460676])
            * point_process_to_dist_factor
        },
    ]

    for true_g, new_g in zip(grad_080424, g):
        for key in true_g:
            max_error = np.max(np.abs(true_g[key] - new_g[key]))
            tolerance = 1e-3  # Leak cond has a huge gradient...
            assert (
                max_error <= tolerance
            ), f"Error for {key} is {max_error} > {tolerance}"
