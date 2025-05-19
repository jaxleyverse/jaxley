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
import pytest
from jax import value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse


@pytest.mark.slow
def test_network_grad(SimpleNet):
    net = SimpleNet(7, 5, 4)
    net.insert(HH())

    _ = np.random.seed(0)
    pre = net.cell([0, 1, 2])
    post = net.cell([3, 4, 5])
    fully_connect(pre, post, IonotropicSynapse(), random_post_comp=True)
    fully_connect(pre, post, TestSynapse(), random_post_comp=True)

    pre = net.cell([3, 4, 5])
    post = net.cell(6)
    fully_connect(pre, post, IonotropicSynapse(), random_post_comp=True)
    fully_connect(pre, post, TestSynapse(), random_post_comp=True)

    area = 2 * pi * 10.0 * 1.0
    point_process_to_dist_factor = 100_000.0 / area

    net.set("IonotropicSynapse_gS", 0.44 / point_process_to_dist_factor)
    net.set("TestSynapse_gC", 0.62 / point_process_to_dist_factor)
    net.IonotropicSynapse.edge([0, 2, 4]).set(
        "IonotropicSynapse_gS", 0.32 / point_process_to_dist_factor
    )
    net.TestSynapse.edge([0, 3, 5]).set(
        "TestSynapse_gC", 0.24 / point_process_to_dist_factor
    )

    current = jx.step_current(
        i_delay=0.5, i_dur=0.5, i_amp=0.1, delta_t=0.025, t_max=10.0
    )
    for i in range(3):
        net.cell(i).branch(0).loc(0.0).stimulate(current)

    net.cell(6).branch(0).loc(0.0).record()

    def simulate(params):
        return jnp.sum(
            jx.integrate(net, params=params, voltage_solver="jaxley.dhs.cpu")[0, ::40]
        )

    net.make_trainable("HH_gNa")
    net.cell([0, 1, 4]).make_trainable("HH_gK")
    net.cell("all").make_trainable("HH_gLeak")

    net.IonotropicSynapse.make_trainable("IonotropicSynapse_gS")
    net.TestSynapse.edge([0, 2]).make_trainable("TestSynapse_gC")

    params = net.get_parameters()
    grad_fn = value_and_grad(simulate)
    v, g = grad_fn(params)

    value_300724 = jnp.asarray(-592.26920963)
    max_error = np.max(np.abs(v - value_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
    grad_300724 = [
        {"HH_gNa": jnp.asarray([-1420.82493379])},
        {"HH_gK": jnp.asarray([36.76796492, 14.50837022, 127.31361882])},
        {
            "HH_gLeak": jnp.asarray(
                [
                    -1511.83061117,
                    -657.06543355,
                    -276.59064858,
                    -348.48049676,
                    -10613.6754718,
                    -1240.62600873,
                    -268475.23225046,
                ]
            )
        },
        {"IonotropicSynapse_gS": jnp.asarray([-149134.15237682])},
        {"TestSynapse_gC": jnp.asarray([-75.34438431, -163.50398586])},
    ]

    for true_g, new_g in zip(grad_300724, g):
        for key in true_g:
            max_error = np.max(np.abs(true_g[key] - new_g[key]))
            tolerance = 1e-3  # Leak cond has a huge gradient...
            assert (
                max_error <= tolerance
            ), f"Error for {key} is {max_error} > {tolerance}"
