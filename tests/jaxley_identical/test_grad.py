# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from math import pi

import jax.numpy as jnp
import numpy as np
import pytest
from jax import value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse
from tests import builders as build


@pytest.mark.slow
def test_network_grad(golden):
    net = build.net(ncells=7, nbranches=5, ncomp=4)
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

    # Gradients range from ~10 to ~3e5, so they are compared with a relative tolerance.
    # The check fails if a trainable parameter is added or missing.
    grads = {f"grad_{key}": np.asarray(val) for d in g for key, val in d.items()}
    golden.check(
        {"value": np.asarray(v), **grads},
        rtol=1e-8,
        tolerances={"value": dict(atol=1e-8)},
    )
