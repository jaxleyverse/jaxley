# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import jax.numpy as jnp
import numpy as np
import pytest
from jax import value_and_grad

import jaxley as jx
from tests import builders as build
from tests.helpers import wire_seven_cell_net


@pytest.mark.slow
def test_network_grad(golden):
    net = build.net(ncells=7, nbranches=5, ncomp=4)
    wire_seven_cell_net(net)

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
