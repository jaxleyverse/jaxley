# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, value_and_grad

import jaxley as jx
from jaxley.channels import HH
from tests import builders as build


def _run_long_branch(dt, t_max, current, branch):
    branch.insert(HH())

    branch.loc("all").make_trainable("radius", 1.0)
    params = branch.get_parameters()

    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    def loss(params):
        s = jx.integrate(branch, params=params)
        return s[0, -1]

    jitted_loss_grad = jit(value_and_grad(loss))

    l, g = jitted_loss_grad(params)
    return l, g


def _run_short_branches(dt, t_max, current, cell):
    cell.insert(HH())

    cell.branch("all").loc("all").make_trainable("radius", 1.0)
    params = cell.get_parameters()

    cell.branch(0).loc(0.0).record()
    cell.branch(0).loc(0.0).stimulate(current)

    def loss(params):
        s = jx.integrate(cell, params=params)
        return s[0, -1]

    jitted_loss_grad = jit(value_and_grad(loss))

    l, g = jitted_loss_grad(params)
    return l, g


@pytest.mark.slow
def test_equivalence():
    """Test whether a single long branch matches a cell of two shorter branches."""
    dt = 0.025
    t_max = 5.0  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    l1, g1 = _run_long_branch(dt, t_max, current, build.branch(ncomp=8))
    l2, g2 = _run_short_branches(dt, t_max, current, build.cell(nbranches=2, ncomp=4))

    assert np.allclose(l1, l2), "Losses do not match."

    for g_1, g_2 in zip(g1, g2):
        for key in g_1:
            assert np.allclose(g_1[key], g_2[key]), "Gradients do not match."
