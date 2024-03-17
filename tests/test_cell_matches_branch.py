import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad

import jaxley as jx
from jaxley.channels import HH


def _run_long_branch(dt, t_max):
    nseg_per_branch = 8

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    branch.insert(HH())

    branch.comp("all").make_trainable("radius", 1.0)
    params = branch.get_parameters()

    branch.comp(0.0).record()
    branch.comp(0.0).stimulate(jx.step_current(0.5, 5.0, 0.1, dt, t_max))

    def loss(params):
        s = jx.integrate(branch, params=params)
        return s[0, -1]

    jitted_loss_grad = jit(value_and_grad(loss))

    l, g = jitted_loss_grad(params)
    return l, g


def _run_short_branches(dt, t_max):
    nseg_per_branch = 4
    parents = jnp.asarray([-1, 0])

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    cell = jx.Cell(branch, parents=parents)
    cell.insert(HH())

    cell.branch("all").comp("all").make_trainable("radius", 1.0)
    params = cell.get_parameters()

    cell.branch(0).comp(0.0).record()
    cell.branch(0).comp(0.0).stimulate(jx.step_current(0.5, 5.0, 0.1, dt, t_max))

    def loss(params):
        s = jx.integrate(cell, params=params)
        return s[0, -1]

    jitted_loss_grad = jit(value_and_grad(loss))

    l, g = jitted_loss_grad(params)
    return l, g


def test_equivalence():
    """Test whether a single long branch matches a cell of two shorter branches."""
    dt = 0.025
    t_max = 5.0  # ms
    l1, g1 = _run_long_branch(dt, t_max)
    l2, g2 = _run_short_branches(dt, t_max)

    assert np.allclose(l1, l2), "Losses do not match."

    for g_1, g_2 in zip(g1, g2):
        for key in g_1:
            rearranged = np.zeros_like(g_1[key])
            rearranged[:4] = g_1[key][4:]
            rearranged[4:] = g_1[key][:4]
            assert np.allclose(rearranged, g_2[key]), "Gradients do not match."
