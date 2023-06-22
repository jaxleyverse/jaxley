from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad

import neurax as nx
from neurax.channels import HHChannel


def _run_long_branch(time_vec):
    nseg_per_branch = 8

    comp = nx.Compartment()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)])
    branch.insert(HHChannel())

    branch.comp("all").make_trainable("radius", 1.0)
    params = branch.get_parameters()

    stims = [
        nx.Stimulus(0, 0, 0.0, nx.step_current(0.5, 5.0, 0.1, time_vec)),
    ]
    recs = [nx.Recording(0, 0, 0.0)]

    def loss(params):
        s = nx.integrate(branch, stims, recs, params=params)
        return s[0, -1]

    jitted_loss_grad = jit(value_and_grad(loss))

    l, g = jitted_loss_grad(params)
    return l, g


def _run_short_branches(time_vec):
    nseg_per_branch = 4
    parents = jnp.asarray([-1, 0])

    comp = nx.Compartment()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)])
    cell = nx.Cell([branch for _ in range(2)], parents=parents)
    cell.insert(HHChannel())

    cell.branch("all").comp("all").make_trainable("radius", 1.0)
    params = cell.get_parameters()

    stims = [
        nx.Stimulus(0, 0, 0.0, nx.step_current(0.5, 5.0, 0.1, time_vec)),
    ]
    recs = [nx.Recording(0, 0, 0.0)]

    def loss(params):
        s = nx.integrate(cell, stims, recs, params=params)
        return s[0, -1]

    jitted_loss_grad = jit(value_and_grad(loss))

    l, g = jitted_loss_grad(params)
    return l, g


def test_equivalence():
    """Test whether a single long branch matches a cell of two shorter branches."""
    dt = 0.025
    t_max = 5.0  # ms
    time_vec = jnp.arange(0.0, t_max + dt, dt)
    l1, g1 = _run_long_branch(time_vec)
    l2, g2 = _run_short_branches(time_vec)

    assert np.allclose(l1, l2), "Losses do not match."

    for g_1, g_2 in zip(g1, g2):
        for key in g_1:
            rearranged = np.zeros_like(g_1[key])
            rearranged[:4] = g_1[key][4:]
            rearranged[4:] = g_1[key][:4]
            assert np.allclose(rearranged, g_2[key]), "Gradients do not match."
