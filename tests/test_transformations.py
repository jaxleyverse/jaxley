# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""JAX transformations of simulations: `vmap` and checkpointed gradients."""

import jax
import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from tests import builders as build


def _hh_cell():
    cell = build.cell(nbranches=3, ncomp=2)
    cell.insert(HH())
    cell.branch(0).comp(0).stimulate(jx.step_current(0.5, 1.0, 0.1, 0.025, 5.0))
    cell.branch(2).comp(1).record()
    cell.make_trainable("HH_gNa")
    cell.make_trainable("radius")
    return cell


def test_vmap_over_parameters():
    """`vmap` over parameters gives the same voltages as separate simulations."""
    cell = _hh_cell()
    params = cell.get_parameters()
    simulate = jax.jit(lambda p: jx.integrate(cell, params=p))

    scales = jnp.asarray([0.8, 1.0, 1.2])
    batched = jax.tree.map(lambda x: scales[:, None] * x, params)
    v_batched = jax.vmap(simulate)(batched)

    for i, scale in enumerate(scales):
        v = simulate(jax.tree.map(lambda x: scale * x, params))
        np.testing.assert_allclose(v_batched[i], v, atol=1e-10, rtol=0)


def test_checkpointing_does_not_change_gradient():
    """Checkpointing (`checkpoint_lengths`) changes memory use, not the result."""
    cell = _hh_cell()
    params = cell.get_parameters()

    def loss(params, checkpoint_lengths):
        v = jx.integrate(cell, params=params, checkpoint_lengths=checkpoint_lengths)
        return jnp.mean(v)

    # 200 steps; the checkpointed run pads to 25 * 9 = 225 steps and truncates.
    value, grad = jax.value_and_grad(loss)(params, None)
    value_cp, grad_cp = jax.value_and_grad(loss)(params, [25, 9])
    np.testing.assert_allclose(value_cp, value, rtol=1e-12)
    for g, g_cp in zip(grad, grad_cp):
        for key in g:
            np.testing.assert_allclose(g_cp[key], g[key], rtol=1e-10, err_msg=key)
