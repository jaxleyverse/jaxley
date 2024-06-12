import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH, CaL, CaT, Channel, K, Km, Leak, Na


def test_clamp_pointneuron():
    comp = jx.Compartment()
    comp.insert(HH())
    comp.record()
    comp.clamp("v", -50.0 * jnp.ones((1000,)))

    v = jx.integrate(comp, t_max=1.0)
    assert np.all(v[:, 1:] == -50.0)


def test_clamp_multicompartment():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    branch.insert(HH())
    branch.record()
    branch.comp(0).clamp("v", -50.0 * jnp.ones((1000,)))

    v = jx.integrate(branch, t_max=1.0)

    # The clamped compartment should be fixed.
    assert np.all(v[0, 1:] == -50.0)

    # For other compartments, the voltage should have non-zero std.
    assert np.all(np.std(v[1:, 1:], axis=1) > 0.1)


def test_clamp_and_stimulate_api():
    """Ensure proper behaviour when `.clamp()` and `.stimulate()` are combined."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell1 = jx.Cell(branch, [-1])
    cell2 = jx.Cell(branch, [-1])
    net = jx.Network([cell1, cell2])

    net.insert(HH())
    net[0, 0, 0].record()
    net[1, 0, 0].record()

    net[0, 0, 0].clamp("v", 0.1 * jnp.ones((1000,)))
    net[1, 0, 0].stimulate(0.1 * jnp.ones((1000,)))

    vs1 = jx.integrate(net, t_max=1.0)

    cell1.insert(HH())
    cell1[0, 0].record()
    cell1[0, 0].clamp("v", 0.1 * jnp.ones((1000,)))
    vs21 = jx.integrate(cell1, t_max=1.0)

    cell2.insert(HH())
    cell2[0, 0].record()
    cell2[0, 0].stimulate(0.1 * jnp.ones((1000,)))
    vs22 = jx.integrate(cell2, t_max=1.0)

    vs2 = jnp.concatenate([vs21, vs22])
    assert np.max(np.abs(vs1 - vs2)) < 1e-8
