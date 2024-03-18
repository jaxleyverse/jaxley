import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jax import jit

import jaxley as jx


def test_direct_distance():
    nseg = 4
    length = 15.0

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell.branch("all").comp("all").set("length", length)
    cell.compute_xyz()
    dist = cell.branch(0).comp(0.0).distance(cell.branch(0).comp(1.0))
    assert dist == (nseg - 1) * length

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 1])
    cell.branch("all").comp("all").set("length", length)
    cell.compute_xyz()
    dist = cell.branch(0).comp(0.0).distance(cell.branch(2).comp(1.0))
    assert dist == (3 * nseg - 1) * length

    move_x = 220.0
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 1])
    cell.branch("all").comp("all").set("length", length)
    net = jx.Network([cell for _ in range(2)])
    net.compute_xyz()
    net.cell(1).move(move_x, 0, 0)
    dist = net.cell(1).branch(0).comp(0.0).distance(net.cell(0).branch(0).comp(0.0))
    assert dist == move_x

    dist = net.cell(0).branch(0).comp(0.0).distance(net.cell(0).branch(0).comp(0.0))
    assert dist == 0.0

    dist = net.cell(1).branch(2).comp(0.3).distance(net.cell(1).branch(2).comp(0.3))
    assert dist == 0.0

    dist = net.cell(1).branch(0).comp(0.0).distance(net.cell(1).branch(2).comp(1.0))
    assert dist == (3 * nseg - 1) * length
