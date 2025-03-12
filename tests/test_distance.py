# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

import jaxley as jx
from jaxley.channels import Leak


def test_direct_distance(SimpleCell):
    ncomp = 4
    length = 15.0

    cell = SimpleCell(5, ncomp)
    cell.branch("all").loc("all").set("length", length)
    cell.compute_xyz()
    dist = cell.branch(0).loc(0.0).distance(cell.branch(0).loc(1.0))
    assert dist == (ncomp - 1) * length

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0, 1])
    cell.branch("all").loc("all").set("length", length)
    cell.compute_xyz()
    dist = cell.branch(0).loc(0.0).distance(cell.branch(2).loc(1.0))
    assert dist == (3 * ncomp - 1) * length

    move_x = 220.0
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0, 1])
    cell.branch("all").loc("all").set("length", length)
    net = jx.Network([cell for _ in range(2)])
    net.compute_xyz()
    net.cell(1).move(move_x, 0, 0)
    dist = net.cell(1).branch(0).loc(0.0).distance(net.cell(0).branch(0).loc(0.0))
    assert dist == move_x

    dist = net.cell(0).branch(0).loc(0.0).distance(net.cell(0).branch(0).loc(0.0))
    assert dist == 0.0

    dist = net.cell(1).branch(2).loc(0.3).distance(net.cell(1).branch(2).loc(0.3))
    assert dist == 0.0

    dist = net.cell(1).branch(0).loc(0.0).distance(net.cell(1).branch(2).loc(1.0))
    assert dist == (3 * ncomp - 1) * length


def test_distance_within_jit(SimpleCell):
    cell = SimpleCell(2, 3)
    cell.insert(Leak())
    cell.compute_xyz()

    cell[0, 0].stimulate(0.1 * jnp.ones((100,)))
    cell[0, 0].record()

    def simulate(sigmoid_offset, global_scaling):
        pstate = None
        for branch in cell:
            for comp in branch:
                distance = cell[0, 0].distance(comp)
                conductance = global_scaling / (jnp.exp(-(distance + sigmoid_offset)))
                pstate = comp.data_set("Leak_gLeak", conductance * 1e-4, pstate)
        return jx.integrate(cell, param_state=pstate)

    sigmoid_offset = jnp.ones((1,))
    global_scaling = jnp.ones((1,))

    jitted_simulate = jit(simulate)
    v = jitted_simulate(sigmoid_offset, global_scaling)
    assert np.invert(np.any(np.isnan(v)))
