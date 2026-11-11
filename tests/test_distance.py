# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

import jaxley as jx
from jaxley.channels import Leak
from jaxley.morphology import distance_direct, distance_pathwise
from tests import builders as build


def test_direct_distance():
    ncomp = 4
    length = 15.0

    cell = build.cell(nbranches=5, ncomp=ncomp)
    cell.branch("all").loc("all").set("length", length)
    cell.compute_xyz()
    cell.compute_compartment_centers()
    dist = distance_direct(cell.branch(0).loc(0.0), cell.branch(0).loc(1.0))
    assert dist[0] == (ncomp - 1) * length

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0, 1])
    cell.branch("all").loc("all").set("length", length)
    cell.compute_xyz()
    cell.compute_compartment_centers()
    dist = distance_direct(cell.branch(0).loc(0.0), cell.branch(2).loc(1.0))
    assert dist[0] == (3 * ncomp - 1) * length

    move_x = 220.0
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0, 1])
    cell.branch("all").loc("all").set("length", length)
    net = jx.Network([cell for _ in range(2)])
    net.compute_xyz()
    net.cell(1).move(move_x, 0, 0)
    net.compute_compartment_centers()
    dist = distance_direct(
        net.cell(1).branch(0).loc(0.0), net.cell(0).branch(0).loc(0.0)
    )
    assert dist[0] == move_x

    dist = distance_direct(
        net.cell(0).branch(0).loc(0.0), net.cell(0).branch(0).loc(0.0)
    )
    assert dist[0] == 0.0

    dist = distance_direct(
        net.cell(1).branch(2).loc(0.3), net.cell(1).branch(2).loc(0.3)
    )
    assert dist[0] == 0.0

    dist = distance_direct(
        net.cell(1).branch(0).loc(0.0), net.cell(1).branch(2).loc(1.0)
    )
    assert dist[0] == (3 * ncomp - 1) * length


def test_pathwise_distance():
    ncomp = 4
    length = 25.0
    cell = build.cell(nbranches=5, ncomp=ncomp)
    cell.branch("all").loc("all").set("length", length)
    pathdist = distance_pathwise(cell[0, 0], cell)
    cell.nodes["path_dist"] = pathdist

    dist_to_tip = cell[3, 3].nodes["path_dist"].item()
    assert (
        dist_to_tip == (ncomp * 3 - 1) * length
    ), f"{dist_to_tip} != {(ncomp * 3 - 1) * length}"


@pytest.mark.parametrize("kind", ["direct", "pathwise"])
def test_distance_within_jit(kind: str):
    cell = build.cell(nbranches=2, ncomp=3)
    cell.insert(Leak())
    cell.compute_xyz()
    cell.compute_compartment_centers()

    cell[0, 0].stimulate(0.1 * jnp.ones((100,)))
    cell[0, 0].record()

    if kind == "direct":
        dists = distance_direct(cell[0, 0], cell)
    elif kind == "pathwise":
        dists = distance_pathwise(cell[0, 0], cell)
    else:
        raise ValueError

    def conductances(sigmoid_offset, global_scaling):
        return global_scaling / (jnp.exp(-(jnp.asarray(dists) + sigmoid_offset)))

    def build_pstate(sigmoid_offset, global_scaling):
        pstate = None
        counter = 0
        for branch in cell:
            for comp in branch:
                dist = dists[counter]
                conductance = global_scaling / (jnp.exp(-(dist + sigmoid_offset)))
                pstate = comp.data_set("Leak_gLeak", conductance * 1e-4, pstate)
                counter += 1
        return pstate

    def simulate(sigmoid_offset, global_scaling):
        pstate = build_pstate(sigmoid_offset, global_scaling)
        return jx.integrate(cell, param_state=pstate)

    sigmoid_offset = jnp.ones((1,))
    global_scaling = jnp.ones((1,))

    # Every compartment gets the conductance computed from its own distance.
    pstate = build_pstate(sigmoid_offset, global_scaling)
    set_values = jnp.concatenate([p["val"] for p in pstate])
    expected = conductances(sigmoid_offset, global_scaling) * 1e-4
    np.testing.assert_allclose(set_values, expected)

    jitted_simulate = jit(simulate)
    v = jitted_simulate(sigmoid_offset, global_scaling)
    assert np.invert(np.any(np.isnan(v)))
    np.testing.assert_allclose(v, simulate(sigmoid_offset, global_scaling))


def test_distance_swc(golden):
    """Ensures that the distance computation for an SWC file remains unchanged."""
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_ca1_n120_250.swc")

    cell = build.swc_cell(fname, ncomp=1, max_branch_len=2_000.0)
    cell.set("axial_resistivity", 100.0)
    # Reasonable default values for most models.
    frequency = 100.0
    d_lambda = 0.1  # Larger -> more coarse-grained.

    for branch in cell.branches:
        diameter = 2 * branch.nodes["radius"].to_numpy()[0]
        c_m = branch.nodes["capacitance"].to_numpy()[0]
        r_a = branch.nodes["axial_resistivity"].to_numpy()[0]
        l = branch.nodes["length"].to_numpy()[0]

        lambda_f = 1e5 * np.sqrt(diameter / (4 * np.pi * frequency * c_m * r_a))
        ncomp = int((l / (d_lambda * lambda_f) + 0.9) / 2) * 2 + 1
        branch.set_ncomp(ncomp, initialize=False)

    # After the loop, you have to run `cell.initialize()` because we passed
    # `set_ncomp(..., initialize=False)` for speeding up the loop over branches.
    cell.initialize()

    soma = cell.soma.branch(0).comp(0)
    golden.check(
        {
            "pathwise": np.asarray(distance_pathwise(soma, cell))[::10],
            "direct": np.asarray(distance_direct(soma, cell))[::10],
        },
        atol=1e-8,
    )
