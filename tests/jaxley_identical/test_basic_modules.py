# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from math import pi

import numpy as np
import pytest
from jaxley_mech.channels.l5pc import CaHVA

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.pumps import CaFaradayConcentrationChange, CaNernstReversal
from jaxley.synapses import IonotropicSynapse
from tests import builders as build
from tests.helpers import wire_seven_cell_net


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_compartment(voltage_solver, golden):
    """Test a compartment."""
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )
    voltages = {}

    # Test compartment.
    comp = build.comp()
    comp.insert(HH())
    comp.record()
    comp.stimulate(current)
    v = jx.integrate(comp, delta_t=dt, voltage_solver=voltage_solver)
    voltages["comp"] = np.asarray(v[:, ::20])

    # Test branch of a single compartment.
    branch = build.branch(ncomp=1)
    branch.insert(HH())
    branch.record()
    branch.stimulate(current)
    v = jx.integrate(branch, delta_t=dt, voltage_solver=voltage_solver)
    voltages["branch"] = np.asarray(v[:, ::20])

    # Test cell of a single compartment.
    cell = build.cell(nbranches=1, ncomp=1)
    cell.insert(HH())
    cell.record()
    cell.stimulate(current)
    v = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)
    voltages["cell"] = np.asarray(v[:, ::20])

    # Test net of a single compartment.
    net = build.net(ncells=1, nbranches=1, ncomp=1)
    net.insert(HH())
    net.record()
    net.stimulate(current)
    v = jx.integrate(net, delta_t=dt, voltage_solver=voltage_solver)
    voltages["net"] = np.asarray(v[:, ::20])

    # All four are compared against the same values.
    golden.check(voltages, atol=1e-8)


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_branch(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    branch = build.branch(ncomp=2)
    branch.insert(HH())
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    voltages = jx.integrate(branch, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )


@pytest.mark.parametrize("solver", ["fwd_euler", "bwd_euler", "exp_euler"])
def test_branch_uneven_radiuses(solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=2.0, delta_t=0.025, t_max=10.0
    )

    branch = build.branch(ncomp=8)
    branch.set("axial_resistivity", 500.0)

    rands1 = np.linspace(20, 300, 8)
    rands2 = np.linspace(1, 5, 8)
    branch.set("length", rands1)
    branch.set("radius", rands2)

    branch.insert(HH())
    branch.loc(1.0).stimulate(current)
    branch.loc(0.0).record()

    if solver == "bwd_euler":
        voltage_solver = "jaxley.stone"
    else:
        # unused: `voltage_solver` is ignored for fwd_euler.
        voltage_solver = "jaxley.dhs"
    voltages = jx.integrate(
        branch, delta_t=dt, solver=solver, voltage_solver=voltage_solver
    )

    golden.check(
        {"voltages": np.asarray(voltages[0, ::50])},
        atol=1e-5,
    )


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_cell(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    cell = build.cell(nbranches=3, ncomp=2)
    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("solver", "voltage_solver"),
    [
        ("exp_euler", "jaxley.dhs.cpu"),
        ("bwd_euler", "jaxley.dhs.cpu"),
        ("bwd_euler", "jaxley.dhs.gpu"),
        ("bwd_euler", "jax.sparse"),
    ],
)
def test_complex_cell(solver, voltage_solver, golden):
    """Test cell with a variety of channels, pumps, diffusion, and comps per branch."""
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    branch1 = build.branch(ncomp=1)
    branch2 = build.branch(ncomp=2)
    branch3 = build.branch(ncomp=3)
    branch4 = build.branch(ncomp=4)
    cell = jx.Cell([branch1, branch2, branch3, branch4], parents=[-1, 0, 0, 1])
    cell.set("axial_resistivity", 10_000.0)
    cell.insert(HH())

    # Calcium mechanisms.
    cell.branch(0).insert(CaHVA())
    cell.branch(0).set("CaHVA_gCaHVA", 0.0001)  # Larger than default to see impact on v
    cell.insert(CaFaradayConcentrationChange())
    cell.insert(CaNernstReversal())
    cell.diffuse("CaCon_i")
    # Large value to see the diffusion after a few milliseconds already.
    cell.set("axial_diffusion_CaCon_i", 180.2)

    cell.branch(1).comp(1).stimulate(current)
    cell.branch(0).comp(0).record("v")
    cell.branch(2).comp(2).record("v")
    cell.branch(3).comp(1).record("v")
    cell.branch(3).comp(3).record("v")
    cell.branch(0).comp(0).record("CaCon_i")
    cell.branch(2).comp(2).record("CaCon_i")
    cell.branch(3).comp(1).record("CaCon_i")
    cell.branch(3).comp(3).record("CaCon_i")

    if voltage_solver == "jaxley.dhs.gpu":
        # On CPU we have to run this manually. On GPU, it gets run automatically with
        # allowed_nodes_per_level=32.
        cell._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=4)

    recordings = jx.integrate(
        cell, delta_t=dt, solver=solver, voltage_solver=voltage_solver
    )

    golden.check(
        {
            "voltages": np.asarray(recordings[:4, ::50]),
            "cacon": np.asarray(recordings[4:, ::50]),
        },
        atol=1e-8,
        tolerances={"cacon": dict(atol=1e-10)},
    )


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_net(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    # net = build.net(ncells=2, nbranches=3, ncomp=2)
    cell1 = build.cell(nbranches=3, ncomp=2)
    cell2 = build.cell(nbranches=3, ncomp=2)
    net = jx.Network([cell1, cell2])

    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    net.insert(HH())

    for cell_ind in range(2):
        net.cell(cell_ind).branch(1).loc(0.0).record()

    for stim_ind in range(2):
        net.cell(stim_ind).branch(1).loc(0.0).stimulate(current)

    area = 2 * pi * 10.0 * 1.0
    point_process_to_dist_factor = 100_000.0 / area
    net.IonotropicSynapse.set(
        "IonotropicSynapse_gS", 0.5 / point_process_to_dist_factor
    )
    voltages = jx.integrate(net, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("solver", "voltage_solver"),
    [
        ("exp_euler", "jaxley.dhs.cpu"),
        ("bwd_euler", "jaxley.dhs.cpu"),
        ("bwd_euler", "jaxley.dhs.gpu"),
        ("bwd_euler", "jax.sparse"),
    ],
)
def test_complex_net(solver, voltage_solver, golden):
    cell = build.cell(nbranches=5, ncomp=4)
    if voltage_solver == "jaxley.dhs.gpu":
        # On CPU we have to run this manually. On GPU, it gets run automatically with
        # allowed_nodes_per_level=32.
        cell._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=4)

    if voltage_solver == "jaxley.dhs.cpu":
        net = jx.Network([cell for _ in range(7)], vectorize_cells=False)
    else:
        net = jx.Network([cell for _ in range(7)], vectorize_cells=True)

    wire_seven_cell_net(net)

    voltages = jx.integrate(net, solver=solver, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::40])},
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "voltage_solver", ["jaxley.dhs.cpu", "jaxley.dhs.gpu", "jax.sparse"]
)
def test_mixed_net(voltage_solver, golden):
    simple_cell = jx.Cell()
    morph_cell = build.cell(nbranches=5, ncomp=4)
    if voltage_solver == "jaxley.dhs.gpu":
        # On CPU we have to run this manually. On GPU, it gets run automatically with
        # allowed_nodes_per_level=32.
        morph_cell._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=4)

    if voltage_solver == "jaxley.dhs.cpu":
        net = jx.Network(
            [simple_cell for _ in range(6)] + [morph_cell], vectorize_cells=False
        )
    else:
        net = jx.Network(
            [simple_cell for _ in range(6)] + [morph_cell], vectorize_cells=True
        )

    wire_seven_cell_net(net, stimulated_cells=6)

    voltages = jx.integrate(net, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::40])},
        atol=1e-8,
    )
