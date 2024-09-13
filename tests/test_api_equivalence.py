# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse


def test_api_equivalence_morphology():
    """Test the API for how one can build morphologies from scratch."""
    nseg_per_branch = 2
    depth = 2
    dt = 0.025

    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)
    num_branches = len(parents)

    comp = jx.Compartment()

    branch1 = jx.Branch([comp for _ in range(nseg_per_branch)])
    cell1 = jx.Cell([branch1 for _ in range(num_branches)], parents=parents)

    branch2 = jx.Branch(comp, nseg=nseg_per_branch)
    cell2 = jx.Cell(branch2, parents=parents)

    cell1.branch(2).loc(0.4).record()
    cell2.branch(2).loc(0.4).record()

    current = jx.step_current(0.5, 1.0, 1.0, dt, 3.0)
    cell1.branch(1).loc(1.0).stimulate(current)
    cell2.branch(1).loc(1.0).stimulate(current)

    voltages1 = jx.integrate(cell1, delta_t=dt)
    voltages2 = jx.integrate(cell2, delta_t=dt)
    assert (
        jnp.max(jnp.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between morphology APIs."


def test_solver_backends_comp():
    """Test whether ways of adding synapses are equivalent."""
    comp = jx.Compartment()

    current = jx.step_current(0.5, 1.0, 0.5, 0.025, 5.0)
    comp.stimulate(current)
    comp.record()

    voltages_jx_thomas = jx.integrate(comp, voltage_solver="jaxley.thomas")
    voltages_jx_stone = jx.integrate(comp, voltage_solver="jaxley.stone")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_thomas - voltages_jx_stone))
    assert max_error < 1e-8, f"{message} thomas/stone. Error={max_error}"


def test_solver_backends_branch():
    """Test whether ways of adding synapses are equivalent."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)

    current = jx.step_current(0.5, 1.0, 0.5, 0.025, 5.0)
    branch.loc(0.0).stimulate(current)
    branch.loc(0.5).record()

    voltages_jx_thomas = jx.integrate(branch, voltage_solver="jaxley.thomas")
    voltages_jx_stone = jx.integrate(branch, voltage_solver="jaxley.stone")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_thomas - voltages_jx_stone))
    assert max_error < 1e-8, f"{message} thomas/stone. Error={max_error}"


def test_solver_backends_cell():
    """Test whether ways of adding synapses are equivalent."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    current = jx.step_current(0.5, 1.0, 0.5, 0.025, 5.0)
    cell.branch(0).loc(0.0).stimulate(current)
    cell.branch(0).loc(0.5).record()
    cell.branch(4).loc(0.5).record()

    voltages_jx_thomas = jx.integrate(cell, voltage_solver="jaxley.thomas")
    voltages_jx_stone = jx.integrate(cell, voltage_solver="jaxley.stone")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_thomas - voltages_jx_stone))
    assert max_error < 1e-8, f"{message} thomas/stone. Error={max_error}"


def test_solver_backends_net():
    """Test whether ways of adding synapses are equivalent."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell1 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell2 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net = jx.Network([cell1, cell2])
    connect(
        net.cell(0).branch(0).loc(1.0),
        net.cell(1).branch(4).loc(1.0),
        IonotropicSynapse(),
    )
    connect(
        net.cell(1).branch(1).loc(0.8),
        net.cell(0).branch(4).loc(0.1),
        IonotropicSynapse(),
    )

    current = jx.step_current(0.5, 1.0, 0.5, 0.025, 5.0)
    net.cell(0).branch(0).loc(0.0).stimulate(current)
    net.cell(0).branch(0).loc(0.5).record()
    net.cell(1).branch(4).loc(0.5).record()

    voltages_jx_thomas = jx.integrate(net, voltage_solver="jaxley.thomas")
    voltages_jx_stone = jx.integrate(net, voltage_solver="jaxley.stone")

    message = "Voltages do not match between"
    max_error = np.max(np.abs(voltages_jx_thomas - voltages_jx_stone))
    assert max_error < 1e-8, f"{message} thomas/stone. Error={max_error}"


def test_api_equivalence_synapses():
    """Test whether ways of adding synapses are equivalent."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell1 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell2 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net1 = jx.Network([cell1, cell2])
    connect(
        net1.cell(0).branch(0).loc(1.0),
        net1.cell(1).branch(4).loc(1.0),
        IonotropicSynapse(),
    )
    connect(
        net1.cell(1).branch(1).loc(0.8),
        net1.cell(0).branch(4).loc(0.1),
        IonotropicSynapse(),
    )

    net2 = jx.Network([cell1, cell2])
    pre = net2.cell(0).branch(0).loc(1.0)
    post = net2.cell(1).branch(4).loc(1.0)
    connect(pre, post, IonotropicSynapse())

    pre = net2.cell(1).branch(1).loc(0.8)
    post = net2.cell(0).branch(4).loc(0.1)
    connect(pre, post, IonotropicSynapse())

    for net in [net1, net2]:
        current = jx.step_current(0.5, 1.0, 0.5, 0.025, 5.0)
        net.cell(0).branch(0).loc(0.0).stimulate(current)
        net.cell(0).branch(0).loc(0.5).record()
        net.cell(1).branch(4).loc(0.5).record()

    voltages1 = jx.integrate(net1)
    voltages2 = jx.integrate(net2)

    assert (
        np.max(np.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between synapse APIs."


def test_api_equivalence_continued_simulation():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 2)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(HH())
    cell[0, 1].record()

    v1 = jx.integrate(cell, t_max=4.0)
    v21, states = jx.integrate(cell, return_states=True, t_max=2.0)
    v22 = jx.integrate(cell, all_states=states, t_max=2.0)

    v2 = jnp.concatenate([v21, v22[:, 1:]], axis=1)
    assert np.max(np.abs(v1 - v2)) < 1e-8


def test_api_equivalence_network_matches_cell():
    """Test whether a network with w=0 synapses equals the individual cells.

    This runs an unequal number of compartments per branch."""
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.1, dt, t_max)

    comp = jx.Compartment()
    branch1 = jx.Branch(comp, nseg=1)
    branch2 = jx.Branch(comp, nseg=2)
    branch3 = jx.Branch(comp, nseg=3)
    cell1 = jx.Cell([branch1, branch2, branch3], parents=[-1, 0, 0])
    cell2 = jx.Cell([branch1, branch2], parents=[-1, 0])
    cell1.insert(HH())
    cell2.insert(HH())

    net = jx.Network([cell1, cell2])
    pre = net.cell(0).branch(2).comp(2)
    post = net.cell(1).branch(1).comp(1)
    connect(pre, post, IonotropicSynapse())
    net.IonotropicSynapse("all").set("IonotropicSynapse_gS", 0.0)

    net.cell(0).branch(2).comp(2).stimulate(current)
    net.cell(0).branch(0).comp(0).record()

    net.cell(1).branch(1).comp(1).stimulate(current)
    net.cell(1).branch(0).comp(0).record()
    voltages_net = jx.integrate(net, delta_t=dt, voltage_solver="jax.sparse")

    cell1.branch(2).comp(2).stimulate(current)
    cell1.branch(0).comp(0).record()

    cell2.branch(1).comp(1).stimulate(current)
    cell2.branch(0).comp(0).record()
    voltages_cell1 = jx.integrate(cell1, delta_t=dt, voltage_solver="jax.sparse")
    voltages_cell2 = jx.integrate(cell2, delta_t=dt, voltage_solver="jax.sparse")
    voltages_cells = jnp.concatenate([voltages_cell1, voltages_cell2], axis=0)

    max_error = np.max(np.abs(voltages_net - voltages_cells))
    assert max_error < 1e-8, f"Error is {max_error}"
