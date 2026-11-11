# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


from math import pi

import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse
from tests import builders as build


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_compartment(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    comp = build.comp()

    np.random.seed(1)
    comp.set("length", 5 * np.random.rand(1))
    comp.set("radius", np.random.rand(1))

    comp.insert(HH())
    comp.record()
    comp.stimulate(current)

    voltages = jx.integrate(comp, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_branch(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    branch = build.branch(ncomp=2)

    np.random.seed(1)
    branch.set("length", np.flip(5 * np.random.rand(2)))
    branch.set("radius", np.flip(np.random.rand(2)))

    branch.insert(HH())
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    voltages = jx.integrate(branch, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_cell(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    num_branches = 3
    cell = build.cell(nbranches=num_branches, ncomp=2)

    np.random.seed(1)
    rands1 = 5 * np.random.rand(2 * num_branches)
    rands2 = np.random.rand(2 * num_branches)
    for b in range(num_branches):
        cell.branch(b).set("length", np.flip(rands1[2 * b : 2 * b + 2]))
        cell.branch(b).set("radius", np.flip(rands2[2 * b : 2 * b + 2]))

    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_net(voltage_solver, golden):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    num_branches = 3
    net = build.net(ncells=2, nbranches=num_branches, ncomp=2)

    np.random.seed(1)
    rands1 = 5 * np.random.rand(2 * num_branches)
    rands2 = np.random.rand(2 * num_branches)
    for b in range(num_branches):
        net.cell(0).branch(b).set("length", np.flip(rands1[2 * b : 2 * b + 2]))
        net.cell(0).branch(b).set("radius", np.flip(rands2[2 * b : 2 * b + 2]))

    np.random.seed(2)
    rands1 = 5 * np.random.rand(2 * num_branches)
    rands2 = np.random.rand(2 * num_branches)
    for b in range(num_branches):
        net.cell(1).branch(b).set("length", np.flip(rands1[2 * b : 2 * b + 2]))
        net.cell(1).branch(b).set("radius", np.flip(rands2[2 * b : 2 * b + 2]))

    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    net.insert(HH())

    # first cell, 0-eth branch, 0-st compartment because loc=0.0
    radius_post = net[1, 0, 0].nodes["radius"].item()
    length_post = net[1, 0, 0].nodes["length"].item()
    area = 2 * pi * length_post * radius_post
    point_process_to_dist_factor = 100_000.0 / area
    net.set("IonotropicSynapse_gS", 0.5 / point_process_to_dist_factor)

    for cell_ind in range(2):
        net.cell(cell_ind).branch(1).loc(0.0).record()

    for stim_ind in range(2):
        net.cell(stim_ind).branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(net, delta_t=dt, voltage_solver=voltage_solver)

    golden.check(
        {"voltages": np.asarray(voltages[:, ::20])},
        atol=1e-8,
    )
