# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from jax import config

from jaxley.connect import connect

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
from math import pi

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_compartment(voltage_solver, SimpleComp):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    comp = SimpleComp()

    np.random.seed(1)
    comp.set("length", 5 * np.random.rand(1))
    comp.set("radius", np.random.rand(1))

    comp.insert(HH())
    comp.record()
    comp.stimulate(current)

    voltages = jx.integrate(comp, delta_t=dt, voltage_solver=voltage_solver)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                45.83306656,
                29.72581199,
                -15.28462737,
                -39.8274255,
                -66.59985023,
                -75.55397377,
                -75.74933676,
                -75.48829132,
                -75.15524717,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_branch(voltage_solver, SimpleBranch):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    branch = SimpleBranch(ncomp=2)

    np.random.seed(1)
    branch.set("length", np.flip(5 * np.random.rand(2)))
    branch.set("radius", np.flip(np.random.rand(2)))

    branch.insert(HH())
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    voltages = jx.integrate(branch, delta_t=dt, voltage_solver=voltage_solver)

    voltages_300724 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703217397,
                57.69711962162869,
                27.503641673097068,
                -20.26260571077068,
                -44.42160900167978,
                -70.6396420990157,
                -75.76980477606747,
                -75.7453994826729,
                -75.46741856343822,
                -75.1296290199045,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_cell(voltage_solver, SimpleCell):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    num_branches = 3
    cell = SimpleCell(num_branches, ncomp=2)

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

    voltages_300724 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703217234,
                -1.8726881060703575,
                39.98176090770511,
                -2.853384097766068,
                -30.048377518182228,
                -54.259868414343295,
                -73.9359886731972,
                -75.74042745179534,
                -75.5791988501548,
                -75.26829188655702,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_radius_and_length_net(voltage_solver, SimpleNet):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    num_branches = 3
    net = SimpleNet(2, num_branches, 2)

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

    voltages_300724 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703217131,
                -1.8726881060701073,
                39.98176090770495,
                -2.8533840977659226,
                -30.04837751818565,
                -54.25986841434464,
                -73.93598867319699,
                -75.74042745179538,
                -75.57919885015413,
                -75.2682918865562,
            ],
            [
                -70.0,
                -66.47133094286725,
                -22.42632884453572,
                42.81477131595844,
                5.7572444690008036,
                -23.81466327013234,
                -47.82796157947444,
                -71.80106951477758,
                -75.57738636633145,
                -75.49698994483342,
                -75.17441392712844,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
