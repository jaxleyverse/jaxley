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


@pytest.mark.parametrize("voltage_solver", ["jaxley.stone", "jax.sparse"])
def test_radius_and_length_compartment(voltage_solver: str):
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = jx.Compartment()

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


@pytest.mark.parametrize("voltage_solver", ["jaxley.stone", "jax.sparse"])
def test_radius_and_length_branch(voltage_solver: str):
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])

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


@pytest.mark.parametrize("voltage_solver", ["jaxley.stone", "jax.sparse"])
def test_radius_and_length_cell(voltage_solver: str):
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])
    cell = jx.Cell([branch for _ in range(len(parents))], parents=parents)

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


@pytest.mark.parametrize("voltage_solver", ["jaxley.stone", "jax.sparse"])
def test_radius_and_length_net(voltage_solver: str):
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])
    cell1 = jx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = jx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    rands1 = 5 * np.random.rand(2 * num_branches)
    rands2 = np.random.rand(2 * num_branches)
    for b in range(num_branches):
        cell1.branch(b).set("length", np.flip(rands1[2 * b : 2 * b + 2]))
        cell1.branch(b).set("radius", np.flip(rands2[2 * b : 2 * b + 2]))

    np.random.seed(2)
    rands1 = 5 * np.random.rand(2 * num_branches)
    rands2 = np.random.rand(2 * num_branches)
    for b in range(num_branches):
        cell2.branch(b).set("length", np.flip(rands1[2 * b : 2 * b + 2]))
        cell2.branch(b).set("radius", np.flip(rands2[2 * b : 2 * b + 2]))

    network = jx.Network([cell1, cell2])
    connect(
        network.cell(0).branch(0).loc(0.0),
        network.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    network.insert(HH())

    # first cell, 0-eth branch, 0-st compartment because loc=0.0
    radius_post = network[1, 0, 0].view["radius"].item()
    lenght_post = network[1, 0, 0].view["length"].item()
    area = 2 * pi * lenght_post * radius_post
    point_process_to_dist_factor = 100_000.0 / area
    network.set("IonotropicSynapse_gS", 0.5 / point_process_to_dist_factor)

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).loc(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(network, delta_t=dt, voltage_solver=voltage_solver)

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
