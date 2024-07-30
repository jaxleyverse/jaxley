# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

from math import pi

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect, fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse


def test_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = jx.Compartment()
    comp.insert(HH())
    comp.record()
    comp.stimulate(current)

    voltages = jx.integrate(comp, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -50.08343917,
                -20.4930498,
                32.6178327,
                2.57004029,
                -26.19517261,
                -50.50027566,
                -72.97019524,
                -75.69907171,
                -75.58874436,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_branch():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])
    branch.insert(HH())
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    voltages = jx.integrate(branch, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -56.84345977,
                -47.34841899,
                -32.63377325,
                30.99759112,
                8.03521283,
                -21.81119545,
                -45.8401624,
                -70.88960656,
                -75.66123707,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_cell():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])
    cell = jx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt)

    voltages_300724 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703217415,
                -61.30007495702876,
                -57.72869617299199,
                -57.907984804456134,
                -56.54785861252702,
                -54.63372123466727,
                -51.29073832791313,
                -41.84338223543283,
                12.629557592604355,
                16.375747329556404,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_net():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])
    cell1 = jx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = jx.Cell([branch for _ in range(len(parents))], parents=parents)

    network = jx.Network([cell1, cell2])
    connect(
        network.cell(0).branch(0).loc(0.0),
        network.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    network.insert(HH())

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).loc(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).loc(0.0).stimulate(current)

    area = 2 * pi * 10.0 * 1.0
    point_process_to_dist_factor = 100_000.0 / area
    network.IonotropicSynapse.set(
        "IonotropicSynapse_gS", 0.5 / point_process_to_dist_factor
    )
    voltages = jx.integrate(network, delta_t=dt)

    voltages_300724 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703217415,
                -61.300074957028755,
                -57.728696172992,
                -57.90798480445608,
                -56.54785861252695,
                -54.633721234667185,
                -51.29073832791293,
                -41.843382235431946,
                12.629557592599513,
                16.37574732956111,
            ],
            [
                -70.0,
                -66.16039111452706,
                -60.50177192696074,
                -56.47792566140247,
                -55.905759661906906,
                -52.98761212851238,
                -46.3513320143946,
                -13.082888133400296,
                25.761263776404153,
                -4.627404050763336,
                -30.796093242248364,
            ],
        ]
    )

    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_complex_net():
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net = jx.Network([cell for _ in range(7)])
    net.insert(HH())

    _ = np.random.seed(0)
    pre = net.cell([0, 1, 2])
    post = net.cell([3, 4, 5])
    fully_connect(pre, post, IonotropicSynapse())
    fully_connect(pre, post, TestSynapse())

    pre = net.cell([3, 4, 5])
    post = net.cell(6)
    fully_connect(pre, post, IonotropicSynapse())
    fully_connect(pre, post, TestSynapse())

    area = 2 * pi * 10.0 * 1.0
    point_process_to_dist_factor = 100_000.0 / area
    net.set("IonotropicSynapse_gS", 0.44 / point_process_to_dist_factor)
    net.set("TestSynapse_gC", 0.62 / point_process_to_dist_factor)
    net.IonotropicSynapse([0, 2, 4]).set(
        "IonotropicSynapse_gS", 0.32 / point_process_to_dist_factor
    )
    net.TestSynapse([0, 3, 5]).set(
        "TestSynapse_gC", 0.24 / point_process_to_dist_factor
    )

    current = jx.step_current(0.5, 0.5, 0.1, 0.025, 10.0)
    for i in range(3):
        net.cell(i).branch(0).loc(0.0).stimulate(current)

    net.cell(6).branch(0).loc(0.0).record()

    voltages = jx.integrate(net)

    voltages_300724 = jnp.asarray(
        [
            [
                -70.0,
                -64.5478417407739,
                -61.37506280116741,
                -58.27492707181707,
                -51.53610508254835,
                31.739248535280243,
                -23.276048469250686,
                -73.58311542313007,
                -75.5489796956953,
                -74.69162422333675,
                -73.52874849932951,
            ]
        ]
    )

    max_error = np.max(np.abs(voltages[:, ::40] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
