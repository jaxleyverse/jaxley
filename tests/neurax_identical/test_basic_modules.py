from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np

import neurax as nx
from neurax.channels import HHChannel
from neurax.synapses import GlutamateSynapse


def test_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = nx.Compartment().initialize()
    comp.insert(HHChannel())
    comp.record()
    comp.stimulate(current)

    voltages = nx.integrate(comp, delta_t=dt)

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
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    branch.insert(HHChannel())
    branch.comp(0.0).record()
    branch.comp(0.0).stimulate(current)

    voltages = nx.integrate(branch, delta_t=dt)

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
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell.insert(HHChannel())
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -60.77514388,
                -57.04487857,
                -57.6428012,
                -56.38585422,
                -54.48571678,
                -51.04843459,
                -40.91384567,
                16.39867998,
                14.60929039,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_net():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = nx.step_current(0.5, 1.0, 0.02, dt, t_max)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell1 = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = nx.Cell([branch for _ in range(len(parents))], parents=parents)

    connectivities = [
        nx.Connectivity(GlutamateSynapse(), [nx.Connection(0, 0, 0.0, 1, 0, 0.0)])
    ]
    network = nx.Network([cell1, cell2], connectivities)
    network.insert(HHChannel())

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).comp(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(network, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -60.77514388,
                -57.04487857,
                -57.6428012,
                -56.38585422,
                -54.48571678,
                -51.04843459,
                -40.91384567,
                16.39867998,
                14.60929039,
            ],
            [
                -70.0,
                -66.12980895,
                -59.94208128,
                -55.74082517,
                -55.53078341,
                -52.62810187,
                -45.67771987,
                -9.62810283,
                24.39010232,
                -6.25386699,
                -32.08138378,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
