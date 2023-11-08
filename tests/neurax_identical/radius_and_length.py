from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import numpy as np
import jax.numpy as jnp

import neurax as nx
from neurax.channels import HHChannel
from neurax.synapses import GlutamateSynapse


def test_radius_and_length_compartment():
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)

    comp = nx.Compartment().initialize()

    np.random.seed(1)
    comp.set_params("length", 5 * np.random.rand(1))
    comp.set_params("radius", np.random.rand(1))

    comp.insert(HHChannel())

    current = nx.step_current(0.5, 1.0, 0.02, time_vec)
    comp.record()
    comp.stimulate(current)

    voltages = nx.integrate(comp, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_radius_and_length_branch():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()

    np.random.seed(1)
    branch.set_params("length", 5 * np.random.rand(2))
    branch.set_params("radius", np.random.rand(2))

    branch.insert(HHChannel())

    current = nx.step_current(0.5, 1.0, 0.02, time_vec)
    branch.comp(0.0).record()
    branch.comp(0.0).stimulate(current)

    voltages = nx.integrate(branch, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_radius_and_length_cell():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = nx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    cell.set_params("length", 5 * np.random.rand(2 * num_branches))
    cell.set_params("radius", np.random.rand(2 * num_branches))

    cell.insert(HHChannel())

    current = nx.step_current(0.5, 1.0, 0.02, time_vec)
    cell.branch(1).comp(0.0).record()
    cell.branch(1).comp(0.0).stimulate(current)

    voltage = nx.integrate(cell, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_radius_and_length_net():
    nseg_per_branch = 2
    dt = 0.025  # ms
    t_max = 5.0  # ms

    time_vec = jnp.arange(0.0, t_max + dt, dt)

    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    num_branches = len(parents)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell1 = nx.Cell([branch for _ in range(len(parents))], parents=parents)
    cell2 = nx.Cell([branch for _ in range(len(parents))], parents=parents)

    np.random.seed(1)
    cell1.set_params("length", 5 * np.random.rand(2 * num_branches))
    cell1.set_params("radius", np.random.rand(2 * num_branches))

    np.random.seed(2)
    cell2.set_params("length", 5 * np.random.rand(2 * num_branches))
    cell2.set_params("radius", np.random.rand(2 * num_branches))

    connectivities = [
        nx.Connectivity(GlutamateSynapse(), [nx.Connection(0, 0, 0.0, 1, 0, 0.0)])
    ]
    network = nx.Network([cell1, cell2], connectivities)
    network.insert(HHChannel())

    current = nx.step_current(0.5, 1.0, 0.02, time_vec)

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).comp(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).comp(0.0).stimulate(current)

    voltages = nx.integrate(network, delta_t=dt)

    voltages_081123 = None
    max_error = np.max(np.abs(voltages[:, ::10] - voltages_081123))
    tolerance = 0.0
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
