from jax import config

from jaxley.connection import connect

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
from math import pi

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse


def test_swc_cell():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.2, dt, t_max)

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../morph.swc")
    cell = jx.read_swc(fname, nseg=2, max_branch_len=300.0, assign_groups=True)
    _ = cell.soma  # Only to test whether the `soma` group was created.
    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt)

    voltages_081123 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -57.02065054,
                -49.74541341,
                -46.92144876,
                -27.85593412,
                25.00515973,
                4.52875333,
                -23.36842446,
                -46.25720697,
                -68.22233297,
            ]
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


def test_swc_net():
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.2, dt, t_max)

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../morph.swc")
    cell1 = jx.read_swc(fname, nseg=2, max_branch_len=300.0)
    cell2 = jx.read_swc(fname, nseg=2, max_branch_len=300.0)

    network = jx.Network([cell1, cell2])
    connect(
        network.cell(0).branch(0).loc(0.0),
        network.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    network.insert(HH())

    # first cell, 0-eth branch, 1-st compartment because loc=0.0 -> comp = nseg-1 = 1
    radius_post = network[1, 0, 1].view["radius"].item()
    lenght_post = network[1, 0, 1].view["length"].item()
    area = 2 * pi * lenght_post * radius_post
    point_process_to_dist_factor = 100_000.0 / area
    network.set("IonotropicSynapse_gS", 0.5 / point_process_to_dist_factor)

    for cell_ind in range(2):
        network.cell(cell_ind).branch(1).loc(0.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(network, delta_t=dt)

    voltages_040224 = jnp.asarray(
        [
            [
                -70.0,
                -66.53085703,
                -57.02065054,
                -49.74541341,
                -46.92144876,
                -27.85593412,
                25.00515973,
                4.52875333,
                -23.36842446,
                -46.25720697,
                -68.22233297,
            ],
            [
                -70.0,
                -66.52401241,
                -57.01032941,
                -49.7286954,
                -46.8735511,
                -27.60193351,
                25.04011476,
                4.35660272,
                -23.5081408,
                -46.38426466,
                -68.28499428,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_040224))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
