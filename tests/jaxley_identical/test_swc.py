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
@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_swc_cell(voltage_solver: str, file: str):
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.2, dt, t_max)

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../swc_files", file)
    cell = jx.read_swc(fname, nseg=2, max_branch_len=300.0, assign_groups=True)
    _ = cell.soma  # Only to test whether the `soma` group was created.
    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)

    if file == "morph_single_point_soma.swc":
        voltages_300724 = jnp.asarray(
            [
                [
                    -70.0,
                    -66.53085703217431,
                    -57.86539099487056,
                    -50.68402476185379,
                    -47.116280145195034,
                    -26.54674397910345,
                    25.86902654237883,
                    2.7545847203679648,
                    -23.49507300176727,
                    -46.176932023665,
                    -68.13277872699207,
                ]
            ]
        )
    elif file == "morph.swc":
        voltages_300724 = jnp.asarray(
            [
                [
                    -70.0,
                    -66.53085703215352,
                    -57.02081109294148,
                    -49.74061944051031,
                    -46.887867347343146,
                    -27.613724386581872,
                    25.092386748387153,
                    4.367651506995822,
                    -23.53096242684505,
                    -46.43371990417977,
                    -68.37487528619394,
                ]
            ]
        )
    else:
        raise NameError
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.stone", "jax.sparse"])
def test_swc_net(voltage_solver: str):
    dt = 0.025  # ms
    t_max = 5.0  # ms
    current = jx.step_current(0.5, 1.0, 0.2, dt, t_max)

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../swc_files/morph.swc")
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
    radius_post = network[1, 0, 1].nodes["radius"].item()
    lenght_post = network[1, 0, 1].nodes["length"].item()
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
                -66.53085703215763,
                -57.020811092945316,
                -49.74061944051095,
                -46.88786734735504,
                -27.613724386645945,
                25.092386748379845,
                4.3676515070377,
                -23.530962426808408,
                -46.433719904148994,
                -68.37487528619184,
            ],
            [
                -70.0,
                -66.52401203413784,
                -57.01048571919544,
                -49.723881897201146,
                -46.83983197382557,
                -27.35802551334217,
                25.122266645646633,
                4.1962717936840805,
                -23.669714625969657,
                -46.560355102122436,
                -68.4360112878163,
            ],
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_300724))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
