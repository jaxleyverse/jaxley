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
import pytest
from jaxley_mech.channels.l5pc import CaHVA

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect, fully_connect
from jaxley.pumps import CaFaradayConcentrationChange, CaNernstReversal
from jaxley.synapses import IonotropicSynapse, TestSynapse


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_compartment(voltage_solver, SimpleComp, SimpleBranch, SimpleCell, SimpleNet):
    """Test a compartment."""
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )
    tolerance = 1e-8

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

    # Test compartment.
    comp = SimpleComp()
    comp.insert(HH())
    comp.record()
    comp.stimulate(current)
    voltages = jx.integrate(comp, delta_t=dt, voltage_solver=voltage_solver)
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    assert max_error <= tolerance, f"Compartment error is {max_error} > {tolerance}"

    # Test branch of a single compartment.
    branch = SimpleBranch(ncomp=1)
    branch.insert(HH())
    branch.record()
    branch.stimulate(current)
    voltages = jx.integrate(branch, delta_t=dt, voltage_solver=voltage_solver)
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    assert max_error <= tolerance, f"Branch error is {max_error} > {tolerance}"

    # Test cell of a single compartment.
    cell = SimpleCell(1, 1)
    cell.insert(HH())
    cell.record()
    cell.stimulate(current)
    voltages = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    assert max_error <= tolerance, f"Cell error is {max_error} > {tolerance}"

    # Test net of a single compartment.
    net = SimpleNet(1, 1, 1)
    net.insert(HH())
    net.record()
    net.stimulate(current)
    voltages = jx.integrate(net, delta_t=dt, voltage_solver=voltage_solver)
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_081123))
    assert max_error <= tolerance, f"Network error is {max_error} > {tolerance}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_branch(voltage_solver, SimpleBranch):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    branch = SimpleBranch(2)
    branch.insert(HH())
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(current)

    voltages = jx.integrate(branch, delta_t=dt, voltage_solver=voltage_solver)

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


@pytest.mark.parametrize("solver", ["fwd_euler", "bwd_euler"])
def test_branch_uneven_radiuses(SimpleBranch, solver):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=2.0, delta_t=0.025, t_max=10.0
    )

    branch = SimpleBranch(8)
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

    if solver == "fwd_euler":
        voltages_240920 = jnp.asarray(
            [
                -70.0,
                -64.319374,
                -61.61975,
                -56.971237,
                25.785686,
                -42.466354,
                -75.86178,
                -75.06558,
                -73.95041,
            ]
        )
    else:
        voltages_240920 = jnp.asarray(
            [
                -70.0,
                -64.36480976,
                -61.66232751,
                -56.87833254,
                28.49415671,
                -39.54294182,
                -75.89996029,
                -75.16066305,
                -74.07568059,
            ]
        )
    tolerance = 1e-5
    max_error = jnp.max(jnp.abs(voltages_240920 - voltages[0, ::50]))
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_cell(voltage_solver, SimpleCell):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    cell = SimpleCell(3, 2)
    cell.insert(HH())
    cell.branch(1).loc(0.0).record()
    cell.branch(1).loc(0.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)

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


@pytest.mark.parametrize(
    "voltage_solver", ["jaxley.dhs.cpu", "jaxley.dhs.gpu", "jax.sparse"]
)
def test_complex_cell(voltage_solver, SimpleBranch):
    """Test cell with a variety of channels, pumps, diffusion, and comps per branch."""
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    branch1 = SimpleBranch(ncomp=1)
    branch2 = SimpleBranch(ncomp=2)
    branch3 = SimpleBranch(ncomp=3)
    branch4 = SimpleBranch(ncomp=4)
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

    recordings = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)
    voltages_240225 = jnp.asarray(
        [
            [-70.0, -53.80348826, 22.07746319, -47.29564472, -75.61534365],
            [-70.0, -61.74865308, 34.1167336, -28.89225642, -75.73529806],
            [-70.0, -52.99907533, 20.9735932, -47.83940886, -75.66904041],
            [-70.0, -60.86487129, 35.11018305, -33.89819309, -75.76933684],
        ]
    )
    cacon_240225 = (
        jnp.asarray(
            [
                [5.0, 5.04797687, 6.45337663, 12.1139712, 11.86289905],
                [5.0, 5.0063692, 5.02944485, 5.72251137, 7.49644484],
                [5.0, 5.00170413, 5.00819813, 5.20138788, 5.87906979],
                [5.0, 5.00020329, 5.00223143, 5.03325016, 5.27678607],
            ]
        )
        * 1e-5
    )
    max_error_ca = np.max(np.abs(recordings[4:, ::50] - cacon_240225))
    max_error_v = np.max(np.abs(recordings[:4, ::50] - voltages_240225))
    tolerance_ca = 1e-10
    tolerance_v = 1e-8
    assert max_error_ca <= tolerance_ca, f"Ca Error is {max_error_ca} > {tolerance_ca}"
    assert max_error_v <= tolerance_v, f"Voltage Error is {max_error_v} > {tolerance_v}"


@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
def test_net(voltage_solver, SimpleCell):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.02, delta_t=0.025, t_max=5.0
    )

    # net = SimpleNet(2, 3, 2)
    cell1 = SimpleCell(3, 2)
    cell2 = SimpleCell(3, 2)
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


@pytest.mark.parametrize(
    "voltage_solver", ["jaxley.dhs.cpu", "jaxley.dhs.gpu", "jax.sparse"]
)
def test_complex_net(voltage_solver, SimpleCell):
    cell = SimpleCell(5, 4)
    if voltage_solver == "jaxley.dhs.gpu":
        # On CPU we have to run this manually. On GPU, it gets run automatically with
        # allowed_nodes_per_level=32.
        cell._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=4)

    if voltage_solver == "jaxley.dhs.cpu":
        net = jx.Network([cell for _ in range(7)], vectorize_cells=False)
    else:
        net = jx.Network([cell for _ in range(7)], vectorize_cells=True)

    net.insert(HH())

    _ = np.random.seed(0)
    pre = net.cell([0, 1, 2])
    post = net.cell([3, 4, 5])
    fully_connect(pre, post, IonotropicSynapse(), random_post_comp=True)
    fully_connect(pre, post, TestSynapse(), random_post_comp=True)

    pre = net.cell([3, 4, 5])
    post = net.cell(6)
    fully_connect(pre, post, IonotropicSynapse(), random_post_comp=True)
    fully_connect(pre, post, TestSynapse(), random_post_comp=True)

    area = 2 * pi * 10.0 * 1.0
    point_process_to_dist_factor = 100_000.0 / area
    net.set("IonotropicSynapse_gS", 0.44 / point_process_to_dist_factor)
    net.set("TestSynapse_gC", 0.62 / point_process_to_dist_factor)
    net.IonotropicSynapse.edge([0, 2, 4]).set(
        "IonotropicSynapse_gS", 0.32 / point_process_to_dist_factor
    )
    net.TestSynapse.edge([0, 3, 5]).set(
        "TestSynapse_gC", 0.24 / point_process_to_dist_factor
    )

    current = jx.step_current(
        i_delay=0.5, i_dur=0.5, i_amp=0.1, delta_t=0.025, t_max=10.0
    )
    for i in range(3):
        net.cell(i).branch(0).loc(0.0).stimulate(current)

    net.cell(6).branch(0).loc(0.0).record()

    voltages = jx.integrate(net, voltage_solver=voltage_solver)

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
