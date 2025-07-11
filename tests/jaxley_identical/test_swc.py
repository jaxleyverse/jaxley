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
from jaxley_mech.channels.l5pc import *

import jaxley as jx
from jaxley.channels import HH, K, Leak, Na
from jaxley.synapses import IonotropicSynapse


@pytest.mark.slow
@pytest.mark.parametrize("voltage_solver", ["jaxley.dhs.cpu"])
@pytest.mark.parametrize(
    "file", ["morph_ca1_n120_single_point_soma.swc", "morph_ca1_n120.swc"]
)
def test_swc_cell(voltage_solver: str, file: str, SimpleMorphCell):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.2, delta_t=0.025, t_max=5.0
    )

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../swc_files", file)
    cell = SimpleMorphCell(fname, ncomp=2, max_branch_len=300.0)
    _ = cell.soma  # Only to test whether the `soma` group was created.
    cell.insert(HH())
    cell.soma.branch(0).loc(1.0).record()
    cell.soma.branch(0).loc(1.0).stimulate(current)

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver=voltage_solver)

    if file == "morph_ca1_n120_single_point_soma.swc":
        voltages_170425 = jnp.asarray(
            [
                [
                    -70.0,
                    -66.53085703,
                    -59.29790807,
                    -53.52391766,
                    -52.35430432,
                    -47.27344832,
                    -26.23156449,
                    26.36152268,
                    2.32087057,
                    -25.39217393,
                    -48.79067959,
                ]
            ]
        )
    elif file == "morph_ca1_n120.swc":
        voltages_170425 = jnp.asarray(
            [
                [
                    -70.0,
                    -66.53085703,
                    -56.74039791,
                    -49.09988229,
                    -45.18674417,
                    -18.02906416,
                    24.57915633,
                    -1.09081191,
                    -27.87178471,
                    -50.71482553,
                    -70.38637602,
                ]
            ]
        )
    else:
        raise NameError
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_170425))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("voltage_solver", "morph"),
    [
        ("jaxley.dhs.cpu", "morph_ca1_n120"),
        ("jax.sparse", "morph_ca1_n120"),
        ("jaxley.dhs.cpu", "morph_ca1_n120_250"),
        ("jaxley.dhs.gpu", "morph_ca1_n120_250"),
        # Don't run .gpu on large morph, compile time is ~40 seconds.
    ],
)
def test_swc_net(voltage_solver: str, morph: str, SimpleMorphCell):
    dt = 0.025  # ms
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.2, delta_t=0.025, t_max=5.0
    )

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, f"../swc_files/{morph}.swc")
    cell1 = SimpleMorphCell(fname, ncomp=2, max_branch_len=300.0)
    cell2 = SimpleMorphCell(fname, ncomp=2, max_branch_len=300.0)

    if voltage_solver == "jaxley.dhs.gpu":
        # On CPU we have to run this manually. On GPU, it gets run automatically with
        # allowed_nodes_per_level=32.
        cell1._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=4)
        cell2._init_solver_jaxley_dhs_solve(allowed_nodes_per_level=4)

    if voltage_solver == "jaxley.dhs.gpu":
        network = jx.Network([cell1, cell2], vectorize_cells=True)
    else:
        network = jx.Network([cell1, cell2], vectorize_cells=False)

    connect(
        network.cell(0).soma.branch(0).loc(1.0),
        network.cell(1).soma.branch(0).loc(1.0),
        IonotropicSynapse(),
    )
    network.insert(HH())

    # first cell, 0-eth branch, 1-st compartment because loc=0.0 -> comp = ncomp-1 = 1
    radius_post = network.cell(1).soma.branch(0).comp(1).nodes["radius"].item()
    length_post = network.cell(1).soma.branch(0).comp(1).nodes["length"].item()
    area = 2 * pi * length_post * radius_post
    point_process_to_dist_factor = 100_000.0 / area
    network.set("IonotropicSynapse_gS", 0.5 / point_process_to_dist_factor)

    for cell_ind in range(2):
        network.cell(cell_ind).soma.branch(0).loc(1.0).record()

    for stim_ind in range(2):
        network.cell(stim_ind).soma.branch(0).loc(1.0).stimulate(current)

    voltages = jx.integrate(network, delta_t=dt, voltage_solver=voltage_solver)

    if morph == "morph_ca1_n120":
        voltages_250508 = np.asarray(
            [
                [
                    -70.0,
                    -66.53085703,
                    -56.74039791,
                    -49.09988229,
                    -45.18674417,
                    -18.02906416,
                    24.57915633,
                    -1.09081191,
                    -27.87178471,
                    -50.71482553,
                    -70.38637602,
                ],
                [
                    -70.0,
                    -65.71986463,
                    -55.51779941,
                    -46.97870197,
                    -37.46298705,
                    16.25680606,
                    13.7342436,
                    -14.51311616,
                    -38.04526019,
                    -60.3871947,
                    -71.55525201,
                ],
            ]
        )
    else:
        voltages_250508 = np.asarray(
            [
                [
                    -70.0,
                    -66.53085703,
                    -49.21898099,
                    -22.94224484,
                    28.74045665,
                    2.91546483,
                    -23.02591871,
                    -42.61463332,
                    -64.1234962,
                    -71.49147031,
                    -73.34623772,
                ],
                [
                    -70.0,
                    -65.59415424,
                    -47.68599328,
                    -16.57527038,
                    27.41181873,
                    -0.68072272,
                    -25.03181138,
                    -44.1761724,
                    -64.08079883,
                    -70.0792351,
                    -71.66197441,
                ],
            ]
        )
    max_error = np.max(np.abs(voltages[:, ::20] - voltages_250508))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"


# This test will be skipped for now, due to weird quirk of the swc file, which has two
# different radii for the same xyz coordinates (L10364: 10362 -> 10549). This is handled
# by the differently by the two swc reader backends (graph seems to be the correct one,
# compared to NEURON).
@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize("swc_backend", ["custom", "graph"])
def test_swc_morph(swc_backend, SimpleMorphCell):
    gt_apical = {}
    gt_soma = {}
    gt_axon = {}

    gt_apical["apical_NaTs2T_gNaTs2T"] = 0.026145
    gt_apical["apical_SKv3_1_gSKv3_1"] = 0.004226
    gt_apical["apical_M_gM"] = 0.000143

    gt_soma["somatic_NaTs2T_gNaTs2T"] = 0.983955
    gt_soma["somatic_SKv3_1_gSKv3_1"] = 0.303472
    gt_soma["somatic_SKE2_gSKE2"] = 0.008407
    gt_soma["somatic_CaPump_gamma"] = 0.000609
    gt_soma["somatic_CaPump_decay"] = 210.485291
    gt_soma["somatic_CaHVA_gCaHVA"] = 0.000994
    gt_soma["somatic_CaLVA_gCaLVA"] = 0.000333

    gt_axon["axonal_NaTaT_gNaTaT"] = 3.137968
    gt_axon["axonal_KPst_gKPst"] = 0.973538
    gt_axon["axonal_KTst_gKTst"] = 0.089259
    gt_axon["axonal_SKE2_gSKE2"] = 0.007104
    gt_axon["axonal_SKv3_1_gSKv3_1"] = 1.021945
    gt_axon["axonal_CaHVA_gCaHVA"] = 0.00099
    gt_axon["axonal_CaLVA_gCaLVA"] = 0.008752
    gt_axon["axonal_CaPump_gamma"] = 0.00291
    gt_axon["axonal_CaPump_decay"] = 287.19873

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../swc_files", "bbp_with_axon.swc")  # n120
    cell = SimpleMorphCell(fname, ncomp=2, swc_backend=swc_backend)

    # custom swc reader does not label the root branch that is added to the soma
    # while the graph swc reader does. This is accounted for here.
    cell.groups["soma"] = (
        cell.groups["soma"][2:] if swc_backend == "graph" else cell.groups["soma"]
    )
    apical_inds = cell.groups["apical"]

    ########## APICAL ##########
    cell.apical.set("capacitance", 2.0)
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))

    for c in apical_inds:
        distance = cell.scope("global").comp(c).distance(cell.branch(1).loc(0.0))
        cond = (-0.8696 + 2.087 * np.exp(distance * 0.0031)) * 8e-5
        cell.scope("global").comp(c).set("apical_H_gH", cond)

    ########## SOMA ##########
    cell.soma.insert(NaTs2T().change_name("somatic_NaTs2T"))
    cell.soma.insert(SKv3_1().change_name("somatic_SKv3_1"))
    cell.soma.insert(SKE2().change_name("somatic_SKE2"))
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.insert(CaPump().change_name("somatic_CaPump"))
    cell.soma.insert(CaHVA().change_name("somatic_CaHVA"))
    cell.soma.insert(CaLVA().change_name("somatic_CaLVA"))
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)

    ########## BASAL ##########
    cell.basal.insert(H().change_name("basal_H"))
    cell.basal.set("basal_H_gH", 8e-5)

    # ########## AXON ##########
    cell.insert(CaNernstReversal())
    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)
    cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    cell.axon.insert(KTst().change_name("axonal_KTst"))
    cell.axon.insert(CaPump().change_name("axonal_CaPump"))
    cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    cell.axon.insert(KPst().change_name("axonal_KPst"))
    cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))
    cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))

    ########## WHOLE CELL  ##########
    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)

    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)
    cell.set("v", -65.0)

    for key in gt_apical.keys():
        cell.apical.set(key, gt_apical[key])

    for key in gt_soma.keys():
        cell.soma.set(key, gt_soma[key])

    for key in gt_axon.keys():
        cell.axon.set(key, gt_axon[key])

    dt = 0.025
    t_max = 100.0
    time_vec = np.arange(0, t_max + 2 * dt, dt)

    cell.delete_stimuli()
    cell.delete_recordings()

    i_delay = 10.0
    i_dur = 80.0
    i_amp = 3.0
    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    cell.branch(1).comp(0).stimulate(current)
    cell.branch(1).comp(0).record()

    cell.set("v", -65.0)
    cell.init_states()

    voltages = jx.integrate(cell)

    voltages_250130 = jnp.asarray(
        [
            -65.0,
            -66.22422623,
            -67.23001452,
            -68.06298803,
            -68.75766951,
            -33.91317711,
            -55.24503749,
            -46.11452291,
            -42.18960646,
            -51.12861864,
            -43.65442616,
            -40.62727385,
            -49.56110473,
            -43.24030949,
            -36.71731271,
            -48.7405489,
            -42.98507829,
            -34.64282586,
            -48.24427898,
            -42.6412365,
            -34.70568206,
            -47.90643598,
            -42.15688181,
            -36.17711814,
            -47.65564274,
            -41.52265914,
            -38.1627371,
            -47.44680473,
            -40.70730741,
            -40.15298353,
            -47.25483146,
            -39.63994798,
            -41.96818737,
            -47.06569105,
            -38.17257448,
            -43.50053648,
            -46.87517934,
            -65.40488865,
            -69.96981343,
            -72.24384111,
            -73.46204372,
        ]
    )
    max_error = np.max(np.abs(voltages[:, ::100] - voltages_250130))
    tolerance = 1e-8
    assert max_error <= tolerance, f"Error is {max_error} > {tolerance}"
