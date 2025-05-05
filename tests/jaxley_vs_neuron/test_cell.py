# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"

import jax.numpy as jnp
import numpy as np
import pytest
from jaxley_mech.channels.l5pc import CaHVA
from neuron import h, rxd

import jaxley as jx
from jaxley.channels import HH
from jaxley.pumps import CaFaradayConcentrationChange, CaNernstReversal

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


@pytest.mark.parametrize("solver", ["bwd_euler", "crank_nicolson"])
def test_similarity(solver):
    """Test similarity of jaxley vs neuron."""
    i_delay = 3.0  # ms
    i_dur = 2.0  # ms
    i_amp = 1.0  # nA

    dt = 0.025  # ms
    t_max = 10.0  # ms

    voltages_jaxley = _run_jaxley(i_delay, i_dur, i_amp, dt, t_max, solver)
    voltages_neuron = _run_neuron(i_delay, i_dur, i_amp, dt, t_max, solver)

    assert np.mean(np.abs(voltages_jaxley - voltages_neuron)) < 0.05


def _run_jaxley(i_delay, i_dur, i_amp, dt, t_max, solver):
    ncomp_per_branch = 8
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp_per_branch)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(HH())

    cell.set("radius", 5.0)
    cell.set("length", 10.0)
    cell.set("axial_resistivity", 1_000.0)
    cell.set("capacitance", 7.0)

    cell.set("HH_gNa", 0.120)
    cell.set("HH_gK", 0.036)
    cell.set("HH_gLeak", 0.0003)
    cell.set("HH_m", 0.07490098835688629)
    cell.set("HH_h", 0.488947681848153)
    cell.set("HH_n", 0.3644787002343737)
    cell.set("v", -62.0)

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    cell.branch(0).loc(0.0).stimulate(current)
    cell.branch(0).loc(0.0).record()
    cell.branch(1).loc(1.0).record()
    cell.branch(2).loc(1.0).record()

    voltages = jx.integrate(cell, delta_t=dt, solver=solver)
    return voltages


def _run_neuron(i_delay, i_dur, i_amp, dt, t_max, solver):
    if solver == "bwd_euler":
        h.secondorder = 0
    elif solver == "crank_nicolson":
        h.secondorder = 1
    else:
        raise ValueError

    ncomp_per_branch = 8
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

    branch1 = h.Section()
    branch2 = h.Section()
    branch3 = h.Section()

    branch2.connect(branch1, 1, 0)
    branch3.connect(branch1, 1, 0)

    for sec in h.allsec():
        sec.nseg = ncomp_per_branch

        sec.Ra = 1_000.0
        sec.L = 10.0 * ncomp_per_branch
        sec.diam = 2 * 5.0
        sec.cm = 7.0

        sec.insert("hh")
        sec.gnabar_hh = 0.120  # S/cm2
        sec.gkbar_hh = 0.036  # S/cm2
        sec.gl_hh = 0.0003  # S/cm2
        sec.ena = 50  # mV
        sec.ek = -77.0  # mV
        sec.el_hh = -54.3  # mV

    stim = h.IClamp(branch1(0.05))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    voltage1 = h.Vector()
    voltage1.record(branch1(0.05)._ref_v)
    voltage2 = h.Vector()
    voltage2.record(branch2(0.95)._ref_v)
    voltage3 = h.Vector()
    voltage3.record(branch3(0.95)._ref_v)

    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    voltages = np.asarray([list(voltage1), list(voltage2), list(voltage3)])
    return voltages


def test_similarity_unequal_number_of_compartments():
    """Test similarity of jaxley vs neuron."""
    i_delay = 3.0  # ms
    i_dur = 2.0  # ms
    i_amp = 1.0  # nA

    dt = 0.025  # ms
    t_max = 10.0  # ms

    voltages_jaxley = _run_jaxley_unequal_ncomp(i_delay, i_dur, i_amp, dt, t_max)
    voltages_neuron = _run_neuron_unequal_ncomp(i_delay, i_dur, i_amp, dt, t_max)

    assert np.mean(np.abs(voltages_jaxley - voltages_neuron)) < 0.05


def _run_jaxley_unequal_ncomp(i_delay, i_dur, i_amp, dt, t_max):
    comp = jx.Compartment()
    branch1 = jx.Branch(comp, ncomp=1)
    branch2 = jx.Branch(comp, ncomp=2)
    branch3 = jx.Branch(comp, ncomp=3)
    branch4 = jx.Branch(comp, ncomp=4)
    cell = jx.Cell([branch1, branch2, branch3, branch4], parents=[-1, 0, 0, 1])
    cell.set("axial_resistivity", 10_000.0)
    cell.insert(HH())

    cell.set("radius", 5.0)
    cell.set("length", 20.0)
    cell.set("axial_resistivity", 1_000.0)
    cell.branch(1).set("capacitance", 10.0)
    cell.branch(3).set("capacitance", 20.0)

    cell.set("HH_gNa", 0.120)
    cell.set("HH_gK", 0.036)
    cell.set("HH_gLeak", 0.0003)
    cell.set("HH_m", 0.07490098835688629)
    cell.set("HH_h", 0.488947681848153)
    cell.set("HH_n", 0.3644787002343737)
    cell.set("v", -62.0)

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    cell.branch(1).comp(1).stimulate(current)
    cell.branch(0).comp(0).record()
    cell.branch(2).comp(2).record()
    cell.branch(3).comp(1).record()
    cell.branch(3).comp(3).record()

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver="jaxley.dhs.cpu")
    return voltages


def _run_neuron_unequal_ncomp(i_delay, i_dur, i_amp, dt, t_max):
    h.secondorder = 0
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

    branch1 = h.Section()
    branch2 = h.Section()
    branch3 = h.Section()
    branch4 = h.Section()

    branch2.connect(branch1, 1, 0)
    branch3.connect(branch1, 1, 0)
    branch4.connect(branch2, 1, 0)

    ncomps = [1, 2, 3, 4]
    for i, sec in enumerate(h.allsec()):
        sec.nseg = ncomps[i]

        sec.Ra = 1_000.0
        sec.L = 20.0 * ncomps[i]
        sec.diam = 2 * 5.0

        sec.insert("hh")
        sec.gnabar_hh = 0.120  # S/cm2
        sec.gkbar_hh = 0.036  # S/cm2
        sec.gl_hh = 0.0003  # S/cm2
        sec.ena = 50  # mV
        sec.ek = -77.0  # mV
        sec.el_hh = -54.3  # mV

        if i == 1:
            sec.cm = 10.0
        if i == 3:
            sec.cm = 20.0

    stim = h.IClamp(branch2(0.6))  # The second out of two.
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    voltage1 = h.Vector()
    voltage1.record(branch1(0.5)._ref_v)  # Only 1 compartment.
    voltage2 = h.Vector()
    voltage2.record(branch3(0.8)._ref_v)  # The third out of three compartments.
    voltage3 = h.Vector()
    voltage3.record(branch4(0.3)._ref_v)  # The second out of four comps.
    voltage4 = h.Vector()
    voltage4.record(branch4(0.9)._ref_v)  # The last out of four comps.

    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    voltages = np.asarray(
        [list(voltage1), list(voltage2), list(voltage3), list(voltage4)]
    )
    return voltages


def _jaxley_ion_diffusion(
    dt, t_max, i_delay, i_dur, i_amp, v_init, diams, lengths, r_as
):
    comp = jx.Compartment()
    ncomp = 2
    branch = jx.Branch(comp, ncomp=ncomp)
    cell = jx.Cell(branch, parents=[-1, 0, 0])

    for i in range(3):
        cell.branch(i).set("radius", diams[i] / 2)
        cell.branch(i).set("axial_resistivity", r_as[i])
        cell.branch(i).set("length", lengths[i] / ncomp)

    cell.set("v", v_init)
    cell.branch(0).insert(HH())

    # Define calcium dynamics.
    cell.branch([0, 1]).insert(CaHVA())
    cell.insert(CaFaradayConcentrationChange())
    cell.insert(CaNernstReversal())
    cell.branch(0).set("CaHVA_gCaHVA", 0.00001)
    cell.branch(1).set("CaHVA_gCaHVA", 0.000005)
    cell.set("eCa", 0.0)

    cell.diffuse("CaCon_i")
    cell.set("axial_diffusion_CaCon_i", 1.0)

    cell.init_states()

    # Stimulate.
    step_current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    cell.branch(0).comp(0).stimulate(step_current)

    # Recordings of voltage and cacon_i at different compartments.
    cell.branch(0).comp(0).record("v")
    cell.branch(1).comp(1).record("v")
    cell.branch(2).comp(0).record("v")
    cell.branch(0).comp(1).record("CaCon_i")
    cell.branch(1).comp(0).record("CaCon_i")
    cell.branch(2).comp(1).record("CaCon_i")

    recordings = jx.integrate(cell, t_max=t_max, delta_t=dt)
    return recordings


def _neuron_ion_diffusion(
    dt, t_max, i_delay, i_dur, i_amp, v_init, diams, lengths, r_as
):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../arm64/.libs/libnrnmech.so")
    h.nrn_load_dll(fname)

    h.secondorder = 0
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

    branch0 = h.Section()
    branch1 = h.Section()
    branch2 = h.Section()

    branch0.insert("hh")

    for i, branch in enumerate([branch0, branch1, branch2]):
        branch.nseg = 2
        branch.Ra = r_as[i]
        branch.L = lengths[i]
        branch.diam = diams[i]

    branch0.insert("Ca_HVA")
    branch1.insert("Ca_HVA")
    branch0.gCa_HVAbar_Ca_HVA = 0.00001
    branch1.gCa_HVAbar_Ca_HVA = 0.000005

    branch1.connect(branch0, 1, 0)
    branch2.connect(branch0, 1, 0)

    # Set diffusion to zero. This is causing the different interpretation of `.area()`
    # of NEURON and therefore causes this test to fail (also for d>0) if the radiuses
    # of compartments are not the same (as is the case here).
    r = rxd.Region(h.allsec(), nrn_region="i", geometry=None)
    ca = rxd.Species(r, d=1, name="ca", charge=2)

    # Stimulate.
    stim = h.IClamp(branch0(0.3))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    # Voltage recordings.
    v0 = h.Vector()
    v0.record(branch0(0.3)._ref_v)
    v1 = h.Vector()
    v1.record(branch1(0.7)._ref_v)
    v2 = h.Vector()
    v2.record(branch2(0.3)._ref_v)

    # Calcium recordings.
    cacon0 = h.Vector()
    cacon0.record(branch0(0.7)._ref_cai)
    cacon1 = h.Vector()
    cacon1.record(branch1(0.3)._ref_cai)
    cacon2 = h.Vector()
    cacon2.record(branch2(0.7)._ref_cai)

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    v0 = np.asarray(list(v0))
    v1 = np.asarray(list(v1))
    v2 = np.asarray(list(v2))
    cacon0 = np.asarray(list(cacon0))
    cacon1 = np.asarray(list(cacon1))
    cacon2 = np.asarray(list(cacon2))

    return np.stack([v0, v1, v2, cacon0, cacon1, cacon2])


def test_similarity_ion_diffusion():
    """Test whether ion diffusion behaves as in NEURON.

    This test works, but it can cause a segmentation fault on my MacBook (locally).
    The segmentation fault does not happen while running this, but it happens in
    one of the later calls to NEURON (e.g. in `test_comp.py`). In addition, this test
    relies on having compiled NEURON channel models, which I have not yet set up a
    Github action for.

    This test is closely related to the test
    `pytest test_branch.py -k test_similarity_voltage_clamp_cacon_i_behavior`
    which is skipped (because it fails for different diams between compartments.)
    """
    t_max = 1_000.01
    dt = 0.025

    # The test passes only when diams has the same values. See also the skipped test in
    # `pytest test_branch.py -k test_similarity_voltage_clamp_cacon_i_behavior`
    diams = [8.5, 8.5, 8.5]
    lengths = [100.0, 50.0, 70.0]  # Lengths of each branch.
    r_as = [1_000.0, 500.0, 750.0]  # Axial voltage resistivity of each branch.
    v_init = -65.0
    i_delay = 50.0
    i_dur = 20.0
    i_amp = 0.1

    v_and_cacon_jaxley = _jaxley_ion_diffusion(
        dt, t_max, i_delay, i_dur, i_amp, v_init, diams, lengths, r_as
    )
    # We are not running NEURON, and instead just saving its voltages below. This
    # is because rxd sometimes caused a segmentation fault in our tests.
    # v_and_cacon_neuron = _neuron_ion_diffusion(
    #     dt, t_max, i_delay, i_dur, i_amp, v_init, diams, lengths, r_as
    # )

    # Analysis.
    v_jaxley = v_and_cacon_jaxley[:3]
    cacon_jaxley = v_and_cacon_jaxley[3:]
    # v_neuron = v_and_cacon_neuron[:3]
    # cacon_neuron = v_and_cacon_neuron[3:]

    v_neuron = jnp.asarray(
        [
            [
                -64.9740566,
                -60.88748118,
                -57.9800073,
                -54.45089083,
                21.87809771,
                -73.1579158,
                -73.58213181,
                -70.61212201,
                -67.2714086,
                -67.58651223,
            ],
            [
                -64.97405911,
                -61.7164889,
                -58.78402151,
                -55.49142503,
                13.19733503,
                -69.01489137,
                -74.21113676,
                -71.42037101,
                -68.0896641,
                -67.67389774,
            ],
            [
                -64.97405923,
                -61.73392024,
                -58.79977817,
                -55.51516832,
                12.12792837,
                -68.73055718,
                -74.22165672,
                -71.43820274,
                -68.10784275,
                -67.67786202,
            ],
        ]
    )
    cacon_neuron_spike = (
        jnp.asarray(
            [
                [
                    5.00000453,
                    5.00000486,
                    5.00000626,
                    5.0000127,
                    5.1048348,
                    8.79260654,
                    9.30190049,
                    9.30284364,
                    9.29821517,
                    9.29349743,
                ],
                [
                    5.00000227,
                    5.00000243,
                    5.00000307,
                    5.00000599,
                    5.02163309,
                    6.81641896,
                    7.16615776,
                    7.16847423,
                    7.16688202,
                    7.16524459,
                ],
                [
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.00000213,
                    5.00001658,
                    5.00004703,
                    5.00009347,
                    5.00015577,
                ],
            ]
        )
        * 1e-5
    )
    cacon_neuron_end = (
        jnp.asarray(
            [
                [
                    8.30158656,
                    8.29842432,
                    8.29527959,
                    8.29215224,
                    8.28904216,
                    8.28594922,
                    8.2828733,
                    8.27981429,
                    8.27677206,
                    8.2737465,
                ],
                [
                    7.10557499,
                    7.10586209,
                    7.10614861,
                    7.10643455,
                    7.10671989,
                    7.10700463,
                    7.10728874,
                    7.10757223,
                    7.10785509,
                    7.10813731,
                ],
                [
                    5.46553328,
                    5.4689818,
                    5.47242902,
                    5.47587487,
                    5.47931927,
                    5.48276214,
                    5.48620342,
                    5.48964301,
                    5.49308086,
                    5.49651688,
                ],
            ]
        )
        * 1e-5
    )

    v_max_error = np.max(np.abs(v_jaxley[:, 2_000:3_000:100] - v_neuron))
    cacon_max_error_spike = np.max(
        np.abs(cacon_jaxley[:, 2_000:3_000:100] - cacon_neuron_spike)
    )
    cacon_max_error_end = np.max(np.abs(cacon_jaxley[:, -2000::200] - cacon_neuron_end))
    # Note: for very thin cables (<1um radius), this test can fail by a tiny bit, even
    # if both comps have the same diameter.
    assert cacon_max_error_spike < 2e-6, f"CaCon_i error {cacon_max_error_spike}"
    assert cacon_max_error_end < 2e-6, f"CaCon_i error {cacon_max_error_end}"

    # To be on the safe side, we also check voltages. However, this test is generally
    # about CaCon_i.
    assert v_max_error < 1e-2, f"Voltage error {v_max_error}"
