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
    """Test similarity of jaxley vs neuron for a branch.

    The branch has an uneven radius.
    """
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
    branch = jx.Branch([comp for _ in range(ncomp_per_branch)])
    branch.insert(HH())

    radiuses = np.linspace(3.0, 15.0, ncomp_per_branch)
    for i, loc in enumerate(np.linspace(0, 1, ncomp_per_branch)):
        branch.loc(loc).set("radius", radiuses[i])

    branch.set("length", 10.0)
    branch.set("axial_resistivity", 1_000.0)
    branch.set("capacitance", 5.0)

    branch.set("HH_gNa", 0.120)
    branch.set("HH_gK", 0.036)
    branch.set("HH_gLeak", 0.0003)
    branch.set("HH_m", 0.07490098835688629)
    branch.set("HH_h", 0.488947681848153)
    branch.set("HH_n", 0.3644787002343737)
    branch.set("v", -62.0)

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    branch.loc(0.0).stimulate(current)
    branch.loc(0.0).record()
    branch.loc(1.0).record()

    voltages = jx.integrate(branch, delta_t=dt, solver=solver)

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
    branch = h.Section()

    branch.nseg = ncomp_per_branch
    branch.Ra = 1_000.0
    branch.L = 10.0 * ncomp_per_branch
    branch.cm = 5.0

    radiuses = np.linspace(3.0, 15.0, ncomp_per_branch)
    for i, comp in enumerate(branch):
        comp.diam = 2 * radiuses[i]

    branch.insert("hh")
    branch.gnabar_hh = 0.120  # S/cm2
    branch.gkbar_hh = 0.036  # S/cm2
    branch.gl_hh = 0.0003  # S/cm2
    branch.ena = 50  # mV
    branch.ek = -77.0  # mV
    branch.el_hh = -54.3  # mV

    stim = h.IClamp(branch(0.05))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    voltage1 = h.Vector()
    voltage1.record(branch(0.05)._ref_v)
    voltage2 = h.Vector()
    voltage2.record(branch(0.95)._ref_v)

    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    voltages = np.asarray([list(voltage1), list(voltage2)])
    return voltages


@pytest.mark.parametrize("solver", ["bwd_euler", "crank_nicolson"])
def test_similarity_complex(solver):
    """Test for a branch where radius varies for every seg and l and r_a varies for
    two sub-branches.

    Since NEURON enforces that all segments within a branch have the same length and
    the same axial resistivity, we have to create two NEURON branches. In jaxley,
    we only need one branch.
    """
    i_delay = 2.0
    i_dur = 5.0
    i_amp = 0.1
    t_max = 20.0
    dt = 0.025

    diams = [
        15.925172886886058,
        15.299516921156213,
        14.662901345202302,
        13.471623855881704,
        12.095324558467158,
        9.602043499254922,
        8.155112133121278,
        6.48666134956395,
        3.873367784857285,
        2.729358541238842,
        2.14754416410492,
        2.064491286216302,
        1.6082726642214933,
        1.2703115584172973,
        0.9684275792140471,
        0.8000000119209283,
    ]
    capacitances = np.linspace(1.0, 10.0, 16)
    voltages_jaxley = _jaxley_complex(
        i_delay, i_dur, i_amp, dt, t_max, diams, capacitances, solver
    )
    voltages_neuron = _neuron_complex(
        i_delay, i_dur, i_amp, dt, t_max, diams, capacitances, solver
    )

    assert np.mean(np.abs(voltages_jaxley - voltages_neuron)) < 0.05


def _jaxley_complex(i_delay, i_dur, i_amp, dt, t_max, diams, capacitances, solver):
    ncomp = 16
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp)

    branch.insert(HH())

    branch.set("v", -62.0)
    branch.set("HH_m", 0.074901)
    branch.set("HH_h", 0.4889)
    branch.set("HH_n", 0.3644787)

    for loc in np.linspace(0, 0.45, 8):
        branch.loc(loc).set("length", 20.0)

    for loc in np.linspace(0.55, 1.0, 8):
        branch.loc(loc).set("length", 100.0)

    for loc in np.linspace(0, 0.45, 8):
        branch.loc(loc).set("axial_resistivity", 800.0)

    for loc in np.linspace(0.55, 1.0, 8):
        branch.loc(loc).set("axial_resistivity", 800.0)

    counter = 0
    for loc in np.linspace(0, 1, ncomp):
        branch.loc(loc).set("radius", diams[counter] / 2)
        branch.loc(loc).set("capacitance", capacitances[counter])
        counter += 1

    # 0.02 is fine here because ncomp=8 for NEURON, but ncomp=16 for jaxley.
    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    branch.loc(0.02).stimulate(current)
    branch.loc(0.02).record()
    branch.loc(0.52).record()
    branch.loc(0.98).record()

    s = jx.integrate(branch, delta_t=dt, solver=solver)
    return s


def _neuron_complex(i_delay, i_dur, i_amp, dt, t_max, diams, capacitances, solver):
    if solver == "bwd_euler":
        h.secondorder = 0
    elif solver == "crank_nicolson":
        h.secondorder = 1
    else:
        raise ValueError

    for sec in h.allsec():
        h.delete_section(sec=sec)

    branch1 = h.Section()
    branch2 = h.Section()

    branch2.connect(branch1, 1, 0)

    branch1.nseg = 8
    branch1.L = 160.0
    branch1.Ra = 800.0

    branch2.nseg = 8
    branch2.L = 800.0
    branch2.Ra = 800.0

    counter = 0
    for sec in h.allsec():
        sec.insert("hh")
        sec.gnabar_hh = 0.120  # S/cm2
        sec.gkbar_hh = 0.036  # S/cm2
        sec.gl_hh = 0.0003  # S/cm2
        sec.ena = 50  # mV
        sec.ek = -77.0  # mV
        sec.el_hh = -54.3  # mV

        for i, seg in enumerate(sec):
            seg.diam = diams[counter]
            seg.cm = capacitances[counter]
            counter += 1

    # 0.05 is fine here because ncomp=8, but ncomp=16 for jaxley.
    stim = h.IClamp(branch1(0.05))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    # 0.05 is fine here because ncomp=8, but ncomp=16 for jaxley.
    voltage_recs = {}
    v = h.Vector()
    v.record(branch1(0.05)._ref_v)
    voltage_recs["v0"] = v

    v = h.Vector()
    v.record(branch2(0.05)._ref_v)
    voltage_recs["v1"] = v

    v = h.Vector()
    v.record(branch2(0.95)._ref_v)
    voltage_recs["v2"] = v

    h.dt = dt
    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    return np.asarray([voltage_recs[key] for key in voltage_recs])


def _jaxley_clamped(dt, t_max, diams, v_init, length):
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)

    branch.comp(0).set("radius", diams[0] / 2)
    branch.comp(1).set("radius", diams[1] / 2)

    branch.set("axial_resistivity", 100_000.0)
    branch.set("length", length / 2)
    branch.set("v", v_init)
    branch.insert(CaHVA())
    branch.insert(CaFaradayConcentrationChange())
    branch.insert(CaNernstReversal())
    branch.comp(0).set("CaHVA_gCaHVA", 0.00001)
    branch.comp(1).set("CaHVA_gCaHVA", 0.0)
    branch.set("eCa", 0.0)
    branch.comp(0).clamp("v", jnp.linspace(-65.0, 0.0, 40_001))
    branch.init_states()

    branch.comp(0).record("v")
    branch.comp(1).record("v")

    branch.comp(0).record("CaCon_i")
    branch.comp(1).record("CaCon_i")

    voltages = jx.integrate(branch, t_max=t_max, delta_t=dt)
    return voltages


def _neuron_clamped(dt, t_max, diams, v_init, length):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "arm64/.libs/libnrnmech.so")
    h.nrn_load_dll(fname)

    h.secondorder = 0
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

    branch1 = h.Section()

    branch1.nseg = 2
    branch1.Ra = 100_000.0
    branch1.L = length
    branch1.insert("Ca_HVA")
    for i, seg in enumerate(branch1):
        if i == 0:
            seg.gCa_HVAbar_Ca_HVA = 0.00001
        else:
            seg.gCa_HVAbar_Ca_HVA = 0.0

    for i, seg in enumerate(branch1):
        seg.diam = diams[i]

    # Set diffusion to zero. This is causing the different interpretation of `.area()`
    # of NEURON and therefore causes this test to fail (also for d>0) if the radiuses
    # of compartments are not the same (as is the case here).
    r = rxd.Region(h.allsec(), nrn_region="i", geometry=None)
    ca = rxd.Species(r, d=0, name="ca", charge=2)

    # Voltage clamp.
    t = h.Vector(np.arange(0, 1000, dt))
    y = h.Vector(np.linspace(-65.0, 0.0, len(t)))
    y.play(branch1(0.3)._ref_v, t, True)

    v1 = h.Vector()
    v1.record(branch1(0.3)._ref_v)

    v2 = h.Vector()
    v2.record(branch1(0.7)._ref_v)

    cacon1 = h.Vector()
    cacon1.record(branch1(0.3)._ref_cai)

    cacon2 = h.Vector()
    cacon2.record(branch1(0.7)._ref_cai)

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    v1 = np.asarray(list(v1))
    v2 = np.asarray(list(v2))
    cacon1 = np.asarray(list(cacon1))
    cacon2 = np.asarray(list(cacon2))
    return np.stack([v1, v2, cacon1, cacon2])


@pytest.mark.xfail
def test_similarity_voltage_clamp_cacon_i_behavior():
    """Test whether two compartments were one is clamped gives same results.

    In this test, we clamp the voltage of one of the compartments. The rational is that
    this decouples any influence of _voltage_ diffusion on the calcium dynamics (in
    comp 0). This allowed us to study why the calcium dynamics between NEURON and
    Jaxley differ for comp(0) of a branch, even when the voltages are the same and
    diffusion is d=0.

    This test currently fails. The axial conductance is defined as:
    ```1 / R_long / area_of_sink_compartment```
    However, NEURON computes `area_of_sink_compartment` differently _depending on
    whether diffusion is activated or not_. This make no sense to me (@michaeldeistler),
    so I am not implementing it in Jaxley.

    Jaxley simply assumes a cylindrical compartment: `A = 2 * pi * r * l`
    Without diffusion, NEURON does the same. With diffusion, however, NEURON creates
    frustums: `A = 2 * pi * r * (l/2) + frustum_area`.
    The frustum radiuses are on one side, the radius of the compartment, and on the
    other side the radius at the connection point (which itself is the weighted
    average of all neighboring compartments). Because of this, the radius of `comp(1)`
    influences the `.area()` of `comp(0)`, and thus impacts its dynamics.

    Notably, NEURON even uses this updated area (which can be inspected with
    `branch(0.3).area()` _after having run the simulation with diffusion at least
    once_) even for the voltage equations. See also #140.
    """
    t_max = 1_000.01
    dt = 0.025

    diams = [4.0, 8.0]  # The test passes when diams has the same values.
    length = 100.0
    v_init = -65.0

    v_and_cacon_jaxley = _jaxley_clamped(dt, t_max, diams, v_init, length)
    v_and_cacon_neuron = _neuron_clamped(dt, t_max, diams, v_init, length)
    v_jaxley = v_and_cacon_jaxley[:2]
    cacon_jaxley = v_and_cacon_jaxley[2:]
    v_neuron = v_and_cacon_neuron[:2]
    cacon_neuron = v_and_cacon_neuron[2:]

    v_max_error = np.max(np.abs(v_jaxley - v_neuron))
    cacon_max_error = np.max(np.abs(cacon_jaxley - cacon_neuron))
    # Note: for very thin cables (<1um radius), this test can fail by a tiny bit, even
    # if both comps have the same diameter.
    assert cacon_max_error < 2e-6, f"CaCon_i error {cacon_max_error}"

    # To be on the safe side, we also check voltages. However, this test is generally
    # about CaCon_i.
    assert v_max_error < 1e-2, f"Voltage error {v_max_error}"
