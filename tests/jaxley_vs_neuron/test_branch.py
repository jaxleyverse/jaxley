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
from neuron import h

import jaxley as jx
from jaxley.channels import HH

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
    nseg_per_branch = 8
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)])
    branch.insert(HH())

    radiuses = np.linspace(3.0, 15.0, nseg_per_branch)
    for i, loc in enumerate(np.linspace(0, 1, nseg_per_branch)):
        branch.loc(loc).set("radius", radiuses[i])

    branch.set("length", 10.0)
    branch.set("axial_resistivity", 1_000.0)

    branch.set("HH_gNa", 0.120)
    branch.set("HH_gK", 0.036)
    branch.set("HH_gLeak", 0.0003)
    branch.set("HH_m", 0.07490098835688629)
    branch.set("HH_h", 0.488947681848153)
    branch.set("HH_n", 0.3644787002343737)
    branch.set("v", -62.0)

    branch.loc(0.0).stimulate(jx.step_current(i_delay, i_dur, i_amp, dt, t_max))
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

    nseg_per_branch = 8
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)
    branch = h.Section()

    branch.nseg = nseg_per_branch
    branch.Ra = 1_000.0
    branch.L = 10.0 * nseg_per_branch

    radiuses = np.linspace(3.0, 15.0, nseg_per_branch)
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
    voltages_jaxley = _jaxley_complex(i_delay, i_dur, i_amp, dt, t_max, diams, solver)
    voltages_neuron = _neuron_complex(i_delay, i_dur, i_amp, dt, t_max, diams, solver)

    assert np.mean(np.abs(voltages_jaxley - voltages_neuron)) < 0.05


def _jaxley_complex(i_delay, i_dur, i_amp, dt, t_max, diams, solver):
    nseg = 16
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg)

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
    for loc in np.linspace(0, 1, nseg):
        branch.loc(loc).set("radius", diams[counter] / 2)
        counter += 1

    branch = branch

    # 0.02 is fine here because nseg=8 for NEURON, but nseg=16 for jaxley.
    branch.loc(0.02).stimulate(jx.step_current(i_delay, i_dur, i_amp, dt, t_max))
    branch.loc(0.02).record()
    branch.loc(0.52).record()
    branch.loc(0.98).record()

    s = jx.integrate(branch, delta_t=dt, solver=solver)
    return s


def _neuron_complex(i_delay, i_dur, i_amp, dt, t_max, diams, solver):
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
            counter += 1

    # 0.05 is fine here because nseg=8, but nseg=16 for jaxley.
    stim = h.IClamp(branch1(0.05))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    # 0.05 is fine here because nseg=8, but nseg=16 for jaxley.
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
