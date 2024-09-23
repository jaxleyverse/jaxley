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
    nseg_per_branch = 8
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
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

    cell.branch(0).loc(0.0).stimulate(jx.step_current(i_delay, i_dur, i_amp, dt, t_max))
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

    nseg_per_branch = 8
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

    branch1 = h.Section()
    branch2 = h.Section()
    branch3 = h.Section()

    branch2.connect(branch1, 1, 0)
    branch3.connect(branch1, 1, 0)

    for sec in h.allsec():
        sec.nseg = nseg_per_branch

        sec.Ra = 1_000.0
        sec.L = 10.0 * nseg_per_branch
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
    branch1 = jx.Branch(comp, nseg=1)
    branch2 = jx.Branch(comp, nseg=2)
    branch3 = jx.Branch(comp, nseg=3)
    branch4 = jx.Branch(comp, nseg=4)
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

    voltages = jx.integrate(cell, delta_t=dt, voltage_solver="jax.sparse")
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

    nsegs = [1, 2, 3, 4]
    for i, sec in enumerate(h.allsec()):
        sec.nseg = nsegs[i]

        sec.Ra = 1_000.0
        sec.L = 20.0 * nsegs[i]
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
