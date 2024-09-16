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
