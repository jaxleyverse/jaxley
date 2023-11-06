import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"

import jax.numpy as jnp
import numpy as np
from neuron import h

import neurax as nx
from neurax.channels import HHChannel

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


def test_similarity():
    """Test similarity of neurax vs neuron."""
    i_delay = 3.0  # ms
    i_dur = 2.0  # ms
    i_amp = 1.0  # nA

    dt = 0.025  # ms
    t_max = 10.0  # ms

    voltages_neurax = _run_neurax(i_delay, i_dur, i_amp, dt, t_max)
    voltages_neuron = _run_neurax(i_delay, i_dur, i_amp, dt, t_max)

    assert np.mean(np.abs(voltages_neurax - voltages_neuron)) < 1.0


def _run_neurax(i_delay, i_dur, i_amp, dt, t_max):
    time_vec = jnp.arange(0.0, t_max + dt, dt)

    nseg_per_branch = 8
    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = nx.Cell([branch for _ in range(3)], parents=[-1, 0, 0]).initialize()
    cell.insert(HHChannel())

    cell.set_params("radius", 5.0)
    cell.set_params("length", 10.0)
    cell.set_params("axial_resistivity", 1_000.0)

    cell.set_params("gNa", 0.120)
    cell.set_params("gK", 0.036)
    cell.set_params("gLeak", 0.0003)
    cell.set_states("m", 0.07490098835688629)
    cell.set_states("h", 0.488947681848153)
    cell.set_states("n", 0.3644787002343737)
    cell.set_states("voltages", -62.0)

    cell.branch(0).comp(0.0).stimulate(nx.step_current(i_delay, i_dur, i_amp, time_vec))
    cell.branch(0).comp(0.0).record()
    cell.branch(1).comp(1.0).record()
    cell.branch(2).comp(1.0).record()

    voltages = nx.integrate(cell, delta_t=dt)
    return voltages


def _run_neurax(i_delay, i_dur, i_amp, dt, t_max):
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
