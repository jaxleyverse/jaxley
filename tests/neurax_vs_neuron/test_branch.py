from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"

import numpy as np

import jax.numpy as jnp

import neurax as nx
from neurax.channels import HHChannel

from neuron import h

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


def test_similarity():
    """Test similarity of neurax vs neuron for a branch.

    The branch has an uneven radius.
    """
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
    branch.insert(HHChannel())

    radiuses = np.linspace(3.0, 15.0, nseg_per_branch)
    for i, loc in enumerate(np.linspace(0, 1, nseg_per_branch)):
        branch.comp(loc).set_params("radius", radiuses[i])

    branch.set_params("length", 10.0)
    branch.set_params("axial_resistivity", 1_000.0)

    branch.set_params("gNa", 0.120)
    branch.set_params("gK", 0.036)
    branch.set_params("gLeak", 0.0003)
    branch.set_states("m", 0.07490098835688629)
    branch.set_states("h", 0.488947681848153)
    branch.set_states("n", 0.3644787002343737)
    branch.set_states("voltages", -62.0)

    stims = [nx.Stimulus(0, 0, 0.0, nx.step_current(i_delay, i_dur, i_amp, time_vec))]
    recs = [nx.Recording(0, 0, 0.0), nx.Recording(0, 0, 1.0)]

    voltages = nx.integrate(branch, stims, recs, delta_t=dt)

    return voltages


def _run_neurax(i_delay, i_dur, i_amp, dt, t_max):
    nseg_per_branch = 8
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

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

    stim = h.IClamp(branch(0.0))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    voltage1 = h.Vector()
    voltage1.record(branch(0.0)._ref_v)
    voltage2 = h.Vector()
    voltage2.record(branch(1.0)._ref_v)

    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    voltages = np.asarray([list(voltage1, list(voltage2))])
    return voltages
