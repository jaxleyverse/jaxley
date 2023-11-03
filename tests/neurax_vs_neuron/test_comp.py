from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


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
    voltages_neuron = _run_neuron(i_delay, i_dur, i_amp, dt, t_max)

    assert np.mean(np.abs(voltages_neurax - voltages_neuron)) < 1.0


def _run_neurax(i_delay, i_dur, i_amp, dt, t_max):
    time_vec = jnp.arange(0.0, t_max + dt, dt)

    comp = nx.Compartment().initialize()
    comp.insert(HHChannel())

    comp.set_params("length", 10.0)
    comp.set_params("radius", 10.0)

    comp.set_params("gNa", 0.120)
    comp.set_params("gK", 0.036)
    comp.set_params("gLeak", 0.0003)
    comp.set_states("m", 0.07490098835688629)
    comp.set_states("h", 0.488947681848153)
    comp.set_states("n", 0.3644787002343737)
    comp.set_states("voltages", -62.0)

    comp.stimulate(nx.step_current(i_delay, i_dur, i_amp, time_vec))
    comp.record()

    voltages = nx.integrate(comp, delta_t=dt)
    return voltages


def _run_neuron(i_delay, i_dur, i_amp, dt, t_max):
    h.dt = dt

    for sec in h.allsec():
        h.delete_section(sec=sec)

    comp = h.Section()

    comp.nseg = 1
    comp.Ra = 1_000.0
    comp.L = 10.0
    comp.diam = 20.0

    comp.insert("hh")
    comp.gnabar_hh = 0.120  # S/cm2
    comp.gkbar_hh = 0.036  # S/cm2
    comp.gl_hh = 0.0003  # S/cm2
    comp.ena = 50  # mV
    comp.ek = -77.0  # mV
    comp.el_hh = -54.3  # mV

    stim = h.IClamp(comp(0.0))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    voltage1 = h.Vector()
    voltage1.record(comp(0.0)._ref_v)

    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < t_max:
            h.fadvance()

    initialize()
    integrate()

    voltages = np.asarray([list(voltage1)])
    return voltages
