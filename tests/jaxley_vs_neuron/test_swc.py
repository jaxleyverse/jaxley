# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
import numpy as np
import pytest
from neuron import h

import jaxley as jx
from jaxley.channels import HH

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


@pytest.mark.parametrize(
    "file",
    [
        "morph_ca1_n120_single_point_soma.swc",
        "morph_ca1_n120.swc",
        "morph_l5pc_with_axon.swc",
        "morph_allen_485574832.swc",
        "morph_variable_radiuses_within_branch.swc",
    ],
)
def test_swc_voltages(file, SimpleMorphCell):
    """Check if voltages of SWC recording match.

    To match the branch indices between NEURON and jaxley, we rely on comparing the
    length of the branches.

    It tests whether, on average over time, the voltage is off by less than 0.5 mV
    for every recording.
    """
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../swc_files", file)  # n120

    i_delay = 5.0
    i_dur = 20.0
    i_amp = 1.0
    t_max = 30.0
    dt = 0.025

    stim_loc = 0.51
    loc = 0.51

    if file == "morph_variable_radiuses_within_branch.swc":
        ncomp_per_branch = 1
    else:
        ncomp_per_branch = 3

    ##################### NEURON ##################
    h.secondorder = 0

    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    for sec in h.allsec():
        sec.nseg = ncomp_per_branch

    pathlengths_neuron = np.asarray([sec.L for sec in h.allsec()])

    ####################### jaxley ##################
    cell = SimpleMorphCell(
        fname,
        ncomp_per_branch,
        max_branch_len=2_000.0,
        ignore_swc_tracing_interruptions=False,
    )

    pathlengths = []
    for branch in cell.branches:
        pathlengths.append(branch.nodes["length"].sum())
    pathlengths_jaxley = np.asarray(pathlengths)

    cell.insert(HH())

    jaxley_inds = [0, 1, 2, 5, 10, 16, 20, 40, 60, 80]
    jaxley_inds = [ind for ind in jaxley_inds if ind < len(pathlengths_jaxley)]

    neuron_inds = []
    for jaxley_ind in jaxley_inds:
        for i, p in enumerate(pathlengths_jaxley):
            if i == jaxley_ind:
                closest_match = np.argmin(np.abs(pathlengths_neuron - p))
                neuron_inds.append(closest_match)

    cell.set("axial_resistivity", 100.0)
    cell.set("v", -62.0)
    cell.set("HH_m", 0.074901)
    cell.set("HH_h", 0.4889)
    cell.set("HH_n", 0.3644787)
    cell.soma.branch(0).loc(stim_loc).stimulate(
        jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    )
    for i in jaxley_inds:
        cell.branch(i).loc(loc).record(verbose=False)

    voltages_jaxley = jx.integrate(cell, delta_t=dt, voltage_solver="jaxley.dhs.cpu")

    ################### NEURON #################
    stim = h.IClamp(h.soma[0](stim_loc))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    counter = 0
    voltage_recs = {}

    for r in neuron_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(loc)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for sec in h.allsec():
        sec.insert("hh")
        sec.Ra = 100.0

        sec.gnabar_hh = 0.120  # S/cm2
        sec.gkbar_hh = 0.036  # S/cm2
        sec.gl_hh = 0.0003  # S/cm2
        sec.ena = 50  # mV
        sec.ek = -77.0  # mV
        sec.el_hh = -54.3  # mV

    h.dt = dt
    tstop = t_max
    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < tstop:
            h.fadvance()

    initialize()
    integrate()
    voltages_neuron = np.asarray([voltage_recs[key] for key in voltage_recs])
    errors = np.mean(np.abs(voltages_jaxley - voltages_neuron), axis=1)

    ###################### check ################
    assert all(errors < 0.5), f"Error {np.max(errors)} > 0.5. Voltages do not match."
