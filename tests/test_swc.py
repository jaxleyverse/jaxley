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


# Test is failing for "morph.swc". This is because NEURON and Jaxley handle interrupted
# soma differently, see issue #140.
@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph_minimal.swc"])
def test_swc_reader_lengths(file):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    _, pathlengths, _, _, _ = jx.utils.swc.swc_to_jaxley(fname, max_branch_len=2000.0)
    if pathlengths[0] == 0.1:
        pathlengths = pathlengths[1:]

    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    neuron_pathlengths = []
    for sec in h.allsec():
        neuron_pathlengths.append(sec.L)
    neuron_pathlengths = np.asarray(neuron_pathlengths)

    for p in pathlengths:
        dists = np.abs(neuron_pathlengths - p)
        assert np.min(dists) < 1e-3, "Some branches have too large distance."

    assert len(pathlengths) == len(
        neuron_pathlengths
    ), "Number of branches does not match."


def test_dummy_compartment_length():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_soma_both_ends.swc")

    parents, pathlengths, _, _, _ = jx.utils.swc.swc_to_jaxley(
        fname, max_branch_len=2000.0
    )
    assert parents == [-1, 0, 0, 1]
    assert pathlengths == [0.1, 1.0, 2.6, 2.2]


@pytest.mark.parametrize("file", ["morph_250_single_point_soma.swc", "morph_250.swc"])
def test_swc_radius(file):
    """We expect them to match for sufficiently large nseg. See #140."""
    nseg = 64
    non_split = 1 / nseg
    range_16 = np.linspace(non_split / 2, 1 - non_split / 2, nseg)

    # Can not use full morphology because of branch sorting.
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    _, pathlen, radius_fns, _, _ = jx.utils.swc.swc_to_jaxley(
        fname, max_branch_len=2000.0, sort=False
    )
    jaxley_diams = []
    for r in radius_fns:
        jaxley_diams.append(r(range_16) * 2)

    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    neuron_diams = []
    for sec in h.allsec():
        sec.nseg = nseg
        diams_in_branch = []
        for seg in sec:
            diams_in_branch.append(seg.diam)
        neuron_diams.append(diams_in_branch)
    neuron_diams = np.asarray(neuron_diams)

    for i in range(len(jaxley_diams)):
        max_error = np.max(np.abs(jaxley_diams[i] - neuron_diams[i]))
        assert max_error < 0.5, f"radiuses do not match, error {max_error}."


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_swc_voltages(file):
    """Check if voltages of SWC recording match.

    To match the branch indices between NEURON and jaxley, we rely on comparing the
    length of the branches.

    It tests whether, on average over time and recordings, the voltage is off by less
    than 1.5 mV.
    """
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)  # n120

    i_delay = 2.0
    i_dur = 5.0
    i_amp = 0.25
    t_max = 20.0
    dt = 0.025

    nseg_per_branch = 8

    ##################### NEURON ##################
    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    for sec in h.allsec():
        sec.nseg = nseg_per_branch

    pathlengths_neuron = np.asarray([sec.L for sec in h.allsec()])

    ####################### jaxley ##################
    _, pathlengths, _, _, _ = jx.utils.swc.swc_to_jaxley(fname, max_branch_len=2_000)
    cell = jx.read_swc(fname, nseg_per_branch, max_branch_len=2_000.0)
    cell.insert(HH())

    trunk_inds = [1, 4, 5, 13, 15, 21, 23, 24, 29, 33]
    tuft_inds = [6, 16, 18, 36, 38, 44, 51, 52, 53, 54]
    basal_inds = np.arange(81, 156, 8).tolist()

    neuron_trunk_inds = []
    for i, p in enumerate(pathlengths):
        if i in trunk_inds:
            closest_match = np.argmin(np.abs(pathlengths_neuron - p))
            neuron_trunk_inds.append(closest_match)

    neuron_tuft_inds = []
    for i, p in enumerate(pathlengths):
        if i in tuft_inds:
            closest_match = np.argmin(np.abs(pathlengths_neuron - p))
            neuron_tuft_inds.append(closest_match)

    neuron_basal_inds = []
    for i, p in enumerate(pathlengths):
        if i in basal_inds:
            closest_match = np.argmin(np.abs(pathlengths_neuron - p))
            neuron_basal_inds.append(closest_match)

    cell.set("axial_resistivity", 1_000.0)
    cell.set("v", -62.0)
    cell.set("HH_m", 0.074901)
    cell.set("HH_h", 0.4889)
    cell.set("HH_n", 0.3644787)

    cell.branch(1).loc(0.05).stimulate(
        jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    )
    for i in trunk_inds + tuft_inds + basal_inds:
        cell.branch(i).loc(0.05).record()

    voltages_jaxley = jx.integrate(cell, delta_t=dt)

    ################### NEURON #################
    stim = h.IClamp(h.soma[0](0.1))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    counter = 0
    voltage_recs = {}

    for r in neuron_trunk_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(0.05)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for r in neuron_tuft_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(0.05)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for r in neuron_basal_inds:
        for i, sec in enumerate(h.allsec()):
            if i == r:
                v = h.Vector()
                v.record(sec(0.05)._ref_v)
                voltage_recs[f"v{counter}"] = v
                counter += 1

    for sec in h.allsec():
        sec.insert("hh")
        sec.Ra = 1_000.0

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

    ####################### check ################
    assert np.mean(
        np.abs(voltages_jaxley - voltages_neuron) < 1.5
    ), "voltages do not match."
