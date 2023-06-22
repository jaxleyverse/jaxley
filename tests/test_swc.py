from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import numpy as np
from neuron import h

import neurax as nx
from neurax.channels import HHChannel

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


def test_swc_reader_lengths():
    fname = "../../notebooks/morph.swc"

    _, pathlengths, _, _ = nx.utils.read_swc(fname, max_branch_len=2000.0)
    pathlengths = np.asarray(pathlengths)

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

    for i, p in enumerate(pathlengths):
        # For index two, there is some weird behaviour of NEURON. If I exclude the
        # first traced point from the given branch in neurax, then I can exactly
        # reproduce NEURON, but it is unclear to me why I should do that.
        if i != 2:
            dists = np.abs(neuron_pathlengths - p)
            assert np.min(dists) < 1e-3, "Some branches have too large distance."

    assert len(pathlengths) == len(
        neuron_pathlengths
    ), "Number of branches does not match."


def test_swc_radius():
    """We expect them to match for sufficiently large nseg. See #140."""
    nseg = 32
    non_split = 1 / nseg
    range_16 = np.linspace(non_split / 2, 1 - non_split / 2, nseg)

    # Can not use full morphology because of branch sorting.
    fname = "../../notebooks/morph_250.swc"

    _, _, radius_fns, _ = nx.utils.read_swc(fname, max_branch_len=2000.0, sort=False)
    neurax_diams = []
    for r in radius_fns:
        neurax_diams.append(r(range_16) * 2)

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

    assert np.all(np.abs(neurax_diams - neuron_diams) < 0.5), "radiuses do not match."


def test_swc_voltages():
    """Check if voltages of SWC recording match.

    To match the branch indices between NEURON and neurax, we rely on comparing the
    length of the branches.
    """

    fname = "../../notebooks/morph.swc"  # n120

    i_delay = 2.0
    i_dur = 5.0
    i_amp = 0.25
    t_max = 20.0
    dt = 0.025

    time_vec = np.arange(0, t_max + dt, dt)

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

    ####################### neurax ##################
    parents, pathlengths, radius_fns, _ = nx.utils.read_swc(fname, max_branch_len=2_000)
    nbranches = len(parents)

    nseg = 8
    non_split = 1 / nseg
    range_ = np.linspace(non_split / 2, 1 - non_split / 2, nseg)

    comp = nx.Compartment().initialize()
    branch = nx.Branch([comp for _ in range(nseg)]).initialize()
    cell = nx.Cell([branch for _ in range(nbranches)], parents=parents)
    cell.insert(HHChannel())

    for i, b in enumerate(range(len(parents))):
        l = pathlengths[i]
        radiuses = radius_fns[i](range_)
        for i, loc in enumerate(np.linspace(0, 1, nseg)):
            cell.branch(b).comp(loc).set_params("length", l / nseg)
            cell.branch(b).comp(loc).set_params("radius", radiuses[i])

    cell = cell.initialize()

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

    cell.set_params("axial_resistivity", 1_000.0)
    cell.set_states("voltages", -62.0)
    cell.set_states("m", 0.074901)
    cell.set_states("h", 0.4889)
    cell.set_states("n", 0.3644787)

    stims = [nx.Stimulus(0, 1, 0.05, nx.step_current(i_delay, i_dur, i_amp, time_vec))]
    recs = [nx.Recording(0, i, 0.05) for i in trunk_inds + tuft_inds + basal_inds]

    voltages_neurax = nx.integrate(cell, stims, recs, delta_t=dt)

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
    voltages_neuron = np.asarray([v[key] for key in voltage_recs])

    ####################### check ################
    assert np.all(
        np.abs(voltages_neurax - voltages_neuron) < 1.0
    ), "voltages do not match."
