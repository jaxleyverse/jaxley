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
def test_swc_reader_lengths(file, swc2jaxley):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    _, pathlengths, _, _, _ = swc2jaxley(fname, max_branch_len=2000.0)
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


def test_dummy_compartment_length(swc2jaxley):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_soma_both_ends.swc")

    parents, pathlengths, _, _, _ = swc2jaxley(fname, max_branch_len=2000.0)
    assert parents == [-1, 0, 0, 1]
    assert pathlengths == [0.1, 1.0, 2.6, 2.2]


@pytest.mark.parametrize("file", ["morph_250_single_point_soma.swc", "morph_250.swc"])
def test_swc_radius(file, swc2jaxley):
    """We expect them to match for sufficiently large ncomp. See #140."""
    ncomp = 64
    non_split = 1 / ncomp
    range_16 = np.linspace(non_split / 2, 1 - non_split / 2, ncomp)

    # Can not use full morphology because of branch sorting.
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    _, pathlen, radius_fns, _, _ = swc2jaxley(fname, max_branch_len=2000.0, sort=False)
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
        sec.nseg = ncomp
        diams_in_branch = []
        for seg in sec:
            diams_in_branch.append(seg.diam)
        neuron_diams.append(diams_in_branch)
    neuron_diams = np.asarray(neuron_diams)

    for i in range(len(jaxley_diams)):
        max_error = np.max(np.abs(jaxley_diams[i] - neuron_diams[i]))
        assert max_error < 0.5, f"radiuses do not match, error {max_error}."


@pytest.mark.parametrize("file", ["morph_single_point_soma.swc", "morph.swc"])
def test_swc_voltages(file, SimpleMorphCell, swc2jaxley):
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

    ncomp_per_branch = 8

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
    _, pathlengths, _, _, _ = swc2jaxley(fname, max_branch_len=2_000)
    cell = SimpleMorphCell(fname, ncomp_per_branch, max_branch_len=2_000.0)
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

    voltages_jaxley = jx.integrate(cell, delta_t=dt, voltage_solver="jax.sparse")

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
    errors = np.mean(np.abs(voltages_jaxley - voltages_neuron), axis=1)

    ####################### check ################
    assert all(errors < 2.5), "voltages do not match."


@pytest.mark.parametrize(
    "reader_backend",
    [
        "graph",
        "custom",
    ],
)
@pytest.mark.parametrize(
    "file",
    [
        "morph_3_types.swc",
        "morph_3_types_single_point_soma.swc",
        "morph.swc",
        "bbp_with_axon.swc",
    ],
)
def test_swc_types(reader_backend, file):
    # Can not use full morphology because of branch sorting.
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)
    backend_kwargs = (
        {"ignore_swc_trace_errors": False} if reader_backend == "graph" else {}
    )
    cell = jx.read_swc(fname, ncomp=1, backend=reader_backend, **backend_kwargs)

    # First iteration is with default `ncomp`. At the end of the first loop we change
    # ncomp with `set_ncomp`
    for i in range(2):
        desired_numbers_of_comps = {
            "morph_3_types.swc": {"soma": 1, "axon": 1, "basal": 1},
            "morph_3_types_single_point_soma.swc": {
                "soma": 1,
                "axon": 1,
                "basal": 1,
            },
            "morph.swc": {"soma": 2, "basal": 101, "apical": 53},
            "bbp_with_axon.swc": {"soma": 1, "axon": 128, "basal": 66, "apical": 129},
        }
        # Test soma.
        for key, n_desired in desired_numbers_of_comps[file].items():
            if i == 1 and key in ["soma", "basal"]:
                n_desired += 2  # After `set_ncomp` we should have two more comps.
            n_comps_in_morph = len(cell.__getattr__(key).nodes)
            assert (
                n_comps_in_morph == n_desired
            ), f"{key} has {n_comps_in_morph} != {n_desired} comps!"

        # Additional tests to ensure that `groups` get updated appropriately.
        cell.soma.branch(0).set_ncomp(3)
        cell.basal.branch(0).set_ncomp(3)
