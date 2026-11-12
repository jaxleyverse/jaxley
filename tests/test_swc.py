# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import os

import numpy as np
import pytest
from neuron import h

import jaxley as jx

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


# Test is failing for "morph_ca1_n120.swc". This is because NEURON and Jaxley handle
# interrupted soma differently, see issue #140.
@pytest.mark.parametrize(
    "file", ["morph_ca1_n120_single_point_soma.swc", "morph_minimal.swc"]
)
def test_swc_reader_lengths(file):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    cell = jx.read_swc(fname, ncomp=1)
    pathlengths = cell.nodes.length.to_numpy()
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

    cell = jx.read_swc(fname, ncomp=1)
    pathlengths = cell.nodes.length.to_numpy().tolist()
    parents = cell.comb_parents.tolist()
    assert parents == [-1, 0, 1]
    assert pathlengths == [2.2, 1.0, 2.6]


@pytest.mark.parametrize(
    "file", ["morph_ca1_n120_250_single_point_soma.swc", "morph_ca1_n120_250.swc"]
)
def test_swc_radius(file):
    """We expect them to match for sufficiently large ncomp. See #140."""
    ncomp = 64

    # Can not use full morphology because of branch sorting.
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    # _, _, radius_fns, _, _ = swc2jaxley(fname, max_branch_len=2000.0, sort=False)
    cell = jx.read_swc(fname, ncomp=ncomp)
    jaxley_diams = cell.nodes.radius.to_numpy() * 2
    jaxley_diams = np.sort(jaxley_diams)

    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    neuron_diams = []
    for sec in h.allsec():
        sec.nseg = ncomp
        for seg in sec:
            neuron_diams.append(seg.diam)
    neuron_diams = np.asarray(neuron_diams)
    neuron_diams = np.sort(neuron_diams)

    max_error = np.max(np.abs(jaxley_diams - neuron_diams))
    assert max_error < 0.5, f"radiuses do not match, error {max_error}."


@pytest.mark.parametrize(
    "file, ncomps",
    [
        ("morph_3_types.swc", {"soma": 1, "axon": 1, "basal": 1}),
        ("morph_3_types_single_point_soma.swc", {"soma": 1, "axon": 1, "basal": 1}),
    ],
)
def test_swc_types(file, ncomps):
    """SWC types become groups with the right number of compartments (one per branch
    with `ncomp=1`), and `set_ncomp` updates the groups."""
    fname = os.path.join(os.path.dirname(__file__), "swc_files", file)
    cell = jx.read_swc(fname, ncomp=1, ignore_swc_tracing_interruptions=False)
    assert {key: len(getattr(cell, key).nodes) for key in ncomps} == ncomps

    cell.soma.branch(0).set_ncomp(3)
    cell.basal.branch(0).set_ncomp(3)
    ncomps = {k: n + 2 if k in ["soma", "basal"] else n for k, n in ncomps.items()}
    assert {key: len(getattr(cell, key).nodes) for key in ncomps} == ncomps


def test_single_branch_swc():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_single_branch.swc")
    cell = jx.read_swc(fname, ncomp=1)
    cell.branch(0).set_ncomp(3)
    cell.set_ncomp(4)
    cell[0, 0].record()
    v = jx.integrate(cell, t_max=1.0)
    assert np.invert(np.any(np.isnan(v))), "Found a NaN."
