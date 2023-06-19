from jax.config import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import neurax as nx
import numpy as np

from neuron import h

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
    nseg = 16
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
