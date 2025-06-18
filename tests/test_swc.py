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

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


# Test is failing for "morph_ca1_n120.swc". This is because NEURON and Jaxley handle
# interrupted soma differently, see issue #140.
@pytest.mark.parametrize(
    "file", ["morph_ca1_n120_single_point_soma.swc", "morph_minimal.swc"]
)
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


@pytest.mark.parametrize(
    "file", ["morph_ca1_n120_250_single_point_soma.swc", "morph_ca1_n120_250.swc"]
)
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


@pytest.mark.parametrize("reader_backend", ["graph"])
@pytest.mark.parametrize(
    "file",
    [
        "morph_3_types.swc",
        "morph_3_types_single_point_soma.swc",
        "morph_ca1_n120.swc",
        "morph_l5pc_with_axon.swc",
    ],
)
def test_swc_types(reader_backend, file):
    # Can not use full morphology because of branch sorting.
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)
    backend_kwargs = (
        {"ignore_swc_tracing_interruptions": False} if reader_backend == "graph" else {}
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
            "morph_ca1_n120.swc": {"soma": 2, "basal": 101, "apical": 53},
            "morph_l5pc_with_axon.swc": {
                "soma": 1,
                "axon": 128,
                "basal": 66,
                "apical": 129,
            },
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
