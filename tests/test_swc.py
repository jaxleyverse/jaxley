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
from scipy.spatial.distance import cdist

import jaxley as jx

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


# The SWC reader backends under test. Delete `"legacy"` here to drop it everywhere.
BACKENDS = ["graph", "neuron", "legacy"]

MORPH_FILES = [
    "morph_3_types.swc",
    "morph_3_types_single_point_soma.swc",
    "morph_soma_both_ends.swc",
    "morph_minimal.swc",
    "morph_ca1_n120_single_point_soma.swc",
    "morph_ca1_n120.swc",
    "morph_l5pc_with_axon.swc",
    "morph_allen_485574832.swc",
]


def _read_swc(fname, ncomp, backend):
    """`jx.read_swc()` with the kwargs that make the backends comparable to NEURON."""
    return jx.read_swc(
        fname, ncomp=ncomp, backend=backend, ignore_swc_tracing_interruptions=False
    )


def _neuron_morph_params(fname, ncomp):
    """Per-segment length, radius, area and volume, straight from NEURON.

    Must be called *after* any `jx.read_swc(..., backend="neuron")`: that backend rebuilds
    `h.allsec()` from the file and would invalidate a model built here beforehand.
    """
    for sec in h.allsec():
        h.delete_section(sec=sec)
    morph = h.Import3d_SWC_read()
    morph.input(fname)
    h.Import3d_GUI(morph, False).instantiate(None)
    for sec in h.allsec():
        sec.nseg = ncomp

    return {
        "length": np.array([sec.L / sec.nseg for sec in h.allsec() for _ in sec]),
        "radius": np.array([seg.diam / 2 for sec in h.allsec() for seg in sec]),
        "area": np.array([seg.area() for sec in h.allsec() for seg in sec]),
        "volume": np.array([seg.volume() for sec in h.allsec() for seg in sec]),
    }


# All backends are checked against NEURON on all four attributes with a tight relative
# tolerance. `KNOWN_DIVERGENT` lists the (backend, file, attribute) combinations that do
# not meet it, with the measured bound and the reason. Each one is a real, understood
# disagreement, and keeping the measured value means further drift still fails.
SINGLE_POINT_SOMA = [
    "morph_3_types_single_point_soma.swc",
    "morph_ca1_n120_single_point_soma.swc",
    "morph_allen_485574832.swc",
]
INTERRUPTED_SOMA = {"morph_minimal.swc": 5.0, "morph_ca1_n120.swc": 0.2}

KNOWN_DIVERGENT = {
    # A single-point soma gets the sphere volume 4/3 pi r^3, where NEURON builds a
    # cylinder of length 2r and reports 2 pi r^3: a factor 1.5, i.e. 1/3 relative.
    **{("legacy", f, "volume"): 0.34 for f in SINGLE_POINT_SOMA},
    # An interrupted soma gets different branchpoint radii, which propagates into the
    # area and the volume.
    **{
        ("legacy", f, key): bound
        for f, bound in INTERRUPTED_SOMA.items()
        for key in ["radius", "area", "volume"]
    },
    # Coincident traced points with a radius step: the frustum area counts the annular
    # face twice. Fixed for the graph backend in `compute_cone_props()`; `legacy` keeps
    # its own copy of that math and is slated for removal.
    ("legacy", "morph_l5pc_with_axon.swc", "area"): 0.15,
}
RTOL = 1e-4
RTOL_NEURON = 1e-12  # reads NEURON's own values, so it has to be exact


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("file", MORPH_FILES)
@pytest.mark.parametrize("ncomp", [1, 3])
def test_swc_morph_params_vs_neuron(backend, file, ncomp):
    """Every backend must reproduce NEURON's length, radius, area and volume."""
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    cell = _read_swc(fname, ncomp, backend)  # first: rebuilds h.allsec() for "neuron"
    neuron = _neuron_morph_params(fname, ncomp)

    for key in ["length", "radius", "area", "volume"]:
        jaxley = np.sort(cell.nodes[key].to_numpy().astype(float))
        reference = np.sort(neuron[key])
        assert len(jaxley) == len(
            reference
        ), f"{backend}: {len(jaxley)} comps != {len(reference)} NEURON segments."
        error = np.max(np.abs(jaxley - reference) / np.maximum(reference, 1e-30))
        rtol = KNOWN_DIVERGENT.get(
            (backend, file, key), RTOL_NEURON if backend == "neuron" else RTOL
        )
        assert error < rtol, f"{backend}: max rel. |{key} - NEURON| = {error} > {rtol}."


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            b,
            marks=(
                pytest.mark.xfail(
                    reason="The neuron backend takes its root from NEURON, which roots "
                    "at the soma, so `root` cannot select the chain rooting."
                )
                if b == "neuron"
                else ()
            ),
        )
        for b in BACKENDS
    ],
)
def test_dummy_compartment_length(backend):
    """No zero-length dummy compartment is inserted for a soma with dendrites at both ends.

    `morph_soma_both_ends.swc` traces a two-point soma with one dendrite leaving each end,
    so its three branches form a *path*: dendrite(2.6) - soma(1.0) - dendrite(2.2). A path
    has no unique hierarchy, so the parent list depends on where the morphology is rooted:
    rooting at the soma gives `[-1, 0, 0]` (both dendrites are children of the soma, which
    is what NEURON does: `soma[0] L=1.0 parent=None`), rooting at a dendrite tip gives the
    chain `[-1, 0, 1]`. Both describe the same morphology.

    This test pins the chain, so it passes `root` explicitly (node 4 is the tip of the
    2.2 um dendrite). The neuron backend cannot do that, since it reads the root from
    NEURON, hence the xfail.
    """
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_soma_both_ends.swc")

    cell = jx.read_swc(fname, ncomp=1, backend=backend, root=4)
    assert cell.comb_parents.tolist() == [-1, 0, 1]
    assert cell.nodes.length.to_numpy().tolist() == [2.2, 1.0, 2.6]


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "neuron"])
@pytest.mark.parametrize("file", MORPH_FILES)
@pytest.mark.parametrize("ncomp", [1, 3])
def test_comp_centers_agree_across_backends(backend, file, ncomp):
    """Every backend must place its compartment centers at the same coordinates.

    Compared against the neuron backend, which reads NEURON's own geometry. Node indices
    are not comparable between backends (they number branches differently), so each
    compartment is matched to its nearest counterpart. Requiring the match to be a
    bijection is what rules out a backend collapsing several compartments onto one point.
    """
    if backend == "legacy" and file in SINGLE_POINT_SOMA and ncomp > 1:
        pytest.xfail(
            "legacy collapses every single-point-soma compartment onto the traced point. "
            "It keeps its own copy of the compartmentalization and is slated for removal."
        )

    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    cell = _read_swc(fname, ncomp, backend)
    reference = _read_swc(fname, ncomp, "neuron")

    xyz = cell.nodes[["x", "y", "z"]].to_numpy(dtype=float)
    ref_xyz = reference.nodes[["x", "y", "z"]].to_numpy(dtype=float)
    assert len(xyz) == len(ref_xyz), f"{backend}: {len(xyz)} comps != {len(ref_xyz)}."

    dists = cdist(xyz, ref_xyz)
    nearest = dists.argmin(axis=1)
    worst = dists[np.arange(len(xyz)), nearest].max()
    assert worst < 1e-3, f"{backend}: comp centers off by up to {worst} um."
    assert len(set(nearest)) == len(xyz), f"{backend}: comp centers are not one-to-one."


@pytest.mark.parametrize("reader_backend", BACKENDS)
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
    cell = _read_swc(fname, 1, reader_backend)

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


def test_single_branch_swc():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_single_branch.swc")
    cell = jx.read_swc(fname, ncomp=1)
    cell.branch(0).set_ncomp(3)
    cell.set_ncomp(4)
    cell[0, 0].record()
    v = jx.integrate(cell, t_max=1.0)
    assert np.invert(np.any(np.isnan(v))), "Found a NaN."
