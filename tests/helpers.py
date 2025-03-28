# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np
import pandas as pd


def get_segment_xyzrL(section, comp_idx=None, loc=None, ncomp=8):
    assert (
        comp_idx is not None or loc is not None
    ), "Either comp_idx or loc must be provided."
    assert not (
        comp_idx is not None and loc is not None
    ), "Only one of comp_idx or loc can be provided."

    comp_len = 1 / ncomp
    loc = comp_len / 2 + comp_idx * comp_len if loc is None else loc

    n3d = section.n3d()
    x3d = np.array([section.x3d(i) for i in range(n3d)])
    y3d = np.array([section.y3d(i) for i in range(n3d)])
    z3d = np.array([section.z3d(i) for i in range(n3d)])
    L = np.array([section.arc3d(i) for i in range(n3d)])  # Cumulative arc lengths
    r3d = np.array([section.diam3d(i) / 2 for i in range(n3d)])
    if loc is None:
        return x3d, y3d, z3d, r3d
    else:
        total_length = L[-1]
        target_length = loc * total_length

        # Find segment containing target_length
        for i in range(1, n3d):
            if L[i] >= target_length:
                break
        else:
            i = n3d - 1

        # Interpolate between points i-1 and i
        L0, L1 = L[i - 1], L[i]
        t = (target_length - L0) / (L1 - L0)
        x = x3d[i - 1] + t * (x3d[i] - x3d[i - 1])
        y = y3d[i - 1] + t * (y3d[i] - y3d[i - 1])
        z = z3d[i - 1] + t * (z3d[i] - z3d[i - 1])
        r = r3d[i - 1] + t * (r3d[i] - r3d[i - 1])
        return x, y, z, r, L[-1] / ncomp


def jaxley2neuron_by_coords(jx_cell, neuron_secs, comp_idx=None, loc=None, ncomp=8):
    neuron_coords = {
        i: np.vstack(get_segment_xyzrL(sec, comp_idx=comp_idx, loc=loc, ncomp=ncomp))[
            :3
        ].T
        for i, sec in enumerate(neuron_secs)
    }
    neuron_coords = np.vstack(
        [np.hstack([k * np.ones((v.shape[0], 1)), v]) for k, v in neuron_coords.items()]
    )
    neuron_coords = pd.DataFrame(
        neuron_coords, columns=["global_branch_index", "x", "y", "z"]
    )
    neuron_coords["global_branch_index"] = neuron_coords["global_branch_index"].astype(
        int
    )

    neuron_loc_xyz = neuron_coords.groupby("global_branch_index").mean()
    jaxley_loc_xyz = (
        jx_cell.branch("all")
        .loc(loc)
        .nodes.set_index("global_branch_index")[["x", "y", "z"]]
    )

    jaxley2neuron_inds = {}
    for i, xyz in enumerate(jaxley_loc_xyz.to_numpy()):
        d = np.sqrt(((neuron_loc_xyz - xyz) ** 2)).sum(axis=1)
        jaxley2neuron_inds[i] = d.argmin()
    return jaxley2neuron_inds


def jaxley2neuron_by_group(
    jx_cell,
    neuron_secs,
    comp_idx=None,
    loc=None,
    ncomp=8,
    num_apical=20,
    num_tuft=20,
    num_basal=10,
):
    y_apical = (
        jx_cell.apical.nodes.groupby("global_branch_index")
        .mean()["y"]
        .abs()
        .sort_values()
    )
    trunk_inds = y_apical.index[:num_apical].tolist()
    tuft_inds = y_apical.index[-num_tuft:].tolist()
    basal_inds = (
        jx_cell.basal.nodes["global_branch_index"].unique()[:num_basal].tolist()
    )

    jaxley2neuron = jaxley2neuron_by_coords(
        jx_cell, neuron_secs, comp_idx=comp_idx, loc=loc, ncomp=ncomp
    )

    neuron_trunk_inds = [jaxley2neuron[i] for i in trunk_inds]
    neuron_tuft_inds = [jaxley2neuron[i] for i in tuft_inds]
    neuron_basal_inds = [jaxley2neuron[i] for i in basal_inds]

    neuron_inds = {
        "trunk": neuron_trunk_inds,
        "tuft": neuron_tuft_inds,
        "basal": neuron_basal_inds,
    }
    jaxley_inds = {"trunk": trunk_inds, "tuft": tuft_inds, "basal": basal_inds}
    return neuron_inds, jaxley_inds


def match_stim_loc(jx_cell, neuron_sec, comp_idx=None, loc=None, ncomp=8):
    stim_coords = get_segment_xyzrL(
        neuron_sec, comp_idx=comp_idx, loc=loc, ncomp=ncomp
    )[:3]
    stim_idx = (
        ((jx_cell.nodes[["x", "y", "z"]] - stim_coords) ** 2).sum(axis=1).argmin()
    )
    return stim_idx


def import_neuron_morph(fname, ncomp=8):
    from neuron import h

    _ = h.load_file("stdlib.hoc")
    _ = h.load_file("import3d.hoc")
    ncomp = 8

    ##################### NEURON ##################
    for sec in h.allsec():
        h.delete_section(sec=sec)

    cell = h.Import3d_SWC_read()
    cell.input(fname)
    i3d = h.Import3d_GUI(cell, False)
    i3d.instantiate(None)

    for sec in h.allsec():
        sec.nseg = ncomp
    return h, cell


def equal_both_nan_or_empty_df(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Return whether all elements of two dataframes are identical."""
    if "xyzr" in a.columns:
        a = a.drop(columns="xyzr")
    if "xyzr" in b.columns:
        b = b.drop(columns="xyzr")
    if a.empty and b.empty:
        return True
    a[a.isna()] = -1
    b[b.isna()] = -1
    if set(a.columns) != set(b.columns):
        return False
    else:
        a = a[b.columns]
    return (a == b).all()
