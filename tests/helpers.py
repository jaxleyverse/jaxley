# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np
import pandas as pd

from jaxley.utils.cell_utils import params_to_pstate


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


def import_neuron_morph(fname, ncomp=8):
    from neuron import h

    _ = h.load_file("stdlib.hoc")
    _ = h.load_file("import3d.hoc")

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
    if set(a.columns) != set(b.columns):
        return False
    else:
        a = a[b.columns]

    equal = a.eq(b) | (a.isna() & b.isna())
    return bool(equal.to_numpy().all())


# Accessors for jaxley internals. The data backend of v1 changes these internals, so
# tests read them only through these functions, which then are the only place to update.


def all_parameters(module, params=[], pstate=None) -> dict:
    """All parameters of `module` as a simulation sees them.

    The defaults, overwritten by the trainable `params` (from `get_parameters()`) or,
    if given, by `pstate` (e.g. from `data_set`).
    """
    module.to_jax()
    if pstate is None:
        pstate = params_to_pstate(params, module.indices_set_by_trainables)
    return module.get_all_parameters(pstate)


def comp_edges(module) -> pd.DataFrame:
    """The compartment-to-compartment edges used by the voltage solver."""
    return module._comp_edges
