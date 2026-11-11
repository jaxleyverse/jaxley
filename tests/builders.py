# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Cached builders for the modules used in tests.

Each builder constructs its module once per process and returns a deep copy on every
call, so tests can modify the result freely:

    from tests import builders as build

    net = build.net(ncells=2, nbranches=4, ncomp=1)
"""

import os
from copy import deepcopy
from functools import cache, wraps

import jaxley as jx
from jaxley.synapses import IonotropicSynapse

SWC_DIR = os.path.join(os.path.dirname(__file__), "swc_files")
CA1 = os.path.join(SWC_DIR, "morph_ca1_n120.swc")


def cached(build):
    """Build once per set of arguments, then return a deep copy on every call."""
    build_once = cache(build)

    @wraps(build)
    def builder(*args, **kwargs):
        return deepcopy(build_once(*args, **kwargs))

    builder.cache_info = build_once.cache_info
    return builder


def binary_tree_parents(nbranches: int) -> list:
    """Parents of a binary tree of branches: `[-1, 0, 0, 1, 1, 2, 2, ...]`."""
    return [-1] + [b // 2 for b in range(nbranches - 1)]


@cached
def comp() -> jx.Compartment:
    return jx.Compartment()


@cached
def branch(*, ncomp: int) -> jx.Branch:
    return jx.Branch([comp()] * ncomp)


@cached
def cell(*, nbranches: int, ncomp: int) -> jx.Cell:
    """A cell whose branches form a binary tree (see `binary_tree_parents`)."""
    return jx.Cell([branch(ncomp=ncomp)] * nbranches, binary_tree_parents(nbranches))


@cached
def net(
    *, ncells: int, nbranches: int, ncomp: int, connect: bool = False
) -> jx.Network:
    """A network of identical `cell`s. With `connect=True`, the first compartment of
    cell 0 is connected to the first compartment of cell 1 by an `IonotropicSynapse`."""
    net = jx.Network([cell(nbranches=nbranches, ncomp=ncomp)] * ncells)
    if connect:
        jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
    return net


@cached
def swc_cell(
    fname: str = CA1,
    *,
    ncomp: int = 1,
    max_branch_len: float = 2_000.0,
    backend: str = "graph",
    ignore_swc_tracing_interruptions: bool = True,
) -> jx.Cell:
    """A cell read from an SWC file, with groups assigned."""
    return jx.read_swc(
        fname,
        ncomp=ncomp,
        max_branch_len=max_branch_len,
        assign_groups=True,
        backend=backend,
        ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
    )
