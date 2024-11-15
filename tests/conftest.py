# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
import warnings
from copy import deepcopy

import pytest

import jaxley as jx
from jaxley.synapses import IonotropicSynapse


@pytest.fixture(scope="session")
def SimpleComp():
    comps = {}

    def get_or_build_comp(copy=True, force_init=False):
        if "comp" not in comps or force_init:
            comps["comp"] = jx.Compartment()
        return deepcopy(comps["comp"]) if copy else comps["comp"]

    yield get_or_build_comp
    comps = {}


@pytest.fixture(scope="session")
def SimpleBranch(SimpleComp):
    branches = {}

    def get_or_build_branch(nseg, copy=True, force_init=False):
        if nseg not in branches or force_init:
            comp = SimpleComp(force_init=force_init)
            branches[nseg] = jx.Branch([comp] * nseg)
        return deepcopy(branches[nseg]) if copy else branches[nseg]

    yield get_or_build_branch
    branches = {}


@pytest.fixture(scope="session")
def SimpleCell(SimpleBranch):
    cells = {}

    def get_or_build_cell(nbranches, nseg, copy=True, force_init=False):
        if key := (nbranches, nseg) not in cells or force_init:
            parents = [-1]
            depth = 0
            while nbranches > len(parents):
                parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
                depth += 1
            parents = parents[:nbranches]
            branch = SimpleBranch(nseg=nseg, force_init=force_init)
            cells[key] = jx.Cell([branch] * nbranches, parents)
        return deepcopy(cells[key]) if copy else cells[key]

    yield get_or_build_cell
    cells = {}


@pytest.fixture(scope="session")
def SimpleNet(SimpleCell):
    nets = {}

    def get_or_build_net(
        ncells, nbranches, nseg, connect=False, copy=True, force_init=False
    ):
        if key := (ncells, nbranches, nseg, connect) not in nets or force_init:
            net = jx.Network(
                [SimpleCell(nbranches=nbranches, nseg=nseg, force_init=force_init)]
                * ncells
            )
            if connect:
                jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
            nets[key] = net
        return deepcopy(nets[key]) if copy else nets[key]

    yield get_or_build_net
    nets = {}


@pytest.fixture(scope="session")
def SimpleMorphCell():
    dirname = os.path.dirname(__file__)
    default_fname = os.path.join(dirname, "swc_files", "morph.swc")  # n120

    cells = {}

    def get_or_build_cell(
        fname=None, nseg=1, max_branch_len=2_000.0, copy=True, force_init=False
    ):
        fname = default_fname if fname is None else fname
        if key := (fname, nseg, max_branch_len) not in cells or force_init:
            cells[key] = jx.read_swc(fname, nseg, max_branch_len, assign_groups=True)
        return deepcopy(cells[key]) if copy else cells[key]

    yield get_or_build_cell
    cells = {}


@pytest.fixture(scope="session")
def swc2jaxley():
    dirname = os.path.dirname(__file__)
    default_fname = os.path.join(dirname, "swc_files", "morph.swc")  # n120

    params = {}

    def get_or_compute_swc2jaxley_params(
        fname=None, max_branch_len=2_000.0, sort=True, force_init=False
    ):
        fname = default_fname if fname is None else fname
        if key := (fname, max_branch_len, sort) not in params or force_init:
            params[key] = jx.utils.swc.swc_to_jaxley(fname, max_branch_len, sort)
        return params[key]

    yield get_or_compute_swc2jaxley_params
    params = {}
