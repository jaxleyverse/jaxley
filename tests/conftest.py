# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
from copy import deepcopy

import pytest

import jaxley as jx
from jaxley.synapses import IonotropicSynapse


@pytest.fixture(scope="session")
def SimpleComp():
    comp = jx.Compartment()

    def get_comp(copy=True):
        return deepcopy(comp) if copy else comp

    yield get_comp
    del comp


@pytest.fixture(scope="session")
def SimpleBranch(SimpleComp):
    branches = {}

    def branch_w_shape(nseg, copy=True):
        if nseg not in branches:
            comp = SimpleComp()
            branches[nseg] = jx.Branch([comp] * nseg)
        return deepcopy(branches[nseg]) if copy else branches[nseg]

    yield branch_w_shape
    branches = {}


@pytest.fixture(scope="session")
def SimpleCell(SimpleBranch):
    cells = {}

    def cell_w_shape(nbranches, nseg, copy=True):
        if key := (nbranches, nseg) not in cells:
            parents = [-1]
            depth = 0
            while nbranches > len(parents):
                parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
                depth += 1
            parents = parents[:nbranches]
            branch = SimpleBranch(nseg)
            cells[key] = jx.Cell([branch] * nbranches, parents)
        return deepcopy(cells[key]) if copy else cells[key]

    yield cell_w_shape
    cells = {}


@pytest.fixture(scope="session")
def SimpleNet(SimpleCell):
    nets = {}

    def net_w_shape(ncells, nbranches, nseg, connect=False, copy=True):
        if key := (ncells, nbranches, nseg, connect) not in nets:
            net = jx.Network([SimpleCell(nbranches, nseg)] * ncells)
            if connect:
                jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
            nets[key] = net
        return deepcopy(nets[key]) if copy else nets[key]

    yield net_w_shape
    nets = {}


@pytest.fixture(scope="session")
def SimpleMorphCell():
    dirname = os.path.dirname(__file__)
    default_fname = os.path.join(dirname, "swc_files", "morph.swc")  # n120

    cells = {}

    def cell_w_params(fname=None, nseg=1, max_branch_len=2_000.0, copy=True):
        fname = default_fname if fname is None else fname
        if key := (fname, nseg, max_branch_len) not in cells:
            cells[key] = jx.read_swc(fname, nseg, max_branch_len, assign_groups=True)
        return deepcopy(cells[key]) if copy else cells[key]

    yield cell_w_params
    cells = {}


@pytest.fixture(scope="session")
def swc2jaxley():
    dirname = os.path.dirname(__file__)
    default_fname = os.path.join(dirname, "swc_files", "morph.swc")  # n120

    params = {}

    def swc2jaxley_params(fname=None, max_branch_len=2_000.0, sort=True):
        fname = default_fname if fname is None else fname
        if key := (fname, max_branch_len, sort) not in params:
            params[key] = jx.utils.swc.swc_to_jaxley(fname, max_branch_len, sort)
        return params[key]

    yield swc2jaxley_params
    params = {}
