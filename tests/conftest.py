import os
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse, TestSynapse


@pytest.fixture(scope="session")
def SimpleComp():
    comp = jx.Compartment()

    def get_comp(copy=False):
        return deepcopy(comp) if copy else comp

    yield get_comp
    comp = None


@pytest.fixture(scope="session")
def SimpleBranch(SimpleComp):
    branches = {}

    def branch_w_shape(nseg, copy=False):
        if nseg not in branches:
            branches[nseg] = jx.Branch([SimpleComp()] * nseg)
        return deepcopy(branches[nseg]) if copy else branches[nseg]

    yield branch_w_shape
    branches = {}


@pytest.fixture(scope="session")
def SimpleCell(SimpleBranch):
    cells = {}

    def cell_w_shape(nbranches, nseg_per_branch, copy=False):
        if key := (nbranches, nseg_per_branch) not in cells:
            parents = [-1]
            depth = 0
            while nbranches > len(parents):
                parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
                depth += 1
            parents = parents[:nbranches]
            cells[key] = jx.Cell([SimpleBranch(nseg_per_branch)] * nbranches, parents)
        return deepcopy(cells[key]) if copy else cells[key]

    yield cell_w_shape
    cells = {}


@pytest.fixture(scope="session")
def SimpleNet(SimpleCell):
    nets = {}

    def net_w_shape(n_cells, nbranches, nseg_per_branch, connect=False, copy=False):
        if key := (n_cells, nbranches, nseg_per_branch, connect) not in nets:
            net = jx.Network([SimpleCell(nbranches, nseg_per_branch)] * n_cells)
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

    def cell_w_params(fname=None, nseg=2, max_branch_len=2_000.0, copy=False):
        fname = default_fname if fname is None else fname
        if key := (fname, nseg, max_branch_len) not in cells:
            cells[key] = jx.read_swc(fname, nseg, max_branch_len, assign_groups=True)
        return deepcopy(cells[key]) if copy else cells[key]

    yield cell_w_params
    cells = {}
