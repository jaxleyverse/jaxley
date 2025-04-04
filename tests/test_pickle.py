# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
import pickle

import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse

# create modules (cannot use fixtures for pickling, since they rely on local func defs)
comp = jx.Compartment()
branch = jx.Branch(comp, 4)
cell = jx.Cell([branch] * 3, [-1, 0, 0])
net = jx.Network([cell] * 2)
fname = os.path.join(os.path.dirname(__file__), "swc_files", "morph_ca1_n120.swc")
morph_cell = jx.read_swc(fname, ncomp=1, max_branch_len=2_000, assign_groups=True)

# insert mechanisms
net.cell(0).branch("all").insert(HH())
net.cell(0).branch(0).comp(0).record("v")
jx.connect(
    net.cell(0).branch(0).comp(0), net.cell(1).branch(0).comp(0), IonotropicSynapse()
)


@pytest.mark.parametrize(
    "module", [comp, branch, cell, morph_cell, net], ids=lambda x: x.__class__.__name__
)
def test_pickle(module):
    pickled = pickle.dumps(module)
    unpickled = pickle.loads(pickled)

    view = module.select(0)
    pickled = pickle.dumps(view)
    unpickled = pickle.loads(pickled)

    # assert module == unpickled # TODO: implement __eq__ for all classes
