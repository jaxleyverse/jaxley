import pickle

import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse

# create modules
comp = jx.Compartment()
branch = jx.Branch(comp, 4)
cell = jx.Cell([branch] * 3, [-1, 0, 0])
net = jx.Network([cell] * 2)

# insert mechanisms
net.cell(0).branch("all").insert(HH())
net.cell(0).branch(0).comp(0).record("v")
jx.connect(
    net.cell(0).branch(0).comp(0), net.cell(1).branch(0).comp(0), IonotropicSynapse()
)


@pytest.mark.parametrize(
    "module", [comp, branch, cell, net], ids=lambda x: x.__class__.__name__
)
def test_pickle(module):
    pickled = pickle.dumps(module)
    unpickled = pickle.loads(pickled)
    # assert module == unpickled # TODO: implement __eq__ for all classes
