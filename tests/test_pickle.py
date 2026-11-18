# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pickle

import numpy as np
import pandas as pd
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse
from tests import builders as build


def _net_with_mechanisms():
    net = build.net(ncells=2, nbranches=3, ncomp=4)
    net.cell(0).branch("all").insert(HH())
    net.cell(0).branch(0).comp(0).record("v")
    jx.connect(
        net.cell(0).branch(0).comp(0),
        net.cell(1).branch(0).comp(0),
        IonotropicSynapse(),
    )
    return net


MODULES = {
    "comp": build.comp,
    "branch": lambda: build.branch(ncomp=4),
    "cell": lambda: build.cell(nbranches=3, ncomp=4),
    "swc_cell": lambda: build.swc_cell(ncomp=1),
    "net": _net_with_mechanisms,
}


@pytest.mark.parametrize("name", MODULES)
def test_pickle(name):
    module = MODULES[name]()
    unpickled = pickle.loads(pickle.dumps(module))
    assert type(unpickled) is type(module)
    pd.testing.assert_frame_equal(unpickled.nodes, module.nodes)
    pd.testing.assert_frame_equal(unpickled.edges, module.edges)
    pd.testing.assert_frame_equal(unpickled.rec_info, module.rec_info)

    view = module.select(0)
    unpickled_view = pickle.loads(pickle.dumps(view))
    pd.testing.assert_frame_equal(unpickled_view.nodes, view.nodes)


def test_pickle_after_integrate():
    """Simulating a module returned by `to_jax()` caches compiled solve functions on
    it, which must not stop it from being pickled."""
    module = _net_with_mechanisms().to_jax()
    expected = jx.integrate(module, t_max=1.0)
    unpickled = pickle.loads(pickle.dumps(module))
    np.testing.assert_allclose(jx.integrate(unpickled, t_max=1.0), expected)
