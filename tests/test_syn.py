import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from typing import List

import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import GlutamateSynapse, Synapse, TestSynapse


def test_set_and_querying_params_one_type():
    """Test if the correct parameters are set if one type of synapses is inserted."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(4)])

    for pre_ind in [0, 1]:
        for post_ind in [2, 3]:
            pre = net.cell(pre_ind).branch(0).comp(0.0)
            post = net.cell(post_ind).branch(0).comp(0.0)
            pre.connect(post, GlutamateSynapse())

    # Get the synapse parameters to test setting
    syn_params = list(GlutamateSynapse().synapse_params.keys())
    for p in syn_params:
        net.set(p, 0.15)
        assert np.all(net.edges[p].to_numpy() == 0.15)

    full_syn_view = net.GlutamateSynapse
    single_syn_view = net.GlutamateSynapse(1)
    double_syn_view = net.GlutamateSynapse([2, 3])

    # There shouldn't be too many synapse_params otherwise this will take a long time
    for p in syn_params:
        full_syn_view.set(p, 0.32)
        assert np.all(net.edges[p].to_numpy() == 0.32)

        single_syn_view.set(p, 0.18)
        assert net.edges[p].to_numpy()[1] == 0.18
        assert np.all(net.edges[p].to_numpy()[np.asarray([0, 2, 3])] == 0.32)

        double_syn_view.set(p, 0.12)
        assert net.edges[p][0] == 0.32
        assert net.edges[p][1] == 0.18
        assert np.all(net.edges[p].to_numpy()[np.asarray([2, 3])] == 0.12)
