import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from typing import List
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import Synapse, GlutamateSynapse, TanhRateSynapse, TestSynapse


def _get_synapse_view(net, synapse_name, single_idx=1, double_idxs=[2, 3]):
    """Is there a better way to access the synapse view?"""
    if synapse_name == "GlutamateSynapse":
        full_syn_view = net.GlutamateSynapse
        single_syn_view = net.GlutamateSynapse(single_idx)
        double_syn_view = net.GlutamateSynapse(double_idxs)
    if synapse_name == "TanhRateSynapse":
        full_syn_view = net.TanhRateSynapse
        single_syn_view = net.TanhRateSynapse(single_idx)
        double_syn_view = net.TanhRateSynapse(double_idxs)
    if synapse_name == "TestSynapse":
        full_syn_view = net.TestSynapse
        single_syn_view = net.TestSynapse(single_idx)
        double_syn_view = net.TestSynapse(double_idxs)
    return full_syn_view, single_syn_view, double_syn_view


def test_set_and_querying_params_one_type(synapse_type):
    """Test if the correct parameters are set if one type of synapses is inserted."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(4)])

    for pre_ind in [0, 1]:
        for post_ind in [2, 3]:
            pre = net.cell(pre_ind).branch(0).comp(0.0)
            post = net.cell(post_ind).branch(0).comp(0.0)
            pre.connect(post, synapse_type)

    # Get the synapse parameters to test setting
    syn_params = list(synapse_type.synapse_params.keys())
    for p in syn_params:
        net.set(p, 0.15)
        assert np.all(net.edges[p].to_numpy() == 0.15)

    synapse_name = type(synapse_type).__name__
    full_syn_view, single_syn_view, double_syn_view = _get_synapse_view(
        net, synapse_name
    )

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


def test_set_and_querying_params_two_types(type1, type2):
    """Test whether the correct parameters are set."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(4)])

    for pre_ind in [0, 1]:
        for post_ind, synapse in zip([2, 3], [type1, type2]):
            pre = net.cell(pre_ind).branch(0).comp(0.0)
            post = net.cell(post_ind).branch(0).comp(0.0)
            pre.connect(post, synapse)

    type1_params = list(type1.synapse_params.keys())
    type2_params = list(type2.synapse_params.keys())

    default_type2 = net.edges[type2_params[0]].to_numpy()[[1, 3]]

    net.set(type1_params[0], 0.15)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.15)
    if type2_params[0] != type1_params[0]:
        assert np.all(net.edges[type2_params[0]].to_numpy()[[1, 3]] == default_type2)
    else:
        default_type2 = 0.15

    type1_name = type(type1).__name__
    type1_full, type1_single, type1_double = _get_synapse_view(
        net, type1_name, double_idxs=[0, 1]
    )
    type2_name = type(type2).__name__
    type2_full, type2_single, type2_double = _get_synapse_view(
        net, type2_name, double_idxs=[0, 1]
    )

    # Generalize to all parameters
    type1_full.set(type1_params[0], 0.32)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.32)
    assert np.all(net.edges[type2_params[0]].to_numpy()[[1, 3]] == default_type2)

    type2_full.set(type2_params[0], 0.18)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.32)
    assert np.all(net.edges[type2_params[0]].to_numpy()[[1, 3]] == 0.18)

    type1_single.set(type1_params[0], 0.24)
    assert net.edges[type1_params[0]][0] == 0.32
    assert net.edges[type1_params[0]][2] == 0.24
    assert np.all(net.edges[type2_params[0]].to_numpy()[[1, 3]] == 0.18)

    type1_double.set(type1_params[0], 0.27)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges[type2_params[0]].to_numpy()[[1, 3]] == 0.18)

    type2_double.set(type2_params[0], 0.21)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges[type2_params[0]].to_numpy()[[1, 3]] == 0.21)


def test_shuffling_order_of_set(type1, type2):
    """Test whether the result is the same if the order of synapses is changed."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1])

    net1 = jx.Network([cell for _ in range(4)])
    net2 = jx.Network([cell for _ in range(4)])

    net1.cell(0).branch(0).comp(1.0).connect(net1.cell(1).branch(0).comp(0.1), type1)
    net1.cell(1).branch(0).comp(0.6).connect(net1.cell(2).branch(0).comp(0.7), type1)
    net1.cell(2).branch(0).comp(0.4).connect(net1.cell(3).branch(0).comp(0.3), type2)
    net1.cell(3).branch(0).comp(0.1).connect(net1.cell(1).branch(0).comp(0.1), type2)

    # Different order as for `net1`.
    net2.cell(3).branch(0).comp(0.1).connect(net2.cell(1).branch(0).comp(0.1), type2)
    net2.cell(1).branch(0).comp(0.6).connect(net2.cell(2).branch(0).comp(0.7), type1)
    net2.cell(2).branch(0).comp(0.4).connect(net2.cell(3).branch(0).comp(0.3), type2)
    net2.cell(0).branch(0).comp(1.0).connect(net2.cell(1).branch(0).comp(0.1), type1)

    net1.insert(HH())
    net2.insert(HH())

    for i in range(4):
        net1.cell(i).branch(0).comp(0.0).record()
        net2.cell(i).branch(0).comp(0.0).record()

    voltages1 = jx.integrate(net1, t_max=5.0)
    voltages2 = jx.integrate(net2, t_max=5.0)

    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_synapses(synapses: List[Synapse]):
    base_synapse = synapses[0]
    test_set_and_querying_params_one_type(base_synapse)
    synapses = synapses[1:]

    for syn in synapses:
        test_set_and_querying_params_one_type(syn)
        test_set_and_querying_params_two_types(base_synapse, syn)
        test_shuffling_order_of_set(base_synapse, syn)


if __name__ == "__main__":
    test_synapses([GlutamateSynapse(), TestSynapse(), TanhRateSynapse()])
