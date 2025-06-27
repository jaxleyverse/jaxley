# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from typing import List

import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import (
    IonotropicSynapse,
    Synapse,
    TanhConductanceSynapse,
    TanhRateSynapse,
    TestSynapse,
)


def test_multiparameter_setting(SimpleNet):
    """
    Test if the correct parameters are set if one type of synapses is inserted.

    Tests global index dropping: d4daaf019596589b9430219a15f1dda0b1c34d85
    """
    net = SimpleNet(2, 1, 4)

    pre = net.cell(0).branch(0).loc(0.0)
    post = net.cell(1).branch(0).loc(0.0)
    connect(pre, post, IonotropicSynapse())

    syn_view = net.IonotropicSynapse
    syn_params = ["IonotropicSynapse_gS", "IonotropicSynapse_e_syn"]

    for p in syn_params:
        syn_view.set(p, 0.32)


def _get_synapse_view(net, synapse_name, single_idx=1, double_idxs=[2, 3]):
    """Access to the synapse view"""
    if synapse_name == "IonotropicSynapse":
        full_syn_view = net.IonotropicSynapse
        single_syn_view = net.IonotropicSynapse.edge(single_idx)
        double_syn_view = net.IonotropicSynapse.edge(double_idxs)
    if synapse_name == "TanhRateSynapse":
        full_syn_view = net.TanhRateSynapse
        single_syn_view = net.TanhRateSynapse.edge(single_idx)
        double_syn_view = net.TanhRateSynapse.edge(double_idxs)
    if synapse_name == "TestSynapse":
        full_syn_view = net.TestSynapse
        single_syn_view = net.TestSynapse.edge(single_idx)
        double_syn_view = net.TestSynapse.edge(double_idxs)
    if synapse_name == "TanhConductanceSynapse":
        full_syn_view = net.TanhConductanceSynapse
        single_syn_view = net.TanhConductanceSynapse.edge(single_idx)
        double_syn_view = net.TanhConductanceSynapse.edge(double_idxs)
    return full_syn_view, single_syn_view, double_syn_view


@pytest.mark.parametrize(
    "synapse_type",
    [IonotropicSynapse, TanhRateSynapse, TestSynapse, TanhConductanceSynapse],
)
def test_set_and_querying_params_one_type(synapse_type, SimpleNet):
    """Test if the correct parameters are set if one type of synapses is inserted."""
    synapse_type = synapse_type()
    net = SimpleNet(4, 1, 4)

    for pre_ind in [0, 1]:
        for post_ind in [2, 3]:
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            connect(pre, post, synapse_type)

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


@pytest.mark.parametrize(
    "synapse_type", [TanhRateSynapse, TanhConductanceSynapse, TestSynapse]
)
def test_set_and_querying_params_two_types(synapse_type, SimpleNet):
    """Test whether the correct parameters are set."""
    synapse_type = synapse_type()
    net = SimpleNet(4, 1, 4)

    for pre_ind in [0, 1]:
        for post_ind, synapse in zip([2, 3], [IonotropicSynapse(), synapse_type]):
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            connect(pre, post, synapse)

    type1_params = list(IonotropicSynapse().synapse_params.keys())
    synapse_type_params = list(synapse_type.synapse_params.keys())

    default_synapse_type = net.edges[synapse_type_params[0]].to_numpy()[[1, 3]]

    net.set(type1_params[0], 0.15)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.15)
    if synapse_type_params[0] != type1_params[0]:
        assert np.all(
            net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == default_synapse_type
        )
    else:
        default_synapse_type = 0.15

    synapse_type_name = type(synapse_type).__name__
    synapse_type_full, synapse_type_single, synapse_type_double = _get_synapse_view(
        net, synapse_type_name, double_idxs=[0, 1]
    )

    # Generalize to all parameters
    net.IonotropicSynapse.set(type1_params[0], 0.32)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.32)
    assert np.all(
        net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == default_synapse_type
    )

    synapse_type_full.set(synapse_type_params[0], 0.18)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.32)
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.18)

    net.IonotropicSynapse.edge(1).set(type1_params[0], 0.24)
    assert net.edges[type1_params[0]][0] == 0.32
    assert net.edges[type1_params[0]][2] == 0.24
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.18)

    net.IonotropicSynapse.edge([0, 1]).set(type1_params[0], 0.27)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.18)

    synapse_type_double.set(synapse_type_params[0], 0.21)
    assert np.all(net.edges[type1_params[0]].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges[synapse_type_params[0]].to_numpy()[[1, 3]] == 0.21)


@pytest.mark.parametrize(
    "synapse_type", [TanhRateSynapse, TanhConductanceSynapse, TestSynapse]
)
def test_shuffling_order_of_set(synapse_type, SimpleNet):
    """Test whether the result is the same if the order of synapses is changed."""
    synapse_type = synapse_type()

    net1 = SimpleNet(4, 1, 4)
    net2 = SimpleNet(4, 1, 4)

    connect(
        net1.cell(0).branch(0).loc(1.0),
        net1.cell(1).branch(0).loc(0.1),
        IonotropicSynapse(),
    )
    connect(
        net1.cell(1).branch(0).loc(0.6),
        net1.cell(2).branch(0).loc(0.7),
        IonotropicSynapse(),
    )
    connect(
        net1.cell(2).branch(0).loc(0.4), net1.cell(3).branch(0).loc(0.3), synapse_type
    )
    connect(
        net1.cell(3).branch(0).loc(0.1), net1.cell(1).branch(0).loc(0.1), synapse_type
    )

    # Different order as for `net1`.
    connect(
        net2.cell(3).branch(0).loc(0.1), net2.cell(1).branch(0).loc(0.1), synapse_type
    )
    connect(
        net2.cell(1).branch(0).loc(0.6),
        net2.cell(2).branch(0).loc(0.7),
        IonotropicSynapse(),
    )
    connect(
        net2.cell(2).branch(0).loc(0.4), net2.cell(3).branch(0).loc(0.3), synapse_type
    )
    connect(
        net2.cell(0).branch(0).loc(1.0),
        net2.cell(1).branch(0).loc(0.1),
        IonotropicSynapse(),
    )

    net1.insert(HH())
    net2.insert(HH())

    for i in range(4):
        net1.cell(i).branch(0).loc(0.0).record()
        net2.cell(i).branch(0).loc(0.0).record()

    voltages1 = jx.integrate(net1, t_max=5.0)
    voltages2 = jx.integrate(net2, t_max=5.0)

    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_synapse_recording_order(SimpleNet):
    """Ensure that record finds the correct synapse index"""
    net1 = SimpleNet(4, 1, 1)
    jx.fully_connect(net1.cell([0, 1]), net1.cell([3]), IonotropicSynapse())
    net1.IonotropicSynapse.record("IonotropicSynapse_s")
    stim = np.zeros_like(np.arange(0, 2, 0.025))  # 2 ms stimulus
    stim[int(0.5 / 0.025) :] = 0.05
    net1.cell(1).stimulate(stim)
    soln1 = jx.integrate(net1)

    net2 = SimpleNet(4, 1, 1)
    # Addition of these extra synapses should not throw off the indices recorded
    jx.fully_connect(net2.cell([0, 1]), net2.cell([2]), TestSynapse())
    jx.fully_connect(net2.cell([0, 1]), net2.cell([3]), IonotropicSynapse())
    net2.IonotropicSynapse.record("IonotropicSynapse_s")
    net2.cell(1).stimulate(stim)
    soln2 = jx.integrate(net2)

    assert np.allclose(soln1, soln2)
