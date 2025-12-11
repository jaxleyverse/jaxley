# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from typing import List

import numpy as np
import pytest
from jax.nn import relu

import jaxley as jx
from jaxley.channels import HH, Leak
from jaxley.connect import connect
from jaxley.synapses import (
    ConductanceSynapse,
    CurrentSynapse,
    DynamicSynapse,
    IonotropicSynapse,
    Synapse,
    TestSynapse,
)


def test_set_and_querying_params_one_type(SimpleNet):
    """Test if the correct parameters are set if one type of synapses is inserted."""
    net = SimpleNet(4, 1, 4)

    for pre_ind in [0, 1]:
        for post_ind in [2, 3]:
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            connect(pre, post, IonotropicSynapse())

    # Get the synapse parameters to test setting
    syn_params = list(IonotropicSynapse().synapse_params.keys())
    for p in syn_params:
        net.set(p, 0.15)
        assert np.all(net.edges[p].to_numpy() == 0.15)

    full_syn_view = net.IonotropicSynapse
    single_syn_view = net.IonotropicSynapse.edge(1)
    double_syn_view = net.IonotropicSynapse.edge([2, 3])

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
    "synapse", [CurrentSynapse, ConductanceSynapse, DynamicSynapse, IonotropicSynapse]
)
def test_synapse_correctness(SimpleNet, synapse):
    net = SimpleNet(2, 1, 1)
    connect(net.cell(0), net.cell(1), synapse())
    net.insert(Leak())
    net.cell(1).record()

    i_delay = 5.0
    i_dur = 10.0
    i_amp = 0.1
    net.cell(0).stimulate(jx.step_current(i_delay, i_dur, i_amp, 0.025, 20.0))

    v = jx.integrate(net)
    assert v[0, int(i_delay * 40)] < v[0, int((i_delay + i_dur) * 40)]


@pytest.mark.parametrize(
    "synapse", [CurrentSynapse, ConductanceSynapse, DynamicSynapse]
)
def test_synapse_relu_nonlinearity(SimpleNet, synapse):
    net = SimpleNet(2, 1, 1)
    connect(net.cell(0), net.cell(1), synapse(relu))
    net.insert(Leak())
    net.cell(1).record()
    net.cell(0).stimulate(jx.step_current(5.0, 10.0, 0.1, 0.025, 20.0))
    v = jx.integrate(net)
    assert v[0, int(5.0 * 40)] < v[0, int((15.0) * 40)]


@pytest.mark.parametrize(
    "synapse", [CurrentSynapse, ConductanceSynapse, DynamicSynapse]
)
def test_synapse_custom_nonlinearity(SimpleNet, synapse):
    net = SimpleNet(2, 1, 1)

    def nonlinearity(x):
        return x**2

    connect(net.cell(0), net.cell(1), synapse(nonlinearity))
    net.insert(Leak())
    net.cell(1).record()
    net.cell(0).stimulate(jx.step_current(5.0, 10.0, 0.1, 0.025, 20.0))
    v = jx.integrate(net)
    assert v[0, int(5.0 * 40)] < v[0, int((15.0) * 40)]
