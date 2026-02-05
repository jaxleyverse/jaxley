# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from typing import List

import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse, Synapse, TestSynapse


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


def test_adex_spike_synapse(SpikeNet):
    """Test if AdEx neurons properly propagate current through their synapses."""
    n_neurons = 2
    half = n_neurons // 2
    net = SpikeNet(n_neurons, 1, 1, connect=True)

    dt = 0.1
    t_max = 40.0

    # Stimulate the first half of neurons
    net.cell(range(half)).stimulate(jx.step_current(5.0, 20.0, 100.0, dt, t_max))
    net.cell(n_neurons - 1).record("v")
    recordings = np.asarray(jx.integrate(net, delta_t=dt))
    assert np.invert(np.any(np.isnan(recordings)))
