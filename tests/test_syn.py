# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import os

import numpy as np
import pytest
from jax.nn import relu

import jaxley as jx
from jaxley.channels import HH, Leak
from jaxley.connect import connect
from jaxley.synapses import (
    AlphaSynapse,
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


def test_spike_synapse(SpikeNet):
    """Test if SNNs function properly propogate current through their synapses."""
    n_neurons = 2
    half = n_neurons // 2
    net = SpikeNet(n_neurons, 1, 1, connect=True)

    dt = 0.1
    t_max = 40.0

    net.cell(range(half)).stimulate(jx.step_current(5.0, 20.0, 0.01, dt, t_max))
    net.cell(n_neurons - 1).record("v")
    recordings = np.asarray(jx.integrate(net, delta_t=dt))
    assert np.invert(np.any(np.isnan(recordings)))
    assert np.invert(np.all(recordings[0] == recordings[0][0]))


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
    # Test whether the voltage increases after the synaptic current reaches the
    # post-synaptic site.
    assert v[0, int(i_delay * 40)] < v[0, int((i_delay + i_dur) * 40)]


@pytest.mark.parametrize(
    "synapse", [CurrentSynapse, ConductanceSynapse, DynamicSynapse]
)
@pytest.mark.parametrize(
    "nonlinearity",
    [
        relu,
        lambda x: x**2,
    ],
)
def test_synapse_nonlinearity(SimpleNet, synapse, nonlinearity):
    net = SimpleNet(2, 1, 1)

    connect(net.cell(0), net.cell(1), synapse(nonlinearity))
    net.insert(Leak())
    net.cell(1).record()
    net.cell(0).stimulate(jx.step_current(5.0, 10.0, 0.1, 0.025, 20.0))

    v = jx.integrate(net)

    # Voltage should increase after synaptic current arrives
    assert v[0, int(5.0 * 40)] < v[0, int(15.0 * 40)]


def test_predefined_spike_trains():
    """This runs the following how-to guide:
    https://jaxley.readthedocs.io/en/latest/how_to_guide/predefined_spike_trains.html
    """
    # Generate pre-synaptic spike train.
    _ = np.random.seed(42)

    t_max = 100.0  # ms
    dt = 0.025  # ms
    time_steps = int(t_max // dt)

    firing_rate = 50  # Hz
    spike_prob = firing_rate * dt / 1000

    spike_train = np.random.binomial(1, spike_prob, size=time_steps)

    dummy = jx.Cell()
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_ca1_n120_250.swc")
    cell = jx.read_swc(fname, ncomp=1)
    net = jx.Network([dummy, cell])
    net.cell(1).insert(Leak())

    # Connect pre-synaptic dummy to the morphologically detailed cell.
    connect(net.cell(0), net.cell(1).branch(3).comp(0), AlphaSynapse())
    net.set("AlphaSynapse_gS", 0.1)  # Synaptic strength.
    net.set("AlphaSynapse_tau_decay", 5.0)  # decay time in ms

    # Clamp the voltage of the pre-synaptic cell to the spike train.
    net.cell(0).set("v", 0.0)  # Initial state.
    net.cell(0).clamp("v", spike_train)

    net.cell(1).branch(3).comp(0).record()
    v = jx.integrate(net, delta_t=dt)

    assert np.invert(np.any(np.isnan(v)))
