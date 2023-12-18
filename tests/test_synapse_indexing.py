import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import GlutamateSynapse, TestSynapse


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

    net.set("gS", 0.15)
    assert np.all(net.edges["gS"].to_numpy() == 0.15)

    net.GlutamateSynapse.set("gS", 0.32)
    assert np.all(net.edges["gS"].to_numpy() == 0.32)

    net.GlutamateSynapse(1).set("gS", 0.18)
    assert net.edges["gS"].to_numpy()[1] == 0.18
    assert np.all(net.edges["gS"].to_numpy()[np.asarray([0, 2, 3])] == 0.32)

    net.GlutamateSynapse([2, 3]).set("gS", 0.12)
    assert net.edges["gS"][0] == 0.32
    assert net.edges["gS"][1] == 0.18
    assert np.all(net.edges["gS"].to_numpy()[np.asarray([2, 3])] == 0.12)


def test_set_and_querying_params_two_types():
    """Test whether the correct parameters are set."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(4)])

    for pre_ind in [0, 1]:
        for post_ind, synapse in zip([2, 3], [GlutamateSynapse(), TestSynapse()]):
            pre = net.cell(pre_ind).branch(0).comp(0.0)
            post = net.cell(post_ind).branch(0).comp(0.0)
            pre.connect(post, synapse)

    net.set("gS", 0.15)
    assert np.all(net.edges["gS"].to_numpy()[[0, 2]] == 0.15)
    assert np.all(
        net.edges["gC"].to_numpy()[[1, 3]] == 0.5
    )  # 0.5 is the default value.

    net.GlutamateSynapse.set("gS", 0.32)
    assert np.all(net.edges["gS"].to_numpy()[[0, 2]] == 0.32)
    assert np.all(
        net.edges["gC"].to_numpy()[[1, 3]] == 0.5
    )  # 0.5 is the default value.

    net.TestSynapse.set("gC", 0.18)
    assert np.all(net.edges["gS"].to_numpy()[[0, 2]] == 0.32)
    assert np.all(net.edges["gC"].to_numpy()[[1, 3]] == 0.18)

    net.GlutamateSynapse(1).set("gS", 0.24)
    assert net.edges["gS"][0] == 0.32
    assert net.edges["gS"][2] == 0.24
    assert np.all(net.edges["gC"].to_numpy()[[1, 3]] == 0.18)

    net.GlutamateSynapse([0, 1]).set("gS", 0.27)
    assert np.all(net.edges["gS"].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges["gC"].to_numpy()[[1, 3]] == 0.18)

    net.TestSynapse([0, 1]).set("gC", 0.21)
    assert np.all(net.edges["gS"].to_numpy()[[0, 2]] == 0.27)
    assert np.all(net.edges["gC"].to_numpy()[[1, 3]] == 0.21)


def test_shuffling_order_of_set():
    """Test whether the result is the same if the order of synapses is changed."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1])

    net1 = jx.Network([cell for _ in range(4)])
    net2 = jx.Network([cell for _ in range(4)])

    net1.cell(0).branch(0).comp(1.0).connect(
        net1.cell(1).branch(0).comp(0.1), GlutamateSynapse()
    )
    net1.cell(1).branch(0).comp(0.6).connect(
        net1.cell(2).branch(0).comp(0.7), GlutamateSynapse()
    )
    net1.cell(2).branch(0).comp(0.4).connect(
        net1.cell(3).branch(0).comp(0.3), TestSynapse()
    )
    net1.cell(3).branch(0).comp(0.1).connect(
        net1.cell(1).branch(0).comp(0.1), TestSynapse()
    )

    # Different order as for `net1`.
    net2.cell(3).branch(0).comp(0.1).connect(
        net2.cell(1).branch(0).comp(0.1), TestSynapse()
    )
    net2.cell(1).branch(0).comp(0.6).connect(
        net2.cell(2).branch(0).comp(0.7), GlutamateSynapse()
    )
    net2.cell(2).branch(0).comp(0.4).connect(
        net2.cell(3).branch(0).comp(0.3), TestSynapse()
    )
    net2.cell(0).branch(0).comp(1.0).connect(
        net2.cell(1).branch(0).comp(0.1), GlutamateSynapse()
    )

    net1.insert(HH())
    net2.insert(HH())

    for i in range(4):
        net1.cell(i).branch(0).comp(0.0).record()
        net2.cell(i).branch(0).comp(0.0).record()

    voltages1 = jx.integrate(net1, t_max=5.0)
    voltages2 = jx.integrate(net1, t_max=5.0)

    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8
