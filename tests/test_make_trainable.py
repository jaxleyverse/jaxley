# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from copy import copy

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH, K, Na
from jaxley.connect import connect, fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse
from jaxley.utils.cell_utils import params_to_pstate
from jaxley.utils.morph_attributes import (
    cylinder_area,
    cylinder_resistive_load,
    cylinder_volume,
)


def test_make_trainable(SimpleCell):
    """Test make_trainable."""
    cell = SimpleCell(4, 4)
    cell.insert(HH())

    cell.branch(0).loc(0.0).set("length", 12.0)
    cell.branch(1).loc(1.0).set("HH_gNa", 0.2)
    assert cell.num_trainable_params == 0

    cell.branch([0, 1]).make_trainable("radius", 1.0)
    assert cell.num_trainable_params == 2
    cell.branch([0, 1]).make_trainable("length")
    cell.branch([0, 1]).make_trainable("axial_resistivity", [600.0, 700.0])
    cell.branch([0, 1]).make_trainable("HH_gNa")

    cell.get_parameters()


def test_delete_trainables(SimpleCell):
    """Test make_trainable."""
    cell = SimpleCell(4, 4)

    cell.branch(0).loc(0.0).make_trainable("length", 12.0)
    assert cell.num_trainable_params == 1

    cell.delete_trainables()
    cell.branch(0).loc(0.0).make_trainable("length", 12.0)
    assert cell.num_trainable_params == 1

    cell.get_parameters()


def test_make_trainable_network(SimpleCell):
    """Test make_trainable."""
    cell = SimpleCell(4, 4)
    cell.insert(HH())

    net = jx.Network([cell, cell])
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )

    cell.branch(0).loc(0.0).set("length", 12.0)
    cell.branch(1).loc(1.0).set("HH_gNa", 0.2)

    cell.branch([0, 1]).make_trainable("radius", 1.0)
    cell.branch([0, 1]).make_trainable("length")
    cell.branch([0, 1]).make_trainable("axial_resistivity", [600.0, 700.0])
    cell.branch([0, 1]).make_trainable("HH_gNa")

    cell.get_parameters()
    net.IonotropicSynapse.set("IonotropicSynapse_gS", 0.1)
    assert cell.num_trainable_params == 8  # `set()` is ignored.


def test_diverse_synapse_types(SimpleNet):
    """Runs `.get_all_parameters()` and checks if the output is as expected."""
    net = SimpleNet(4, 1, 1)
    for pre_ind in [0, 1]:
        for post_ind, syn in zip([2, 3], [IonotropicSynapse(), TestSynapse()]):
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            connect(pre, post, syn)

    net.IonotropicSynapse.make_trainable("IonotropicSynapse_gS")
    net.TestSynapse.edge([0, 1]).make_trainable("TestSynapse_gC")
    assert net.num_trainable_params == 3

    params = net.get_parameters()

    # Modify the trainable parameters.
    params[0]["IonotropicSynapse_gS"] = params[0]["IonotropicSynapse_gS"].at[:].set(2.2)
    params[1]["TestSynapse_gC"] = params[1]["TestSynapse_gC"].at[0].set(3.3)
    params[1]["TestSynapse_gC"] = params[1]["TestSynapse_gC"].at[1].set(4.4)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    all_parameters = net.get_all_parameters(pstate)

    assert np.all(all_parameters["radius"] == 1.0)
    assert np.all(all_parameters["length"] == 10.0)
    assert np.all(all_parameters["axial_resistivity"] == 5000.0)
    assert np.all(all_parameters["IonotropicSynapse_gS"][0] == 2.2)
    assert np.all(all_parameters["IonotropicSynapse_gS"][1] == 2.2)
    assert np.all(all_parameters["TestSynapse_gC"][0] == 3.3)
    assert np.all(all_parameters["TestSynapse_gC"][1] == 4.4)

    # Add another trainable parameter and test again.
    net.IonotropicSynapse.edge(1).make_trainable("IonotropicSynapse_gS")
    assert net.num_trainable_params == 4

    params = net.get_parameters()

    # Modify the trainable parameters.
    params[2]["IonotropicSynapse_gS"] = params[2]["IonotropicSynapse_gS"].at[:].set(5.5)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    all_parameters = net.get_all_parameters(pstate)
    assert np.all(all_parameters["IonotropicSynapse_gS"][0] == 2.2)
    assert np.all(all_parameters["IonotropicSynapse_gS"][1] == 5.5)


def test_make_all_trainable_corresponds_to_set(SimpleNet):
    # Scenario 1.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 2.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(1).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 3.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(1).branch(0).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).branch(0).insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 4.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(1).branch(0).loc(0.4).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).branch(0).loc(0.4).insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)


def test_make_subset_trainable_corresponds_to_set(SimpleNet):
    # Scenario 1.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 2.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(0).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 3.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(0).branch(1).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).branch(1).insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 4.
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(0).branch(1).loc(0.4).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).branch(1).loc(0.4).insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)


def test_copy_node_property_to_edges(SimpleNet):
    """Test synaptic parameter sharing via `.copy_node_property_to_edges()`.

    This test does not explicitly use `make_trainable`, but
    `copy_node_property_to_edges` is an important ingredient to parameter sharing.
    """
    net = SimpleNet(6, 2, 2)
    net.insert(HH())
    net.cell(1).set("HH_gNa", 1.0)
    net.cell(0).set("radius", 0.2)
    net.cell(1).branch(0).comp(0).set("capacitance", 0.3)
    fully_connect(net.cell("all"), net.cell("all"), IonotropicSynapse())

    net.copy_node_property_to_edges("global_comp_index", "pre")
    net.copy_node_property_to_edges("global_comp_index", "post")

    net.copy_node_property_to_edges("HH_gNa", "pre")
    # Run it another time to ensure that it can be run twice.
    net.copy_node_property_to_edges("HH_gNa", "pre")
    assert "pre_HH_gNa" in net.edges.columns
    assert "post_HH_gNa" not in net.edges.columns

    # Query the second cell. Each cell has four compartments.
    edges_gna_values = net.edges.query("pre_global_comp_index > 3")
    edges_gna_values = edges_gna_values.query("pre_global_comp_index <= 7")
    assert np.all(edges_gna_values["pre_HH_gNa"] == 1.0)

    # Query the other cells. The first cell has four compartments.
    edges_gna_values = net.edges.query("pre_global_comp_index <= 3")
    assert np.all(edges_gna_values["pre_HH_gNa"] == 0.12)
    edges_gna_values = net.edges.query("pre_global_comp_index > 7")
    assert np.all(edges_gna_values["pre_HH_gNa"] == 0.12)

    # Test whether multiple properties can be copied over.
    net.copy_node_property_to_edges(["radius", "length"])
    assert "pre_radius" in net.edges.columns
    assert "post_radius" in net.edges.columns
    assert "pre_length" in net.edges.columns
    assert "post_length" in net.edges.columns

    edges_gna_values = net.edges.query("pre_global_comp_index <= 3")
    assert np.all(edges_gna_values["pre_radius"] == 0.2)

    edges_gna_values = net.edges.query("pre_global_comp_index > 3")
    assert np.all(edges_gna_values["pre_radius"] == 1.0)

    # Test whether modifying an individual compartment also takes effect.
    net.copy_node_property_to_edges(["capacitance"])
    assert "pre_capacitance" in net.edges.columns
    assert "post_capacitance" in net.edges.columns

    edges_gna_values = net.edges.query("pre_global_comp_index == 4")
    assert np.all(edges_gna_values["pre_capacitance"] == 0.3)

    edges_gna_values = net.edges.query("post_global_comp_index == 4")
    assert np.all(edges_gna_values["post_capacitance"] == 0.3)

    edges_gna_values = net.edges.query("pre_global_comp_index != 4")
    assert np.all(edges_gna_values["pre_capacitance"] == 1.0)

    edges_gna_values = net.edges.query("post_global_comp_index != 4")
    assert np.all(edges_gna_values["post_capacitance"] == 1.0)


def get_params_subset_trainable(net):
    net.cell(0).branch(1).make_trainable("HH_gNa")
    params = net.get_parameters()
    params[0]["HH_gNa"] = params[0]["HH_gNa"].at[:].set(0.0)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate)


def get_params_set_subset(net):
    net.cell(0).branch(1).set("HH_gNa", 0.0)
    params = net.get_parameters()
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate)


def get_params_all_trainable(net):
    net.cell("all").branch("all").loc("all").make_trainable("HH_gNa")
    params = net.get_parameters()
    params[0]["HH_gNa"] = params[0]["HH_gNa"].at[:].set(0.0)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate)


def get_params_set(net):
    net.set("HH_gNa", 0.0)
    params = net.get_parameters()
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate)


def test_make_trainable_corresponds_to_set_pospischil(SimpleNet):
    """Test whether shared parameters are also set correctly."""
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(0).insert(Na())
    net1.insert(K())
    net1.cell("all").branch("all").loc("all").make_trainable("vt")
    params1 = net1.get_parameters()
    params1[0]["vt"] = params1[0]["vt"].at[:].set(0.05)
    net1.to_jax()
    pstate1 = params_to_pstate(params1, net1.indices_set_by_trainables)
    all_params1 = net1.get_all_parameters(pstate1)

    net2.cell(0).insert(Na())
    net2.insert(K())
    net2.cell("all").branch("all").loc("all").make_trainable("vt")
    params2 = net2.get_parameters()
    params2[0]["vt"] = params2[0]["vt"].at[:].set(0.05)
    net2.to_jax()
    pstate2 = params_to_pstate(params2, net2.indices_set_by_trainables)
    all_params2 = net2.get_all_parameters(pstate2)
    assert np.array_equal(all_params1["vt"], all_params2["vt"], equal_nan=True)
    assert np.array_equal(all_params1["Na_gNa"], all_params2["Na_gNa"], equal_nan=True)
    assert np.array_equal(all_params1["K_gK"], all_params2["K_gK"], equal_nan=True)

    # Perform test on simulation.
    net1.cell(1).branch(1).loc(0.0).record()
    net2.cell(1).branch(1).loc(0.0).record()

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    net1.cell(0).branch(1).loc(0.0).stimulate(current)
    net2.cell(0).branch(1).loc(0.0).stimulate(current)
    voltages1 = jx.integrate(net1, params=params1)
    voltages2 = jx.integrate(net2, params=params2)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_group_trainable_corresponds_to_set():
    """Use `GroupView` and make it trainable; test if it gives the same as `set`."""

    def build_net():
        comp = jx.Compartment()
        branch = jx.Branch(comp, ncomp=4)
        cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
        net = jx.Network([cell for _ in range(4)])
        net.cell(0).add_to_group("test")
        net.cell(1).branch(2).add_to_group("test")
        return net

    net1 = build_net()

    net1.test.make_trainable("radius")
    params = net1.get_parameters()
    params[0]["radius"] = params[0]["radius"].at[:].set(2.5)
    net1.to_jax()
    pstate = params_to_pstate(params, net1.indices_set_by_trainables)
    all_parameters1 = net1.get_all_parameters(pstate)

    net2 = build_net()
    net2.test.set("radius", 2.5)
    params = net2.get_parameters()
    net2.to_jax()
    pstate = params_to_pstate(params, net2.indices_set_by_trainables)
    all_parameters2 = net2.get_all_parameters(pstate)

    assert np.allclose(all_parameters1["radius"], all_parameters2["radius"])


def test_data_set_vs_make_trainable_pospischil(SimpleNet):
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    net1.cell(0).insert(Na())
    net1.insert(K())
    net1.make_trainable("vt")
    params1 = net1.get_parameters()
    params1[0]["vt"] = params1[0]["vt"].at[:].set(0.05)
    net1.to_jax()
    pstate1 = params_to_pstate(params1, net1.indices_set_by_trainables)
    all_params1 = net1.get_all_parameters(pstate1)

    net2.cell(0).insert(Na())
    net2.insert(K())
    val = params1[0]["vt"]
    pstate = net2.cell("all").branch("all").loc("all").data_set("vt", val.item(), None)
    net2.to_jax()
    all_params2 = net2.get_all_parameters(pstate)
    assert np.array_equal(all_params1["vt"], all_params2["vt"], equal_nan=True)
    assert np.array_equal(all_params1["Na_gNa"], all_params2["Na_gNa"], equal_nan=True)
    assert np.array_equal(all_params1["K_gK"], all_params2["K_gK"], equal_nan=True)

    # Perform test on simulation.
    net1.cell(1).branch(1).loc(0.0).record()
    net2.cell(1).branch(1).loc(0.0).record()

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    net1.cell(0).branch(1).loc(0.0).stimulate(current)
    net2.cell(0).branch(1).loc(0.0).stimulate(current)
    voltages1 = jx.integrate(net1, params=params1)
    voltages2 = jx.integrate(net2, param_state=pstate)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_data_set_vs_make_trainable_network(SimpleNet):
    net1 = SimpleNet(2, 4, 1)
    net2 = SimpleNet(2, 4, 1)
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    for net in [net1, net2]:
        net.insert(HH())
        pre = net.cell(0).branch(0).loc(0.0)
        post = net.cell(0).branch(1).loc(0.9)
        connect(pre, post, IonotropicSynapse())
        pre = net.cell(0).branch(0).loc(0.1)
        post = net.cell(1).branch(1).loc(0.4)
        connect(pre, post, TestSynapse())
        pre = net.cell(1).branch(0).loc(0.3)
        post = net.cell(0).branch(1).loc(0.0)
        connect(pre, post, IonotropicSynapse())

        net.cell(0).branch(1).loc(1.0).record()
        net.cell(1).branch(0).loc(1.0).record()

        net.cell(1).branch(0).loc(0.4).stimulate(current)
        net.cell(0).branch(0).loc(0.6).stimulate(current)

    net1.make_trainable("radius", 0.9)
    net1.make_trainable("length", 0.99)
    net1.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS", 0.15)
    net1.IonotropicSynapse.edge(1).make_trainable("IonotropicSynapse_e_syn", 0.2)
    net1.TestSynapse.edge(0).make_trainable("TestSynapse_gC", 0.3)
    params1 = net1.get_parameters()

    pstate = None
    pstate = net2.data_set("radius", 0.9, pstate)
    pstate = net2.data_set("length", 0.99, pstate)
    pstate = net2.IonotropicSynapse.edge("all").data_set(
        "IonotropicSynapse_gS", 0.15, pstate
    )
    pstate = net2.IonotropicSynapse.edge(1).data_set(
        "IonotropicSynapse_e_syn", 0.2, pstate
    )
    pstate = net2.TestSynapse.edge(0).data_set("TestSynapse_gC", 0.3, pstate)

    voltages1 = jx.integrate(net1, params=params1)
    voltages2 = jx.integrate(net2, param_state=pstate)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_data_set_vector(SimpleNet):
    net = SimpleNet(2, 2, 4)
    net.set("radius", np.repeat(1.0, 16))
    with pytest.raises(ValueError):
        net.comp(range(2)).set("radius", np.repeat(1, 16))

    param_state = None
    param_state = net.data_set("length", np.repeat(1.0, 16), param_state)
    with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
        param_state = net.comp(range(2)).data_set(
            "radius", np.array([0.1, 0.2]), param_state
        )
    net.record("v")
    jx.integrate(net, t_max=10, param_state=param_state)


def test_make_states_trainable_api(SimpleNet):
    net = SimpleNet(2, 2, 4)
    net.insert(HH())
    net.cell(0).branch(0).comp(0).record()

    net.cell("all").make_trainable("v")
    net.make_trainable("HH_h")
    net.cell(0).branch("all").make_trainable("HH_m")
    net.cell(0).branch("all").comp("all").make_trainable("HH_n")

    def simulate(params):
        return jx.integrate(net, params=params, t_max=10.0)

    parameters = net.get_parameters()
    v = simulate(parameters)
    assert np.invert(np.any(np.isnan(v))), "Found NaN in voltage."


def test_write_trainables(SimpleNet):
    """Test whether `write_trainables()` gives the same result as using the trainables."""
    net = SimpleNet(2, 2, 4)
    connect(
        net.cell(0).branch(0).loc(0.9),
        net.cell(1).branch(1).loc(0.1),
        IonotropicSynapse(),
    )
    connect(
        net.cell(1).branch(0).loc(0.1),
        net.cell(0).branch(1).loc(0.3),
        TestSynapse(),
    )
    connect(
        net.cell(0).branch(0).loc(0.3),
        net.cell(0).branch(1).loc(0.6),
        TestSynapse(),
    )
    connect(
        net.cell(1).branch(0).loc(0.6),
        net.cell(1).branch(1).loc(0.9),
        IonotropicSynapse(),
    )
    net.insert(HH())
    net.cell(0).branch(0).comp(0).record()
    net.cell(1).branch(0).comp(0).record()
    net.cell(0).branch(0).comp(0).stimulate(
        jx.step_current(i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0)
    )

    net.make_trainable("radius")
    net.cell(0).make_trainable("length")
    net.cell("all").make_trainable("axial_resistivity")
    net.cell("all").branch("all").make_trainable("HH_gNa")
    net.cell("all").branch("all").make_trainable("HH_m")
    net.make_trainable("IonotropicSynapse_gS")
    net.make_trainable("IonotropicSynapse_s")
    net.select(edges="all").make_trainable("TestSynapse_gC")
    net.select(edges="all").make_trainable("TestSynapse_c")
    net.cell(0).branch(0).comp(0).make_trainable("radius")

    params = net.get_parameters()

    # Now, we manually modify the parameters.
    for p in params:
        for key in p:
            p[key] = p[key].at[:].set(np.random.rand())

    # Test whether voltages match.
    v1 = jx.integrate(net, params=params)

    previous_nodes = copy(net.nodes)
    previous_edges = copy(net.edges)
    net.write_trainables(params)
    v2 = jx.integrate(net)
    assert np.max(np.abs(v1 - v2)) < 1e-8

    # Test whether nodes and edges actually changed.
    assert not net.nodes.equals(previous_nodes)
    assert not net.edges.equals(previous_edges)

    # Test whether `View` raises with `write_trainables()`.
    with pytest.raises(AssertionError):
        net.cell(0).write_trainables(params)

    # Test whether synapse view raises an error.
    with pytest.raises(AssertionError):
        net.select(edges=[0, 2, 3]).write_trainables(params)


def test_param_sharing_w_different_group_sizes():
    # test if make_trainable corresponds to set
    branch1 = jx.Branch(ncomp=6)
    branch1.nodes["controlled_by_param"] = np.array([0, 0, 0, 1, 1, 2])
    branch1.make_trainable("radius")
    assert branch1.num_trainable_params == 3

    # make trainable
    params = branch1.get_parameters()
    params[0]["radius"] = params[0]["radius"].at[:].set([2, 3, 4])
    branch1.to_jax()
    pstate = params_to_pstate(params, branch1.indices_set_by_trainables)
    params1 = branch1.get_all_parameters(pstate)

    # set
    branch2 = jx.Branch(ncomp=6)
    branch2.set("radius", np.array([2, 2, 2, 3, 3, 4]))
    params = branch2.get_parameters()
    branch2.to_jax()
    pstate = params_to_pstate(params, branch2.indices_set_by_trainables)
    params2 = branch2.get_all_parameters(pstate)

    assert np.array_equal(params1["radius"], params2["radius"], equal_nan=True)


def test_updates_of_membrane_area():
    """Runs `.set("radius")` and checks whether this updates the membrane area."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    branch.comp(1).set("radius", 5.5)
    radiuses = np.asarray([1.0, 5.5, 1.0, 1.0])
    lengths = np.asarray([10.0, 10.0, 10.0, 10.0])
    assert np.array_equal(
        branch.nodes.area.to_numpy(), cylinder_area(lengths, radiuses)
    )
    assert np.array_equal(
        branch.nodes.volume.to_numpy(), cylinder_volume(lengths, radiuses)
    )
    assert np.array_equal(
        branch.nodes.resistive_load_in.to_numpy(),
        cylinder_resistive_load(lengths / 2, radiuses),
    )
    assert np.array_equal(
        branch.nodes.resistive_load_out.to_numpy(),
        cylinder_resistive_load(lengths / 2, radiuses),
    )

    branch.comp(2).set("length", 22.0)
    radiuses = np.asarray([1.0, 5.5, 1.0, 1.0])
    lengths = np.asarray([10.0, 10.0, 22.0, 10.0])
    assert np.array_equal(
        branch.nodes.area.to_numpy(), cylinder_area(lengths, radiuses)
    )
    assert np.array_equal(
        branch.nodes.volume.to_numpy(), cylinder_volume(lengths, radiuses)
    )
    assert np.array_equal(
        branch.nodes.resistive_load_in.to_numpy(),
        cylinder_resistive_load(lengths / 2, radiuses),
    )
    assert np.array_equal(
        branch.nodes.resistive_load_out.to_numpy(),
        cylinder_resistive_load(lengths / 2, radiuses),
    )


def test_updates_to_membrane_area_equal_updates_to_radius():
    comp = jx.Compartment()
    branch1 = jx.Branch(comp, 4)
    branch1.comp(1).set("radius", 5.5)
    branch1.comp(2).set("length", 22.0)

    radiuses = np.asarray([1.0, 5.5, 1.0, 1.0])
    lengths = np.asarray([10.0, 10.0, 22.0, 10.0])

    branch2 = jx.Branch(comp, 4)
    area = cylinder_area(lengths, radiuses)
    volume = cylinder_volume(lengths, radiuses)
    r_in = cylinder_resistive_load(lengths / 2, radiuses)
    r_out = cylinder_resistive_load(lengths / 2, radiuses)
    branch2.set("area", area)
    branch2.set("volume", volume)
    branch2.set("resistive_load_in", r_in)
    branch2.set("resistive_load_out", r_out)

    for branch in [branch1, branch2]:
        branch.comp(3).stimulate(jx.step_current(2.0, 5.0, 0.01, 0.025, 10.0))
        branch.record()

    v1 = jx.integrate(branch1)
    v2 = jx.integrate(branch2)
    assert np.max(np.abs(v1 - v2)) < 1e-8


def test_whether_swc_after_running_set_on_radius_equals_cylinder():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(
        dirname, "swc_files", "morph_variable_radiuses_within_branch.swc"
    )

    cell1 = jx.read_swc(fname, ncomp=1)
    cell2 = jx.Cell()
    cell3 = jx.read_swc(fname, ncomp=1)
    cell3.make_trainable("radius")
    cell3.make_trainable("length")
    params = cell3.get_parameters()

    for cell in [cell1, cell2]:
        cell.set("length", cell1.nodes.length)
        cell.set("radius", cell1.nodes.radius)

    for cell in [cell1, cell2, cell3]:
        cell.stimulate(jx.step_current(2.0, 5.0, 0.01, 0.025, 10.0))
        cell.record()

    v1 = jx.integrate(cell1)
    v2 = jx.integrate(cell2)
    v3 = jx.integrate(cell3, params=params)
    assert np.max(np.abs(v1 - v2)) < 1e-8
    assert np.max(np.abs(v1 - v3)) < 1e-8
