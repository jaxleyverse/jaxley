# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH, K, Na
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse, TestSynapse
from jaxley.utils.cell_utils import params_to_pstate


def test_make_trainable():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    cell = jx.Cell(branch, parents=parents)
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


def test_delete_trainables():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    cell = jx.Cell(branch, parents=parents)

    cell.branch(0).loc(0.0).make_trainable("length", 12.0)
    assert cell.num_trainable_params == 1

    cell.delete_trainables()
    cell.branch(0).loc(0.0).make_trainable("length", 12.0)
    assert cell.num_trainable_params == 1

    cell.get_parameters()


def test_make_trainable_network():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    cell = jx.Cell(branch, parents=parents)
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


def test_diverse_synapse_types():
    """Runs `.get_all_parameters()` and checks if the output is as expected."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=1)
    cell = jx.Cell(branch, parents=[-1])

    net = jx.Network([cell for _ in range(4)])
    for pre_ind in [0, 1]:
        for post_ind, syn in zip([2, 3], [IonotropicSynapse(), TestSynapse()]):
            pre = net.cell(pre_ind).branch(0).loc(0.0)
            post = net.cell(post_ind).branch(0).loc(0.0)
            connect(pre, post, syn)

    net.IonotropicSynapse.make_trainable("IonotropicSynapse_gS")
    net.TestSynapse([0, 1]).make_trainable("TestSynapse_gC")
    assert net.num_trainable_params == 3

    params = net.get_parameters()

    # Modify the trainable parameters.
    params[0]["IonotropicSynapse_gS"] = params[0]["IonotropicSynapse_gS"].at[:].set(2.2)
    params[1]["TestSynapse_gC"] = params[1]["TestSynapse_gC"].at[0].set(3.3)
    params[1]["TestSynapse_gC"] = params[1]["TestSynapse_gC"].at[1].set(4.4)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    all_parameters = net.get_all_parameters(pstate, voltage_solver="jaxley.thomas")

    assert np.all(all_parameters["radius"] == 1.0)
    assert np.all(all_parameters["length"] == 10.0)
    assert np.all(all_parameters["axial_resistivity"] == 5000.0)
    assert np.all(all_parameters["IonotropicSynapse_gS"][0] == 2.2)
    assert np.all(all_parameters["IonotropicSynapse_gS"][1] == 2.2)
    assert np.all(all_parameters["TestSynapse_gC"][0] == 3.3)
    assert np.all(all_parameters["TestSynapse_gC"][1] == 4.4)

    # Add another trainable parameter and test again.
    net.IonotropicSynapse(1).make_trainable("IonotropicSynapse_gS")
    assert net.num_trainable_params == 4

    params = net.get_parameters()

    # Modify the trainable parameters.
    params[2]["IonotropicSynapse_gS"] = params[2]["IonotropicSynapse_gS"].at[:].set(5.5)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    all_parameters = net.get_all_parameters(pstate, voltage_solver="jaxley.thomas")
    assert np.all(all_parameters["IonotropicSynapse_gS"][0] == 2.2)
    assert np.all(all_parameters["IonotropicSynapse_gS"][1] == 5.5)


def test_make_all_trainable_corresponds_to_set():
    # Scenario 1.
    net1, net2 = build_two_networks()
    net1.insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 2.
    net1, net2 = build_two_networks()
    net1.cell(1).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 3.
    net1, net2 = build_two_networks()
    net1.cell(1).branch(0).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).branch(0).insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 4.
    net1, net2 = build_two_networks()
    net1.cell(1).branch(0).loc(0.4).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).branch(0).loc(0.4).insert(HH())
    params2 = get_params_set(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)


def test_make_subset_trainable_corresponds_to_set():
    # Scenario 1.
    net1, net2 = build_two_networks()
    net1.insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 2.
    net1, net2 = build_two_networks()
    net1.cell(0).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 3.
    net1, net2 = build_two_networks()
    net1.cell(0).branch(1).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).branch(1).insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)

    # Scenario 4.
    net1, net2 = build_two_networks()
    net1.cell(0).branch(1).loc(0.4).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).branch(1).loc(0.4).insert(HH())
    params2 = get_params_set_subset(net2)
    assert np.array_equal(params1["HH_gNa"], params2["HH_gNa"], equal_nan=True)


def build_two_networks():
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    cell = jx.Cell(branch, parents=[-1, 0])
    net1 = jx.Network([cell, cell])
    net2 = jx.Network([cell, cell])
    return net1, net2


def get_params_subset_trainable(net):
    net.cell(0).branch(1).make_trainable("HH_gNa")
    params = net.get_parameters()
    params[0]["HH_gNa"] = params[0]["HH_gNa"].at[:].set(0.0)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate, voltage_solver="jaxley.thomas")


def get_params_set_subset(net):
    net.cell(0).branch(1).set("HH_gNa", 0.0)
    params = net.get_parameters()
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate, voltage_solver="jaxley.thomas")


def get_params_all_trainable(net):
    net.cell("all").branch("all").loc("all").make_trainable("HH_gNa")
    params = net.get_parameters()
    params[0]["HH_gNa"] = params[0]["HH_gNa"].at[:].set(0.0)
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate, voltage_solver="jaxley.thomas")


def get_params_set(net):
    net.set("HH_gNa", 0.0)
    params = net.get_parameters()
    net.to_jax()
    pstate = params_to_pstate(params, net.indices_set_by_trainables)
    return net.get_all_parameters(pstate, voltage_solver="jaxley.thomas")


def test_make_trainable_corresponds_to_set_pospischil():
    """Test whether shared parameters are also set correctly."""
    net1, net2 = build_two_networks()
    net1.cell(0).insert(Na())
    net1.insert(K())
    net1.cell("all").branch("all").loc("all").make_trainable("vt")
    params1 = net1.get_parameters()
    params1[0]["vt"] = params1[0]["vt"].at[:].set(0.05)
    net1.to_jax()
    pstate1 = params_to_pstate(params1, net1.indices_set_by_trainables)
    all_params1 = net1.get_all_parameters(pstate1, voltage_solver="jaxley.thomas")

    net2.cell(0).insert(Na())
    net2.insert(K())
    net2.cell("all").branch("all").loc("all").make_trainable("vt")
    params2 = net2.get_parameters()
    params2[0]["vt"] = params2[0]["vt"].at[:].set(0.05)
    net2.to_jax()
    pstate2 = params_to_pstate(params2, net2.indices_set_by_trainables)
    all_params2 = net2.get_all_parameters(pstate2, voltage_solver="jaxley.thomas")
    assert np.array_equal(all_params1["vt"], all_params2["vt"], equal_nan=True)
    assert np.array_equal(all_params1["Na_gNa"], all_params2["Na_gNa"], equal_nan=True)
    assert np.array_equal(all_params1["K_gK"], all_params2["K_gK"], equal_nan=True)

    # Perform test on simulation.
    net1.cell(1).branch(1).loc(0.0).record()
    net2.cell(1).branch(1).loc(0.0).record()

    current = jx.step_current(2.0, 3.0, 0.2, 0.025, 5.0)
    net1.cell(0).branch(1).loc(0.0).stimulate(current)
    net2.cell(0).branch(1).loc(0.0).stimulate(current)
    voltages1 = jx.integrate(net1, params=params1)
    voltages2 = jx.integrate(net2, params=params2)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_group_trainable_corresponds_to_set():
    """Use `GroupView` and make it trainable; test if it gives the same as `set`."""

    def build_net():
        comp = jx.Compartment()
        branch = jx.Branch(comp, nseg=4)
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
    all_parameters1 = net1.get_all_parameters(pstate, voltage_solver="jaxley.thomas")

    net2 = build_net()
    net2.test.set("radius", 2.5)
    params = net2.get_parameters()
    net2.to_jax()
    pstate = params_to_pstate(params, net2.indices_set_by_trainables)
    all_parameters2 = net2.get_all_parameters(pstate, voltage_solver="jaxley.thomas")

    assert np.allclose(all_parameters1["radius"], all_parameters2["radius"])


def test_data_set_vs_make_trainable_pospischil():
    net1, net2 = build_two_networks()
    net1.cell(0).insert(Na())
    net1.insert(K())
    net1.make_trainable("vt")
    params1 = net1.get_parameters()
    params1[0]["vt"] = params1[0]["vt"].at[:].set(0.05)
    net1.to_jax()
    pstate1 = params_to_pstate(params1, net1.indices_set_by_trainables)
    all_params1 = net1.get_all_parameters(pstate1, voltage_solver="jaxley.thomas")

    net2.cell(0).insert(Na())
    net2.insert(K())
    val = params1[0]["vt"]
    pstate = net2.cell("all").branch("all").loc("all").data_set("vt", val.item(), None)
    net2.to_jax()
    all_params2 = net2.get_all_parameters(pstate, voltage_solver="jaxley.thomas")
    assert np.array_equal(all_params1["vt"], all_params2["vt"], equal_nan=True)
    assert np.array_equal(all_params1["Na_gNa"], all_params2["Na_gNa"], equal_nan=True)
    assert np.array_equal(all_params1["K_gK"], all_params2["K_gK"], equal_nan=True)

    # Perform test on simulation.
    net1.cell(1).branch(1).loc(0.0).record()
    net2.cell(1).branch(1).loc(0.0).record()

    current = jx.step_current(2.0, 3.0, 0.2, 0.025, 5.0)
    net1.cell(0).branch(1).loc(0.0).stimulate(current)
    net2.cell(0).branch(1).loc(0.0).stimulate(current)
    voltages1 = jx.integrate(net1, params=params1)
    voltages2 = jx.integrate(net2, param_state=pstate)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_data_set_vs_make_trainable_network():
    net1, net2 = build_two_networks()
    current = jx.step_current(0.1, 4.0, 0.1, 0.025, 5.0)
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
    net1.IonotropicSynapse("all").make_trainable("IonotropicSynapse_gS", 0.15)
    net1.IonotropicSynapse(1).make_trainable("IonotropicSynapse_e_syn", 0.2)
    net1.TestSynapse(0).make_trainable("TestSynapse_gC", 0.3)
    params1 = net1.get_parameters()

    pstate = None
    pstate = net2.data_set("radius", 0.9, pstate)
    pstate = net2.data_set("length", 0.99, pstate)
    pstate = net2.IonotropicSynapse("all").data_set(
        "IonotropicSynapse_gS", 0.15, pstate
    )
    pstate = net2.IonotropicSynapse(1).data_set("IonotropicSynapse_e_syn", 0.2, pstate)
    pstate = net2.TestSynapse(0).data_set("TestSynapse_gC", 0.3, pstate)

    voltages1 = jx.integrate(net1, params=params1)
    voltages2 = jx.integrate(net2, param_state=pstate)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8


def test_make_states_trainable_api():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0])
    net = jx.Network([cell for _ in range(2)])
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
