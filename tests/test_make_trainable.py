import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH, Na, K
from jaxley.synapses import GlutamateSynapse, TestSynapse


def test_make_trainable():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch(comp, nseg_per_branch).initialize()
    cell = jx.Cell(branch, parents=parents).initialize()
    cell.insert(HH())

    cell.branch(0).comp(0.0).set("length", 12.0)
    cell.branch(1).comp(1.0).set("HH_gNa", 0.2)
    assert cell.num_trainable_params == 0

    cell.branch([0, 1]).make_trainable("radius", 1.0)
    assert cell.num_trainable_params == 2
    cell.branch([0, 1]).make_trainable("length")
    cell.branch([0, 1]).make_trainable("axial_resistivity", [600.0, 700.0])
    cell.branch([0, 1]).make_trainable("HH_gNa")

    cell.get_parameters()


def test_make_trainable_network():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch(comp, nseg_per_branch).initialize()
    cell = jx.Cell(branch, parents=parents).initialize()
    cell.insert(HH())

    conns = [
        jx.Connectivity(
            GlutamateSynapse(),
            [
                jx.Connection(0, 0, 0.0, 1, 0, 0.0),
            ],
        )
    ]
    net = jx.Network([cell, cell], conns).initialize()

    cell.branch(0).comp(0.0).set("length", 12.0)
    cell.branch(1).comp(1.0).set("HH_gNa", 0.2)

    cell.branch([0, 1]).make_trainable("radius", 1.0)
    cell.branch([0, 1]).make_trainable("length")
    cell.branch([0, 1]).make_trainable("axial_resistivity", [600.0, 700.0])
    cell.branch([0, 1]).make_trainable("HH_gNa")

    cell.get_parameters()
    net.GlutamateSynapse.set("gS", 0.1)
    assert cell.num_trainable_params == 8  # `set()` is ignored.


def test_diverse_synapse_types():
    """Runs `.get_all_parameters()` and checks if the output is as expected."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=1)
    cell = jx.Cell(branch, parents=[-1])

    net = jx.Network([cell for _ in range(4)])
    for pre_ind in [0, 1]:
        for post_ind, syn in zip([2, 3], [GlutamateSynapse(), TestSynapse()]):
            pre = net.cell(pre_ind).branch(0).comp(0.0)
            post = net.cell(post_ind).branch(0).comp(0.0)
            pre.connect(post, syn)

    net.make_trainable("gS")
    net.TestSynapse([0, 1]).make_trainable("gC")
    assert net.num_trainable_params == 3

    params = net.get_parameters()

    # Modify the trainable parameters.
    params[0]["gS"] = params[0]["gS"].at[:].set(2.2)
    params[1]["gC"] = params[1]["gC"].at[0].set(3.3)
    params[1]["gC"] = params[1]["gC"].at[1].set(4.4)
    net._to_jax()
    all_parameters = net.get_all_parameters(params)

    assert np.all(all_parameters["radius"] == 1.0)
    assert np.all(all_parameters["length"] == 10.0)
    assert np.all(all_parameters["axial_resistivity"] == 5000.0)
    assert np.all(all_parameters["gS"][0] == 2.2)
    assert np.all(all_parameters["gS"][1] == 2.2)
    assert np.all(all_parameters["gC"][0] == 3.3)
    assert np.all(all_parameters["gC"][1] == 4.4)

    # Add another trainable parameter and test again.
    net.GlutamateSynapse(1).make_trainable("gS")
    assert net.num_trainable_params == 4

    params = net.get_parameters()

    # Modify the trainable parameters.
    params[2]["gS"] = params[2]["gS"].at[:].set(5.5)
    net._to_jax()
    all_parameters = net.get_all_parameters(params)
    assert np.all(all_parameters["gS"][0] == 2.2)
    assert np.all(all_parameters["gS"][1] == 5.5)




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
    net1.cell(1).branch(0).comp(0.4).insert(HH())
    params1 = get_params_all_trainable(net1)
    net2.cell(1).branch(0).comp(0.4).insert(HH())
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
    net1.cell(0).branch(1).comp(0.4).insert(HH())
    params1 = get_params_subset_trainable(net1)
    net2.cell(0).branch(1).comp(0.4).insert(HH())
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
    net._to_jax()
    return net.get_all_parameters(trainable_params=params)


def get_params_set_subset(net):
    net.cell(0).branch(1).set("HH_gNa", 0.0)
    params = net.get_parameters()
    net._to_jax()
    return net.get_all_parameters(trainable_params=params)

def get_params_all_trainable(net):
    net.cell("all").branch("all").comp("all").make_trainable("HH_gNa")
    params = net.get_parameters()
    params[0]["HH_gNa"] = params[0]["HH_gNa"].at[:].set(0.0)
    net._to_jax()
    return net.get_all_parameters(trainable_params=params)

def get_params_set(net):
    net.set("HH_gNa", 0.0)
    params = net.get_parameters()
    net._to_jax()
    return net.get_all_parameters(trainable_params=params)


def test_make_trainable_corresponds_to_set_pospischil():
    """Test whether shared parameters are also set correctly."""
    net1, net2 = build_two_networks()
    net1.cell(0).insert(Na())
    net1.insert(K())
    net1.cell("all").branch("all").comp("all").make_trainable("vt")
    params = net1.get_parameters()
    params[0]["vt"] = params[0]["vt"].at[:].set(0.05)
    net1._to_jax()
    params1 = net1.get_all_parameters(trainable_params=params)

    net2.cell(0).insert(Na())
    net2.insert(K())
    net2.cell("all").branch("all").comp("all").make_trainable("vt")
    params = net2.get_parameters()
    params[0]["vt"] = params[0]["vt"].at[:].set(0.05)
    net2._to_jax()
    params2 = net2.get_all_parameters(trainable_params=params)
    assert np.array_equal(params1["vt"], params2["vt"], equal_nan=True)
    assert np.array_equal(params1["Na_gNa"], params2["Na_gNa"], equal_nan=True)
    assert np.array_equal(params1["K_gK"], params2["K_gK"], equal_nan=True)

    # Perform test on simulation.
    net1.cell(1).branch(1).comp(0.0).record()
    net2.cell(1).branch(1).comp(0.0).record()

    current = jx.step_current(2.0, 3.0, 0.2, 0.025, 5.0)
    net1.cell(0).branch(1).comp(0.0).stimulate(current)
    net2.cell(0).branch(1).comp(0.0).stimulate(current)
    voltages1 = jx.integrate(net1)
    voltages2 = jx.integrate(net2)
    assert np.max(np.abs(voltages1 - voltages2)) < 1e-8