import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
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
