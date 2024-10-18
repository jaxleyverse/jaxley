# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import fully_connect
from jaxley.synapses import IonotropicSynapse


def test_subclassing_groups_cell_api():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0, 0, 1, 1])

    cell.branch([0, 3, 4]).add_to_group("subtree")

    # The following lines are made possible by PR #324.
    cell.subtree.branch(0).set("radius", 0.1)
    cell.subtree.branch(0).comp("all").make_trainable("length")

    # TODO: REMOVE THIS IS NOW ALLOWED
    # with pytest.raises(KeyError):
    #     cell.subtree.cell(0).branch("all").make_trainable("length")
    # with pytest.raises(KeyError):
    #     cell.subtree.comp(0).make_trainable("length")


def test_subclassing_groups_net_api():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1])
    net = jx.Network([cell for _ in range(10)])

    net.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    net.excitatory.cell(0).set("radius", 0.1)
    net.excitatory.cell(0).branch("all").make_trainable("length")

    # TODO: REMOVE THIS IS NOW ALLOWED
    # with pytest.raises(KeyError):
    #     cell.excitatory.branch(0).comp("all").make_trainable("length")
    # with pytest.raises(KeyError):
    #     cell.excitatory.comp("all").make_trainable("length")


def test_subclassing_groups_net_set_equivalence():
    """Test whether calling `.set` on subclasses group is same as on view."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0])
    net1 = jx.Network([cell for _ in range(10)])
    net2 = jx.Network([cell for _ in range(10)])

    net1.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    net1.excitatory.cell([0, 3]).branch(0).comp("all").set("radius", 0.14)
    net1.excitatory.cell([0, 5]).branch(1).comp("all").set("length", 0.16)
    net1.excitatory.cell("all").branch(1).comp(2).set("axial_resistivity", 1100.0)

    net2.cell([0, 3]).branch(0).comp("all").set("radius", 0.14)
    net2.cell([0, 5]).branch(1).comp("all").set("length", 0.16)
    net2.cell([0, 3, 5]).branch(1).comp(2).set("axial_resistivity", 1100.0)

    assert all(net1.nodes == net2.nodes)


def test_subclassing_groups_net_make_trainable_equivalence():
    """Test whether calling `.maek_trainable` on subclasses group is same as on view."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0])
    net1 = jx.Network([cell for _ in range(10)])
    net2 = jx.Network([cell for _ in range(10)])

    net1.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    # The new behaviour needs changing of the scope to still conform here
    # TODO: Rewrite this test / reconsider what behaviour is desired
    net1.excitatory.scope("global").cell([0, 3]).scope("local").branch(
        0
    ).make_trainable("radius")
    net1.excitatory.scope("global").cell([0, 5]).scope("local").branch(1).comp(
        "all"
    ).make_trainable("length")
    net1.excitatory.scope("global").cell("all").scope("local").branch(1).comp(
        2
    ).make_trainable("axial_resistivity")
    params1 = jnp.concatenate(jax.tree_flatten(net1.get_parameters())[0])

    net2.cell([0, 3]).branch(0).make_trainable("radius")
    net2.cell([0, 5]).branch(1).comp("all").make_trainable("length")
    net2.cell([0, 3, 5]).branch(1).comp(2).make_trainable("axial_resistivity")
    params2 = jnp.concatenate(jax.tree_flatten(net2.get_parameters())[0])
    assert jnp.array_equal(params1, params2)

    for inds1, inds2 in zip(
        net1.indices_set_by_trainables, net2.indices_set_by_trainables
    ):
        assert jnp.array_equal(inds1, inds2)


def test_subclassing_groups_net_lazy_indexing_make_trainable_equivalence():
    """Test whether groups can be indexing in a lazy way."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0])
    net1 = jx.Network([cell for _ in range(10)])
    net2 = jx.Network([cell for _ in range(10)])

    net1.cell([0, 3, 5]).add_to_group("excitatory")
    net2.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    net1.excitatory.cell([0, 3]).branch(0).make_trainable("radius")
    net1.excitatory.cell([0, 5]).branch(1).comp("all").make_trainable("length")
    net1.excitatory.cell("all").branch(1).comp(2).make_trainable("axial_resistivity")
    params1 = jnp.concatenate(jax.tree_flatten(net1.get_parameters())[0])

    # The following lines are made possible by PR #324.
    net2.excitatory[[0, 3], 0].make_trainable("radius")
    net2.excitatory[[0, 5], 1, :].make_trainable("length")
    net2.excitatory[:, 1, 2].make_trainable("axial_resistivity")
    params2 = jnp.concatenate(jax.tree_flatten(net2.get_parameters())[0])

    assert jnp.array_equal(params1, params2)

    for inds1, inds2 in zip(
        net1.indices_set_by_trainables, net2.indices_set_by_trainables
    ):
        assert jnp.array_equal(inds1, inds2)


def test_fully_connect_groups_equivalence():
    """Test whether groups can be used with `fully_connect`."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0])
    net1 = jx.Network([cell for _ in range(10)])
    net2 = jx.Network([cell for _ in range(10)])

    net1.cell([0, 3, 5]).add_to_group("layer1")
    net1.cell([6, 8]).add_to_group("layer2")

    pre = net1.layer1.cell("all")
    post = net1.layer2.cell("all")
    fully_connect(pre, post, IonotropicSynapse())

    pre = net2.cell([0, 3, 5])
    post = net2.cell([6, 8])
    fully_connect(pre, post, IonotropicSynapse())

    assert all(net1.edges == net2.edges)
