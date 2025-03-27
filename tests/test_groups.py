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


def test_subclassing_groups_cell_api(SimpleCell):
    """Test whether `cell.group.branch(0)` works."""
    cell = SimpleCell(5, 4)

    cell.branch([0, 3, 4]).add_to_group("subtree")

    # The following lines are made possible by PR #324.
    cell.subtree.branch(0).set("radius", 0.1)
    cell.subtree.branch(0).comp("all").make_trainable("length")


def test_stacking_groups_api(SimpleNet):
    """Test whether groups can be intersected (e.g., `net.exc.soma`)."""
    net = SimpleNet(5, 2, 4)

    net.cell([0, 1, 2]).add_to_group("exc")
    net.cell("all").branch(0).comp(0).add_to_group("soma")

    assert len(net.exc.soma.nodes) == 3
    assert len(net.exc.soma.cell(1).nodes) == 1

    net.exc.soma.make_trainable("length")
    params = net.get_parameters()
    assert params[0]["length"] == jnp.asarray([10.0])


def test_subclassing_groups_net_api(SimpleNet):
    net = SimpleNet(10, 2, 4)

    net.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    net.excitatory.cell(0).set("radius", 0.1)
    net.excitatory.cell(0).branch("all").make_trainable("length")


def test_subclassing_groups_net_set_equivalence(SimpleNet):
    """Test whether calling `.set` on subclasses group is same as on view."""
    net1 = SimpleNet(10, 2, 4)
    net2 = SimpleNet(10, 2, 4)

    net1.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    net1.excitatory.cell([0, 3]).branch(0).comp("all").set("radius", 0.14)
    net1.excitatory.cell([0, 5]).branch(1).comp("all").set("length", 0.16)
    net1.excitatory.cell("all").branch(1).comp(2).set("axial_resistivity", 1100.0)

    net2.cell([0, 3]).branch(0).comp("all").set("radius", 0.14)
    net2.cell([0, 5]).branch(1).comp("all").set("length", 0.16)
    net2.cell([0, 3, 5]).branch(1).comp(2).set("axial_resistivity", 1100.0)

    net1.nodes.drop(columns=["excitatory"], inplace=True)
    assert all(net1.nodes == net2.nodes)


def test_subclassing_groups_net_make_trainable_equivalence(SimpleNet):
    """Test whether calling `.maek_trainable` on subclasses group is same as on view."""
    net1 = SimpleNet(10, 2, 4)
    net2 = SimpleNet(10, 2, 4)

    net1.cell([0, 3, 5]).add_to_group("excitatory")

    # The following lines are made possible by PR #324.
    # The new behaviour needs changing of the scope to still conform here
    # TODO FROM #447: Rewrite this test / reconsider what behaviour is desired
    net1.excitatory.scope("global").cell([0, 3]).scope("local").branch(
        0
    ).make_trainable("radius")
    net1.excitatory.scope("global").cell([0, 5]).scope("local").branch(1).comp(
        "all"
    ).make_trainable("length")
    net1.excitatory.scope("global").cell("all").scope("local").branch(1).comp(
        2
    ).make_trainable("axial_resistivity")
    params1 = jnp.concatenate(jax.tree_util.tree_flatten(net1.get_parameters())[0])

    net2.cell([0, 3]).branch(0).make_trainable("radius")
    net2.cell([0, 5]).branch(1).comp("all").make_trainable("length")
    net2.cell([0, 3, 5]).branch(1).comp(2).make_trainable("axial_resistivity")
    params2 = jnp.concatenate(jax.tree_util.tree_flatten(net2.get_parameters())[0])
    assert jnp.array_equal(params1, params2)

    for inds1, inds2 in zip(
        net1.indices_set_by_trainables, net2.indices_set_by_trainables
    ):
        assert jnp.array_equal(inds1, inds2)


def test_fully_connect_groups_equivalence(SimpleNet):
    """Test whether groups can be used with `fully_connect`."""
    net1 = SimpleNet(10, 2, 4)
    net2 = SimpleNet(10, 2, 4)

    net1.cell([0, 3, 5]).add_to_group("layer1")
    net1.cell([6, 8]).add_to_group("layer2")

    pre = net1.layer1.cell("all")
    post = net1.layer2.cell("all")
    fully_connect(pre, post, IonotropicSynapse())

    pre = net2.cell([0, 3, 5])
    post = net2.cell([6, 8])
    fully_connect(pre, post, IonotropicSynapse())

    assert all(net1.edges == net2.edges)


def test_set_ncomp_changes_groups(SimpleCell):
    """Test whether groups get updated appropriately after `set_ncomp`."""
    cell = SimpleCell(3, 4)
    cell.branch(0).add_to_group("exc")
    cell.branch(0).set_ncomp(1)
    assert len(cell.exc.nodes) == 1

    cell.branch(1).add_to_group("exc")
    cell.branch(0).set_ncomp(2)
    assert len(cell.exc.nodes) == 6  # 2 from branch(0) and 4 from branch(1).
