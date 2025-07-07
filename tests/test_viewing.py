# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy

import jax
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.modules.base import View
from jaxley.synapses import TestSynapse
from jaxley.utils.cell_utils import loc_of_index, local_index_of_loc
from jaxley.utils.misc_utils import cumsum_leading_zero


def test_getitem(SimpleBranch, SimpleCell, SimpleNet):
    branch = SimpleBranch(4)
    cell = SimpleCell(3, 4)
    net = SimpleNet(3, 3, 4)

    # test API equivalence
    assert all(net.cell(0).branch(0).show() == net[0, 0].show())
    assert all(net.cell([0, 1]).branch(0).show() == net[[0, 1], 0].show())
    assert all(net.cell(0).branch([0, 1]).show() == net[0, [0, 1]].show())
    assert all(net.cell([0, 1]).branch([0, 1]).show() == net[[0, 1], [0, 1]].show())

    assert all(net.cell(0).branch(0).comp(0).show() == net[0, 0, 0].show())
    assert all(cell.branch(0).comp(0).show() == cell[0, 0].show())
    assert all(cell.branch(0).show() == cell[0].show())

    # test indexing of comps
    assert branch[:2]
    assert cell[:2, :2]
    assert net[:2, :2, :2]

    # test iterability
    for cell in net.cells:
        pass

    for cell in net.cells:
        for branch in cell.branches:
            for comp in branch.comps:
                pass

    for comp in net[0, 0].comps:
        pass


def test_loc_v_comp(SimpleBranch):
    branch = SimpleBranch(4)
    ncomps = branch.ncomp_per_branch
    branch_ind = 0

    assert np.all(branch.comp(0).show() == branch.loc(0.0).show())
    assert np.all(branch.comp(3).show() == branch.loc(1.0).show())

    inferred_loc = loc_of_index(2, branch_ind, ncomps)
    assert np.all(branch.loc(inferred_loc).show() == branch.comp(2).show())

    inferred_ind = local_index_of_loc(0.4, branch_ind, ncomps)
    assert np.all(branch.comp(inferred_ind).show() == branch.loc(0.4).show())


def test_shape(SimpleComp, SimpleBranch, SimpleCell, SimpleNet):
    comp = SimpleComp()
    branch = SimpleBranch(4)
    cell = SimpleCell(3, 4)
    net = SimpleNet(3, 3, 4)

    assert net.shape == (3, 3 * 3, 3 * 3 * 4)
    assert cell.shape == (3, 3 * 4)
    assert branch.shape == (4,)
    assert comp.shape == ()

    assert net.cell("all").shape == net.shape
    assert cell.branch("all").shape == cell.shape

    assert net.cell("all").shape == (3, 3 * 3, 3 * 3 * 4)
    assert net.cell("all").branch("all").shape == (3, 3 * 3, 3 * 3 * 4)
    assert net.cell("all").branch("all").comp("all").shape == (3, 3 * 3, 3 * 3 * 4)

    assert net.cell(0).shape == (1, 3, 3 * 4)
    assert net.cell(0).branch(0).shape == (1, 1, 4)
    assert net.cell(0).branch(0).comp(0).shape == (1, 1, 1)


def test_set_and_insert(SimpleBranch, SimpleCell, SimpleNet):
    branch = SimpleBranch(4)
    cell = SimpleCell(5, 4)
    net = SimpleNet(5, 5, 4)
    net1 = deepcopy(net)
    net2 = deepcopy(net)
    net3 = deepcopy(net)
    net4 = deepcopy(net)

    # insert multiple
    net1.cell([0, 1]).branch(0).insert(HH())
    net1.cell(1).branch([0, 1]).insert(HH())
    net1.cell([2, 3]).branch([2, 3]).insert(HH())
    net1.cell(4).branch(4).comp(0).insert(HH())

    net2[[0, 1], 0].insert(HH())
    net2[1, [0, 1]].insert(HH())
    net2[[2, 3], [2, 3]].insert(HH())
    net2[4, 4, 0].insert(HH())

    # set multiple
    net1.cell([0, 1]).branch(0).set("length", 2.0)
    net1.cell(1).branch([0, 1]).set("length", 2.0)
    net1.cell([2, 3]).branch([2, 3]).set("length", 2.0)
    net1.cell(4).branch(4).comp(0).set("length", 2.0)

    net2[[0, 1], 0].set("length", 2.0)
    net2[1, [0, 1]].set("length", 2.0)
    net2[[2, 3], [2, 3]].set("length", 2.0)
    net2[4, 4, 0].set("length", 2.0)

    # insert / set at different levels
    net3.insert(HH())  # insert at net level
    net3.cell(0).insert(HH())  # insert at cell level
    net3.cell(2).branch(2).insert(HH())  # insert at cell level
    net3.cell(4).branch(3).comp(0).insert(HH())  # insert at cell level

    net3.set("length", 2.0)
    net3.cell(0).set("length", 2.0)
    net3.cell(2).branch(2).set("length", 2.0)
    net3.cell(4).branch(3).comp(0).set("length", 2.0)

    net4.insert(HH())
    net4.cell(0).insert(HH())
    net4.cell(2).branch(2).insert(HH())
    net4.cell(4).branch(3).comp(0).insert(HH())

    net4.set("length", 2.0)
    net4.cell(0).set("length", 2.0)
    net4.cell(2).branch(2).set("length", 2.0)
    net4.cell(4).branch(3).comp(0).set("length", 2.0)

    assert all(net1.show() == net2.show())
    assert all(net3.show() == net4.show())

    # insert at into a branch
    branch1 = deepcopy(branch)
    branch2 = deepcopy(branch)

    branch1.comp(0).insert(HH())
    branch2[0].insert(HH())
    assert all(branch1.show() == branch2.show())

    # test insert multiple stimuli
    single_current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    batch_of_currents = np.vstack([single_current for _ in range(4)])

    cell1 = deepcopy(cell)
    cell2 = deepcopy(cell)

    cell1.branch(0).stimulate(single_current)
    cell1.branch(1).comp(0).stimulate(single_current)
    cell1.branch(0).stimulate(batch_of_currents)
    cell1.branch(0).record("v")

    cell2[0].stimulate(single_current)
    cell2[1].comp(0).stimulate(single_current)
    cell2[0].stimulate(batch_of_currents)
    cell2.branch(0).record("v")

    assert np.all(cell1.externals["i"] == cell2.externals["i"])
    assert np.all(cell1.external_inds["i"] == cell2.external_inds["i"])
    assert np.all(cell1.recordings == cell2.recordings)


def test_local_indexing(SimpleNet):
    net = SimpleNet(2, 5, 4)

    local_idxs = net.nodes[
        ["local_cell_index", "local_branch_index", "local_comp_index"]
    ]
    idx_cols = ["global_cell_index", "global_branch_index", "global_comp_index"]
    global_index = 0
    for cell_idx in range(2):
        for branch_idx in range(5):
            for comp_idx in range(4):
                assert np.all(
                    local_idxs.iloc[global_index] == [cell_idx, branch_idx, comp_idx]
                )
                global_index += 1


def test_indexing_a_compartment_of_many_branches(SimpleBranch):
    branch1 = SimpleBranch(ncomp=3)
    branch2 = SimpleBranch(ncomp=4)
    branch3 = SimpleBranch(ncomp=5)
    cell1 = jx.Cell([branch1, branch2, branch3], parents=[-1, 0, 0])
    cell2 = jx.Cell([branch3, branch2], parents=[-1, 0])
    net = jx.Network([cell1, cell2])

    # Indexing a single compartment of multiple branches is not supported with `loc`.
    # TODO FROM #447: Reevaluate what kind of indexing is allowed and which is not!
    # with pytest.raises(NotImplementedError):
    #     net.cell("all").branch("all").loc(0.0)
    # with pytest.raises(NotImplementedError):
    #     net.cell(0).branch("all").loc(0.0)
    # with pytest.raises(NotImplementedError):
    #     net.cell("all").branch(0).loc(0.0)

    # Indexing a single compartment of multiple branches is still supported with `comp`.
    net.cell("all").branch("all").comp(0)
    net.cell(0).branch("all").comp(0)
    net.cell("all").branch(0).comp(0)

    # Indexing many single compartment of multiple branches is always supported.
    net.cell("all").branch("all").loc("all")
    net.cell(0).branch("all").loc("all")
    net.cell("all").branch(0).loc("all")


# make sure all attrs in module also have a corresponding attr in view
def test_view_attrs(SimpleComp, SimpleBranch, SimpleCell, SimpleNet):
    """Check if all attributes of Module have a corresponding attribute in View.

    To ensure that View behaves like a Module as much as possible, View should support
    all attributes of Module. This test checks if all attributes of Module have a
    corresponding attribute in View. Also checks if the types of the attributes match.
    """
    # attributes of Module that do not have to exist in View
    exceptions = ["view"]

    # TODO: Types are inconsistent between different Modules
    exceptions += ["_cumsum_nbranches"]

    # TODO FROM #447: should be added to View in the future
    exceptions += [
        "_internal_node_inds",
        "_par_inds",
        "_child_inds",
        "_child_belongs_to_branchpoint",
        "_solve_indexer",
        "_comp_edges",
        "_n_nodes",
        "_data_inds",
        "_indices_jax_spsolve",
        "_indptr_jax_spsolve",
        "_off_diagonal_inds",
        "_n_nodes",
        "_data_inds",
        "_indices_jax_spsolve",
        "_indptr_jax_spsolve",
        "_off_diagonal_inds",
        "_dhs_solve_indexer",
        "_solver_device",
    ]  # for base/comp
    exceptions += ["comb_children"]  # for cell
    exceptions += [
        "_cells_list",
        "_cumsum_nbranchpoints_per_cell",
        "_cumsum_ncomp_per_cell",
    ]  # for network

    for module in [
        SimpleComp(),
        SimpleBranch(2),
        SimpleCell(2, 3),
        SimpleNet(2, 2, 3, connect=True),
    ]:
        for name, attr in module.__dict__.items():
            if name not in exceptions:
                # check if attr is in view
                view = View(module)
                assert hasattr(view, name), f"View missing attribute: {name}"
                # check if types match
                assert type(getattr(module, name)) == type(
                    getattr(view, name)
                ), f"Type mismatch: {name}, Module type: {type(getattr(module, name))}, View type: {type(getattr(view, name))}"


def test_view_supported_index_types(SimpleComp, SimpleBranch, SimpleCell, SimpleNet):
    """Check if different ways to index into Modules/Views work correctly."""
    # test int, range, slice, list, np.array, pd.Index

    for module in [
        SimpleComp(),
        SimpleBranch(4),
        SimpleCell(3, 4),
        SimpleNet(2, 3, 4),
    ]:
        index_types = [
            0,
            range(3),
            slice(0, 3),
            [0, 1, 2],
            np.array([0, 1, 2]),
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int64),
            pd.Index([0, 1, 2]),
            pd.Index([0, 1, 2]).to_numpy(),
            np.array([True, False, True, False] * 100)[: len(module.nodes)],
        ]

        # comp.comp is not allowed
        all_inds = module.nodes.index.to_numpy()
        if not isinstance(module, jx.Compartment):
            # `_reformat_index` should always return a np.ndarray
            for index in index_types:
                assert isinstance(
                    module._reformat_index(index), np.ndarray
                ), f"Failed for {type(index)}"

                # test indexing into module and view
                assert module.comp(index), f"Failed for {type(index)}"
                assert View(module).comp(index), f"Failed for {type(index)}"

                expected_inds = all_inds[index]
                assert np.all(module.select(nodes=index).nodes.index == expected_inds)

            # for loc test float and list of floats
            assert module.loc(0.0), "Failed for float"
            assert module.loc([0.0, 0.5, 1.0]), "Failed for List[float]"
        else:
            with pytest.raises(AssertionError):
                module.comp(0)

        if isinstance(module, jx.Network):
            connect(module[0, 0, :], module[1, 0, :], TestSynapse())
            all_inds = module.edges.index.to_numpy()
            for index in index_types[:-1] + [np.array([True, False, True, False])]:
                expected_inds = all_inds[index]
                assert np.all(module.select(edges=index).edges.index == expected_inds)


def test_select(SimpleNet):
    """Ensure `select` works correctly and returns expected View of Modules."""
    net = SimpleNet(3, 3, 2, connect=False)
    connect(net[0, 0, :], net[1, 0, :], TestSynapse())

    np.random.seed(0)

    # select only nodes
    inds = np.random.choice(net.nodes.index, replace=False, size=5)
    view = net.select(nodes=inds)
    assert np.all(view.nodes.index == inds), "Selecting nodes by index failed"

    # select only edges
    inds = np.random.choice(net.edges.index, replace=False, size=2)
    view = net.select(edges=inds)
    assert np.all(view.edges.index == inds), "Selecting edges by index failed"

    # check if pre and post comps of edges are in nodes
    edge_node_inds = np.unique(
        view.edges[["pre_index", "post_index"]].to_numpy().flatten()
    )
    assert np.all(
        view.nodes.index == edge_node_inds
    ), "Selecting edges did not yield the correct nodes."

    # select nodes and edges
    node_inds = np.random.choice(net.nodes.index, replace=False, size=5)
    edge_inds = np.random.choice(net.edges.index, replace=False, size=2)
    view = net.select(nodes=node_inds, edges=edge_inds)
    assert np.all(
        view.nodes.index == node_inds
    ), "Selecting nodes and edges by index failed for nodes."
    assert np.all(
        view.edges.index == edge_inds
    ), "Selecting nodes and edges by index failed for edges."


def test_viewing(SimpleCell, SimpleNet):
    """Test that the View object is working correctly."""
    cell = SimpleCell(3, 3)
    net = SimpleNet(3, 3, 3)

    # test parameter sharing works correctly
    nodes1 = net.branch(0).comp("all").nodes
    nodes2 = net.branch(0).nodes
    nodes3 = net.cell(0).nodes
    control_params1 = nodes1.pop("controlled_by_param")
    control_params2 = nodes2.pop("controlled_by_param")
    control_params3 = nodes3.pop("controlled_by_param")
    assert np.all(nodes1 == nodes2), "Nodes are not the same"
    assert np.all(
        control_params1 == nodes1["global_comp_index"]
    ), "Parameter sharing is not correct"
    assert np.all(
        control_params2 == nodes2["global_branch_index"]
    ), "Parameter sharing is not correct"
    assert np.all(
        control_params3 == nodes3["global_cell_index"]
    ), "Parameter sharing is not correct"

    # test local and global indexes match the expected targets
    for view, local_targets, global_targets in zip(
        [
            net.branch(0),  # shows every comp on 0th branch of all cells
            cell.branch("all"),  # shows all branches and comps of cell
            net.cell(0).comp(0),  # shows every 0th comp for every branch on 0th cell
            net.comp(0),  # shows 0th comp of every branch of every cell
            cell.comp(0),  # shows 0th comp of every branch of cell
        ],
        [[0, 1, 2] * 3, [0, 1, 2] * 3, [0] * 3, [0] * 9, [0] * 3],
        [
            [0, 1, 2, 9, 10, 11, 18, 19, 20],
            list(range(9)),
            [0, 3, 6],
            list(range(0, 27, 3)),
            list(range(0, 9, 3)),
        ],
    ):
        assert np.all(
            view.nodes["local_comp_index"] == local_targets
        ), "Indices do not match that of the target"
        assert np.all(
            view.nodes["global_comp_index"] == global_targets
        ), "Indices do not match that of the target"

    with pytest.raises(ValueError):
        net.scope("global").comp(999)  # Nothing should be in View


def test_scope(SimpleCell):
    """Ensure scope has the intended effect for Modules and Views."""
    cell = SimpleCell(3, 3)

    view = cell.scope("global").branch(1)
    assert view._scope == "global"
    view = view.scope("local").comp(0)
    assert np.all(
        view.nodes[["global_branch_index", "global_comp_index"]] == [1, 3]
    ), "Expected [1,3] but got {}".format(
        view.nodes[["global_branch_index", "global_comp_index"]]
    )

    cell.set_scope("global")
    assert cell._scope == "global"
    view = cell.branch(1).comp(3)
    assert np.all(
        view.nodes[["global_branch_index", "global_comp_index"]] == [1, 3]
    ), "Expected [1,3] but got {}".format(
        view.nodes[["global_branch_index", "global_comp_index"]]
    )

    cell.set_scope("local")
    assert cell._scope == "local"
    view = cell.branch(1).comp(0)
    assert np.all(
        view.nodes[["global_branch_index", "global_comp_index"]] == [1, 3]
    ), "Expected [1,3] but got {}".format(
        view.nodes[["global_branch_index", "global_comp_index"]]
    )


def test_context_manager(SimpleCell):
    """Test that context manager works correctly for Module."""
    cell = SimpleCell(3, 3)

    with cell.branch(0).comp(0) as comp:
        comp.set("v", -71)
        comp.set("radius", 0.123)

    with cell.branch(1).comp([0, 1]) as comps:
        comps.set("v", -71)
        comps.set("radius", 0.123)

    assert np.all(
        cell.branch(0).comp(1).nodes[["v", "radius"]] == [-70, 1.0]
    ), "Set affected nodes not in context manager View."
    assert np.all(
        cell.branch(0).comp(0).nodes[["v", "radius"]] == [-71, 0.123]
    ), "Context management of View not working."
    assert np.all(
        cell.branch(1).comp([0, 1]).nodes[["v", "radius"]] == [-71, 0.123]
    ), "Context management of View not working."


def test_iter(SimpleBranch):
    """Test that __iter__ works correctly for all modules."""
    branch1 = SimpleBranch(2)
    branch2 = SimpleBranch(3)
    cell = jx.Cell([branch1, branch1, branch2], parents=[-1, 0, 0])
    net = jx.Network([cell] * 2)

    # test iterating over branches with different numbers of compartments
    assert np.all(
        [
            len(branch.nodes) == expected_len
            for branch, expected_len in zip(cell.branches, [2, 2, 3])
        ]
    ), "__iter__ failed for branches with different numbers of compartments."

    # test iterating using cells, branches, and comps properties
    nodes1 = []
    for cell in net.cells:
        for branch in cell.branches:
            for comp in branch.comps:
                nodes1.append(comp.nodes)
    assert len(nodes1) == len(net.nodes), "Some compartments were skipped in iteration."

    nodes2 = []
    for cell in net:
        for branch in cell:
            for comp in branch:
                nodes2.append(comp.nodes)
    assert len(nodes2) == len(net.nodes), "Some compartments were skipped in iteration."
    assert np.all(
        [np.all(n1 == n2) for n1, n2 in zip(nodes1, nodes2)]
    ), "__iter__ is not consistent with [comp.nodes for cell in net.cells for branches in cell.branches for comp in branches.comps]"

    assert np.all(
        [len(comp.nodes) for comp in net[0, 0].comps] == [1, 1]
    ), "Iterator yielded unexpected number of compartments"

    # 0th comp in every branch (3), 1st comp in every branch (3), 2nd comp in (every) branch (only 1 branch with > 2 comps)
    assert np.all(
        [len(comp.nodes) for comp in net[0].comps] == [3, 3, 1]
    ), "Iterator yielded unexpected number of compartments"

    # 0th comp in every branch for every cell (6), 1st comp in every branch for every cell , 2nd comp in (every) branch for every cell
    assert np.all(
        [len(comp.nodes) for comp in net.comps] == [6, 6, 2]
    ), "Iterator yielded unexpected number of compartments"

    for comp in branch1:
        comp.set("v", -72)
    assert np.all(branch1.nodes["v"] == -72), "Setting parameters with __iter__ failed."

    # needs to be redefined because cell was overwritten with View object
    cell = jx.Cell([branch1, branch1, branch2], parents=[-1, 0, 0])
    for branch in cell:
        for comp in branch:
            comp.set("v", -73)
    assert np.all(cell.nodes["v"] == -73), "Setting parameters with __iter__ failed."


def test_synapse_and_channel_filtering(SimpleNet):
    """Test that synapses and channels are filtered correctly by View."""
    net = SimpleNet(3, 3, 3, connect=False)
    net.insert(HH())
    connect(net[0, 0, :], net[1, 0, :], TestSynapse())

    assert np.all(net.cell(0).HH.nodes == net.HH.cell(0).nodes)
    view1 = net.cell([0, 1]).TestSynapse
    nodes1 = view1.nodes
    edges1 = view1.edges
    view2 = net.TestSynapse.cell([0, 1])
    nodes2 = view2.nodes
    edges2 = view2.edges
    nodes_control_param1 = nodes1.pop("controlled_by_param")
    nodes_control_param2 = nodes2.pop("controlled_by_param")
    edges_control_param1 = edges1.pop("controlled_by_param")
    edges_control_param2 = edges2.pop("controlled_by_param")

    # convert to dict so order of cols and index dont matter for __eq__
    assert nodes1.to_dict() == nodes2.to_dict()
    assert np.all(nodes_control_param1 == 0)
    assert np.all(nodes_control_param2 == nodes2["global_cell_index"])

    assert np.all(edges1 == edges2)


def test_view_equals_module(SimpleComp, SimpleBranch):
    """Test that View behaves the same as Module for important attrs and methods."""
    comp = SimpleComp(copy=True)
    branch = SimpleBranch(3)

    comp.insert(HH())
    branch.comp([0, 1]).insert(HH())

    comp.set("v", -71.2)
    branch.comp(0).set("v", -71.2)

    comp.record("v")
    branch.comp([0, 1]).record("v")

    comp.stimulate(np.zeros(100))
    branch.comp([0, 1]).stimulate(np.zeros(100))

    comp.make_trainable("HH_gNa")
    comp.make_trainable("HH_gK")
    branch.comp([0, 1]).make_trainable("HH_gNa")
    branch.make_trainable("HH_gK")

    # test deleting subset of attributes
    branch.comp(1).delete_trainables()
    branch.comp(1).delete_recordings()
    branch.comp(1).delete_stimuli()

    assert (
        branch.comp(1).trainable_params == [] and branch.comp(0).trainable_params != []
    )
    assert branch.comp(1).recordings.empty and not branch.comp(0).recordings.empty
    assert branch.comp(1).externals == {} and branch.comp(0).externals != {}

    # convert to dict so order of cols and index dont matter for __eq__
    assert comp.nodes.to_dict() == branch.comp(0).nodes.to_dict()

    assert comp.trainable_params == branch.comp(0).trainable_params
    assert comp.indices_set_by_trainables == branch.comp(0).indices_set_by_trainables
    assert np.all(comp.recordings == branch.comp(0).recordings)
    assert np.all(
        [
            np.all([np.all(v1 == v2), k1 == k2])
            for (k1, v1), (k2, v2) in zip(
                comp.externals.items(), branch.comp(0).externals.items()
            )
        ]
    )

    assert comp._comp_edges.columns.equals(branch.comp(0)._comp_edges.columns)


@pytest.mark.parametrize("ncomp", [1, 4])
def test_comp_edge_indexing(SimpleCell, ncomp: int):
    """Test whether `_comp_edges` are tracked correctly when viewing."""
    ncomp = 4
    cell = SimpleCell(5, ncomp)
    # branchpoint on one side of branch.
    assert len(cell.branch(0)._comp_edges) == (ncomp - 1) * 2 + 2
    # branchpoint on both sides of branch.
    assert len(cell.branch(1)._comp_edges) == (ncomp - 1) * 2 + 4
    # branchpoint on one side of branch.
    assert len(cell.branch(2)._comp_edges) == (ncomp - 1) * 2 + 2

    assert len(cell.branch([0, 1])._comp_edges) == (ncomp - 1) * 2 * 2 + 2 + 4

    # .comp() should never return any branchpoints.
    assert len(cell.branch(1).comp(0)._comp_edges) == 0
    assert len(cell.branch([0, 1]).comp(0)._comp_edges) == 0
    assert len(cell.branch("all").comp(0)._comp_edges) == 0

    # .loc() should not return any branchpoints if the value != 0.0 or 1.0.
    assert len(cell.branch(1).loc(0.1)._comp_edges) == 0
    assert len(cell.branch([0, 1]).loc(0.2)._comp_edges) == 0
    assert len(cell.branch("all").loc(0.9)._comp_edges) == 0

    # .loc(0.0) or .loc(1.0) should return edges to the branchpoint.
    assert len(cell.branch(1).loc(0.0)._comp_edges) == 2
    assert len(cell.branch(0).loc(0.0)._comp_edges) == 0  # 0 is a tip branch.
    assert len(cell.branch(1).loc(1.0)._comp_edges) == 2
    assert len(cell.branch([0, 1]).loc(1.0)._comp_edges) == 4  # 0 + 1 have endpoints.
    assert len(cell.branch([0, 2]).loc(1.0)._comp_edges) == 2  # only 0 has endpoint.

    # Finally, a few checks that `.loc(0.0)` and `.loc(1.0)` return the correct
    # `comp_edges`.
    assert ncomp - 1 in cell.branch(0).loc(1.0)._comp_edges["sink"].to_numpy().tolist()
    assert ncomp in cell.branch(1).loc(0.0)._comp_edges["sink"].to_numpy().tolist()
    assert (
        2 * ncomp - 1
        not in cell.branch(1).loc(0.0)._comp_edges["sink"].to_numpy().tolist()
    )


def test_module_inheritance():
    """Test inheritance of modules works properly. (see #590)"""

    class CustomCompartment(jx.Compartment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class CustomBranch(jx.Branch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class CustomCell(jx.Cell):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class CustomNetwork(jx.Network):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class CustomChildNetwork(CustomNetwork):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    custom_comp = CustomCompartment()
    custom_branch = CustomBranch(custom_comp, ncomp=3)
    custom_cell = CustomCell([custom_branch], [-1])
    custom_network = CustomNetwork([custom_cell])
    custom_child_network = CustomChildNetwork([custom_cell])

    custom_child_network.cell(0).branch(0).comp(0).nodes
