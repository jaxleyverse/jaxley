# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

from jaxley.utils.cell_utils import local_index_of_loc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import pytest

import jaxley as jx
from jaxley.connect import (
    connect,
    connectivity_matrix_connect,
    fully_connect,
    sparse_connect,
)
from jaxley.synapses import IonotropicSynapse, TestSynapse


def test_connect(SimpleBranch, SimpleCell, SimpleNet):
    branch = SimpleBranch(4)
    cell = SimpleCell(3, 4)
    net1 = SimpleNet(4, 3, 8)
    net2 = SimpleNet(4, 3, 8)

    cell1_net1 = net1[0, 0, 0]
    cell2_net1 = net1[1, 0, 0]
    cell1_net2 = net2[0, 0, 0]

    ### test connect single
    # test raise if not part of net
    with pytest.raises(AssertionError):
        connect(cell[0, 0], cell[0, 1], TestSynapse())  # should raise
    with pytest.raises(AssertionError):
        connect(branch[0], branch[1], TestSynapse())  # should raise
    with pytest.raises(AssertionError):
        connect(cell[0, 0], branch[0], TestSynapse())  # should raise

    # test raise if not part of same net
    connect(cell1_net1, cell2_net1, TestSynapse())
    with pytest.raises(AssertionError):
        connect(cell1_net1, cell1_net2, TestSynapse())  # should raise

    ### test connect multiple
    # test connect multiple with single synapse
    connect(net2[1, 0], net2[2, 0], TestSynapse())

    # test after all connections are made, to catch "overwritten" connections
    get_comps = lambda locs: [
        local_index_of_loc(loc, 0, net2.ncomp_per_branch) for loc in locs
    ]

    # check if all connections are made correctly
    first_set_edges = net2.edges.iloc[:8]
    nodes = net2.nodes
    cols = ["pre_index", "post_index"]
    comp_inds = nodes.loc[first_set_edges[cols].to_numpy().flatten()]
    branch_inds = comp_inds["global_branch_index"].to_numpy().reshape(-1, 2)
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(branch_inds == (3, 6))
    assert (cell_inds == (1, 2)).all()
    assert (
        get_comps(first_set_edges["pre_locs"])
        == get_comps(first_set_edges["post_locs"])
        == [0, 1, 2, 3, 4, 5, 6, 7]
    )
    assert (first_set_edges["type"] == "TestSynapse").all()


def test_fully_connect(SimpleNet):
    net1 = SimpleNet(16, 3, 8)
    for i in range(4):
        fully_connect(net1[i], net1[4:8], TestSynapse())

    fully_connect(net1[8:12], net1[12:16], TestSynapse())

    assert all(
        net1.nodes.loc[net1.edges.post_index, "global_comp_index"]
        == [
            96,
            120,
            144,
            168,
            96,
            120,
            144,
            168,
            96,
            120,
            144,
            168,
            96,
            120,
            144,
            168,
            288,
            312,
            336,
            360,
            288,
            312,
            336,
            360,
            288,
            312,
            336,
            360,
            288,
            312,
            336,
            360,
        ]
    )

    # Test with random post synaptic compartment selection
    net2 = SimpleNet(16, 3, 8)
    _ = np.random.seed(0)
    for i in range(4):
        fully_connect(net2[i], net2[4:8], TestSynapse(), random_post_comp=True)

    fully_connect(net2[8:12], net2[12:16], TestSynapse(), random_post_comp=True)

    assert all(
        net2.nodes.loc[net2.edges.post_index, "global_comp_index"]
        == [
            108,
            135,
            165,
            168,
            99,
            123,
            151,
            177,
            115,
            141,
            162,
            172,
            119,
            126,
            156,
            169,
            294,
            329,
            345,
            379,
            295,
            317,
            356,
            365,
            311,
            325,
            355,
            375,
            302,
            320,
            352,
            375,
        ]
    )


def test_sparse_connect(SimpleNet):
    net1 = SimpleNet(4 * 4, 4, 4)
    _ = np.random.seed(0)

    for i in range(4):
        sparse_connect(net1[i], net1[4:8], TestSynapse(), p=0.1)

    sparse_connect(net1[8:12], net1[12:], TestSynapse(), p=0.5)

    assert all(
        net1.nodes.loc[net1.edges.post_index, "global_comp_index"]
        == [112, 112, 240, 192, 208, 208, 208, 224, 208, 192, 240]
    )

    # Test with random post synaptic compartment selection
    net2 = SimpleNet(4 * 4, 4, 4)
    _ = np.random.seed(0)

    for i in range(4):
        sparse_connect(net2[i], net2[4:8], TestSynapse(), p=0.1, random_post_comp=True)

    sparse_connect(net2[8:12], net2[12:], TestSynapse(), p=0.5, random_post_comp=True)

    assert all(
        net2.nodes.loc[net2.edges.post_index, "global_comp_index"]
        == [123, 201, 196, 211, 208, 211, 213, 238, 255, 255]
    )


def test_connectivity_matrix_connect(SimpleNet):
    net = SimpleNet(4 * 4, 3, 8)

    n_by_n_adjacency_matrix = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=bool
    )
    inds_of_connected_cells = np.stack(np.where(n_by_n_adjacency_matrix)).T
    inds_of_connected_cells[:, 1] += 4  # post-syn cell inds start a 4 below

    connectivity_matrix_connect(
        net[:4], net[4:8], TestSynapse(), n_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 4
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_index", "post_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(cell_inds == inds_of_connected_cells)
    assert all(
        net.nodes.loc[net.edges.post_index, "global_comp_index"] == [120, 144, 168, 96]
    )

    m_by_n_adjacency_matrix = np.array(
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=bool
    )
    inds_of_connected_cells = np.stack(np.where(m_by_n_adjacency_matrix)).T

    net = SimpleNet(4 * 4, 3, 8)
    with pytest.raises(AssertionError):
        connectivity_matrix_connect(
            net[:4], net[:4], TestSynapse(), m_by_n_adjacency_matrix
        )  # should raise

    _ = np.random.seed(0)
    connectivity_matrix_connect(
        net[:3], net[:4], TestSynapse(), m_by_n_adjacency_matrix, random_post_comp=True
    )
    assert len(net.edges.index) == 5
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_index", "post_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(cell_inds == inds_of_connected_cells)
    assert all(
        net.nodes.loc[net.edges.post_index, "global_comp_index"] == [36, 63, 69, 72, 75]
    )

    # Test with different cell views
    net = SimpleNet(4 * 4, 3, 8)
    connectivity_matrix_connect(
        net[1:4], net[2:6], TestSynapse(), m_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 5
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_index", "post_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    # adjust the cell indices based on the views passed
    inds_of_connected_cells[:, 0] += 1
    inds_of_connected_cells[:, 1] += 2
    assert np.all(cell_inds == inds_of_connected_cells)

    # Test with single compartment cells
    comp = jx.Compartment()
    branch = jx.Branch([comp], ncomp=1)
    cell = jx.Cell([branch], parents=[-1])
    net = jx.Network([cell for _ in range(4 * 4)])
    connectivity_matrix_connect(
        net[1:4], net[2:6], TestSynapse(), m_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 5
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_index", "post_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(cell_inds == inds_of_connected_cells)
