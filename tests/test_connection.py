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
    nodes = net2.nodes.set_index("global_comp_index")
    cols = ["pre_global_comp_index", "post_global_comp_index"]
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


def test_fully_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))
    net = jx.Network([cell for _ in range(4 * 4)])

    _ = np.random.seed(0)
    for i in range(4):
        fully_connect(net[i], net[4:8], TestSynapse())

    fully_connect(net[8:12], net[12:16], TestSynapse())

    assert all(
        net.edges.post_global_comp_index
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


def test_sparse_connect(SimpleNet):
    net = SimpleNet(4 * 4, 4, 4)

    _ = np.random.seed(0)
    for i in range(4):
        sparse_connect(net[i], net[4:8], TestSynapse(), p=0.5, random_post_comp=True)

    sparse_connect(net[8:12], net[12:], TestSynapse(), p=0.5, random_post_comp=True)

    assert all(
        net.edges.post_global_comp_index
        == [74, 82, 79, 127, 100, 109, 113, 203, 198, 217, 222, 214, 212, 248, 250]
    )


def test_connectivity_matrix_connect(SimpleNet):
    net = SimpleNet(4 * 4, 3, 8)

    _ = np.random.seed(0)
    n_by_n_adjacency_matrix = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=bool
    )
    incides_of_connected_cells = np.stack(np.where(n_by_n_adjacency_matrix)).T
    incides_of_connected_cells[:, 1] += 4

    connectivity_matrix_connect(
        net[:4], net[4:8], TestSynapse(), n_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 4
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_global_comp_index", "post_global_comp_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(cell_inds == incides_of_connected_cells)

    m_by_n_adjacency_matrix = np.array(
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=bool
    )
    incides_of_connected_cells = np.stack(np.where(m_by_n_adjacency_matrix)).T

    net = SimpleNet(4 * 4, 3, 8)
    with pytest.raises(AssertionError):
        connectivity_matrix_connect(
            net[:4], net[:4], TestSynapse(), m_by_n_adjacency_matrix
        )  # should raise

    connectivity_matrix_connect(
        net[:3], net[:4], TestSynapse(), m_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 5
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_global_comp_index", "post_global_comp_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(cell_inds == incides_of_connected_cells)

    # Test with different cell views
    net = SimpleNet(4 * 4, 3, 8)
    connectivity_matrix_connect(
        net[1:4], net[2:6], TestSynapse(), m_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 5
    nodes = net.nodes.set_index("global_comp_index")
    cols = ["pre_global_comp_index", "post_global_comp_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    # adjust the cell indices based on the views passed
    incides_of_connected_cells[:, 0] += 1
    incides_of_connected_cells[:, 1] += 2
    assert np.all(cell_inds == incides_of_connected_cells)

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
    cols = ["pre_global_comp_index", "post_global_comp_index"]
    comp_inds = nodes.loc[net.edges[cols].to_numpy().flatten()]
    cell_inds = comp_inds["global_cell_index"].to_numpy().reshape(-1, 2)
    assert np.all(cell_inds == incides_of_connected_cells)
