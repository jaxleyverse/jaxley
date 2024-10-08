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


def test_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(4)], parents=np.array([-1, 0, 0, 0]))
    net1 = jx.Network([cell for _ in range(4)])
    net2 = jx.Network([cell for _ in range(4)])

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

    # # test raise if not part of same net
    connect(cell1_net1, cell2_net1, TestSynapse())
    with pytest.raises(AssertionError):
        connect(cell1_net1, cell1_net2, TestSynapse())  # should raise

    # test raise if pre and post comp are the same
    with pytest.raises(AssertionError):
        connect(cell1_net1, cell1_net1, TestSynapse())  # should raise

    ### test connect multiple
    # test connect multiple with single synapse
    connect(net2[1, 0], net2[2, 0], TestSynapse())

    # test after all connections are made, to catch "overwritten" connections
    get_comps = lambda locs: [
        local_index_of_loc(loc, 0, net2.nseg_per_branch) for loc in locs
    ]

    # check if all connections are made correctly
    first_set_edges = net2.edges.iloc[:8]
    assert (first_set_edges[["pre_branch_index", "post_branch_index"]] == 0).all().all()
    assert (first_set_edges["pre_cell_index"] == 1).all()
    assert (first_set_edges["post_cell_index"] == 2).all()
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
        net.edges.global_post_comp_index
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


def test_sparse_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))
    net = jx.Network([cell for _ in range(4 * 4)])

    _ = np.random.seed(0)
    for i in range(4):
        sparse_connect(net[i], net[4:8], TestSynapse(), p=0.5)

    sparse_connect(net[8:12], net[12:], TestSynapse(), p=0.5)

    assert all(
        [
            63,
            59,
            65,
            86,
            80,
            58,
            92,
            85,
            168,
            145,
            189,
            153,
            180,
            190,
            184,
            163,
            159,
            179,
            182,
        ]
    )


def test_connectivity_matrix_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))

    _ = np.random.seed(0)
    n_by_n_adjacency_matrix = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=bool
    )
    incides_of_connected_cells = np.stack(np.where(n_by_n_adjacency_matrix)).T
    incides_of_connected_cells[:, 1] += 4

    net = jx.Network([cell for _ in range(4 * 4)])
    connectivity_matrix_connect(
        net[:4], net[4:8], TestSynapse(), n_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 4
    assert (
        (net.edges[["pre_cell_index", "post_cell_index"]] == incides_of_connected_cells)
        .all()
        .all()
    )

    m_by_n_adjacency_matrix = np.array(
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=bool
    )
    incides_of_connected_cells = np.stack(np.where(m_by_n_adjacency_matrix)).T

    net = jx.Network([cell for _ in range(4 * 4)])
    with pytest.raises(AssertionError):
        connectivity_matrix_connect(
            net[:4], net[:4], TestSynapse(), m_by_n_adjacency_matrix
        )  # should raise

    connectivity_matrix_connect(
        net[:3], net[:4], TestSynapse(), m_by_n_adjacency_matrix
    )
    assert len(net.edges.index) == 5
    assert (
        (net.edges[["pre_cell_index", "post_cell_index"]] == incides_of_connected_cells)
        .all()
        .all()
    )
