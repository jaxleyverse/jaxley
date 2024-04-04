import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import pytest

import jaxley as jx
from jaxley.connection import connect, custom_connect, fully_connect, sparse_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse


def test_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))
    net1 = jx.Network([cell for _ in range(4)])
    net2 = jx.Network([cell for _ in range(4)])

    cell1_net1 = net1[0, 0, 0]
    cell2_net1 = net1[1, 0, 0]
    cell1_net2 = net2[0, 0, 0]

    ### test connect single
    # test raise if not part of net
    connect(cell[0, 0], cell[0, 1], TestSynapse())  # should raise
    connect(branch[0], branch[1], TestSynapse())  # should raise
    connect(cell[0, 0], branch[0], TestSynapse())  # should raise

    # test raise if not part of same net
    connect(cell1_net1, cell2_net1, TestSynapse())
    connect(cell1_net1, cell1_net2, TestSynapse())  # should raise

    # test raise if pre and post comp are the same
    connect(cell1_net1, cell1_net1, TestSynapse())  # should raise

    ### test connect multiple
    # test connect multiple with single synapse
    connect(net1[1, 0], net1[2, 0], TestSynapse())
    connect(net1[1, 1], net1[2, 1], [TestSynapse()])
    # TODO: verify that the synapses are connected correctly

    # test connect multiple with same synapses
    connect(net1[1, 2], net1[2, 2], [TestSynapse()] * 8)
    # TODO: verify that the synapses are connected correctly

    # test connect multiple with different synapses
    connect(
        net1[1, 3, :3],
        net1[2, 3, :3],
        [IonotropicSynapse(), IonotropicSynapse(), TestSynapse()],
    )  # should raise
    # TODO: verify that the synapses are connected correctly
    # TODO: verify that the synapses are registered correctly
    # net1.edges...

    # test connect raise if num synapses does not match
    connect(net1[1, 3, :4], net1[2, 3, :4], [TestSynapse()] * 3)  # should raise


def test_fully_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))
    net = jx.Network([cell for _ in range(4 * 4)])

    for i in range(4):
        fully_connect(net[i], net[4:8], TestSynapse())

    fully_connect(net[8:12], net[12:16], TestSynapse())


def test_sparse_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))
    net = jx.Network([cell for _ in range(4 * 4)])

    for i in range(4):
        sparse_connect(net[i], net[4:8], TestSynapse(), p=0.5)

    sparse_connect(net[8:12], net[12:16], TestSynapse(), p=0.5)


def test_custom_connect():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(8)])
    cell = jx.Cell([branch for _ in range(3)], parents=np.array([-1, 0, 0]))

    n_by_n_adjacency_matrix = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]

    net = jx.Network([cell for _ in range(4)])
    custom_connect(net[:4], net[4:8], TestSynapse(), n_by_n_adjacency_matrix)
    # TODO: verify that the synapses are connected correctly
    # net1.edges...

    m_by_n_adjacency_matrix = [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]

    net = jx.Network([cell for _ in range(4)])
    custom_connect(
        net[:4], net[:4], TestSynapse(), m_by_n_adjacency_matrix
    )  # should raise
    custom_connect(net[:3], net[:4], TestSynapse(), m_by_n_adjacency_matrix)
    # TODO: verify that the synapses are connected correctly
    # net1.edges...
