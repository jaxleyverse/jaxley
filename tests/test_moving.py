# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import os

import jax.numpy as jnp
import numpy as np
from jax import jit

import jaxley as jx


def test_move_cell(SimpleBranch, SimpleCell):
    # Test move on a cell with compute_xyz()
    cell = SimpleCell(5, ncomp=4)
    cell.compute_xyz()
    cell.move(20.0, 30.0, 5.0)
    assert cell.xyzr[0][0, 0] == 20.0
    assert cell.xyzr[0][0, 1] == 30.0
    assert cell.xyzr[0][0, 2] == 5.0

    # Test move_to on a cell that starts with a specified xyzr
    branch = SimpleBranch(ncomp=4)
    cell = jx.Cell(
        branch,
        parents=[-1],
        xyzr=[
            np.array(
                [
                    [5.0, 10.0, 1.0, 10.0],
                    [10.0, 15.0, 0.0, 10.0],
                ]
            )
        ],
    )
    cell.move(6.0, 9.0, 3.0)
    assert cell.xyzr[0][0, 0] == 11.0
    assert cell.xyzr[0][0, 1] == 19.0
    assert cell.xyzr[0][0, 2] == 4.0
    assert cell.xyzr[0][0, 3] == 10.0


def test_move_network(SimpleCell):
    cell = SimpleCell(3, 3)
    cell.compute_xyz()
    net = jx.Network([cell, cell, cell])
    net.move(20.0, 30.0, 5.0)
    for i in [0, 3, 6]:
        assert net.xyzr[i][0, 0] == 20.0
        assert net.xyzr[i][0, 1] == 30.0
        assert net.xyzr[i][0, 2] == 5.0


def test_move_to_cell(SimpleBranch, SimpleCell):
    cell = SimpleCell(5, 4)
    cell.compute_xyz()
    cell.move_to(20.0, 30.0, 5.0)
    assert cell.xyzr[0][0, 0] == 20.0
    assert cell.xyzr[0][0, 1] == 30.0
    assert cell.xyzr[0][0, 2] == 5.0

    branch = SimpleBranch(ncomp=4)
    cell = jx.Cell(
        branch,
        parents=[-1],
        xyzr=[
            np.array(
                [
                    [5.0, 10.0, 1.0, 10.0],
                    [10.0, 15.0, 0.0, 10.0],
                ]
            )
        ],
    )
    cell.move_to(6.0, 9.0, 3.0)
    assert cell.xyzr[0][0, 0] == 6.0
    assert cell.xyzr[0][0, 1] == 9.0
    assert cell.xyzr[0][0, 2] == 3.0
    assert cell.xyzr[0][0, 3] == 10.0


def test_move_to_network(SimpleNet):
    net = SimpleNet(3, 3, 4)
    net.compute_xyz()
    net.move_to(10.0, 20.0, 30.0)
    # Branch 0 of cell 0
    assert net.xyzr[0][0, 0] == 10.0
    assert net.xyzr[0][0, 1] == 20.0
    assert net.xyzr[0][0, 2] == 30.0
    # Branch 0 of cell 1
    assert net.xyzr[3][0, 0] == 10.0
    assert net.xyzr[3][0, 1] == 20.0
    assert net.xyzr[3][0, 2] == 30.0


def test_move_to_arrays(SimpleNet):
    """Test with network"""
    ncomp = 4
    net = SimpleNet(3, 3, ncomp)
    net.compute_xyz()
    x_coords = np.array([10.0, 20.0, 30.0])
    y_coords = np.array([5.0, 15.0, 25.0])
    z_coords = np.array([1.0, 2.0, 3.0])
    net.move_to(x_coords, y_coords, z_coords)
    assert net.xyzr[0][0, 0] == 10.0
    assert net.xyzr[0][1, 0] == ncomp * 10.0 + 10.0
    assert net.xyzr[0][0, 1] == 5.0
    assert net.xyzr[0][0, 2] == 1.0
    assert net.xyzr[3][0, 0] == 20.0
    assert net.xyzr[3][0, 2] == 2.0
    assert net.xyzr[6][0, 1] == 25.0


def test_move_to_cellview(SimpleNet):
    net = SimpleNet(3, 3, 2)
    net.compute_xyz()

    # Test with float input
    net.cell(0).move_to(50.0, 3.0, 40.0)
    assert net.xyzr[0][0, 0] == 50.0
    assert net.xyzr[0][0, 1] == 3.0
    assert net.xyzr[0][0, 2] == 40.0

    # Test with array input
    net = SimpleNet(4, 3, 2)
    net.compute_xyz()
    testx = np.array([1.0, 2.0, 3.0])
    testy = np.array([4.0, 5.0, 6.0])
    testz = np.array([7.0, 8.0, 9.0])
    net.cell([0, 1, 2]).move_to(testx, testy, testz)
    assert net.xyzr[0][0, 0] == 1.0
    assert net.xyzr[3][0, 1] == 5.0
    assert net.xyzr[6][0, 2] == 9.0
    assert net.xyzr[9][0, 0] == 0.0


def test_move_to_swc_cell(SimpleMorphCell):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph_ca1_n120.swc")
    cell1 = SimpleMorphCell(fname, ncomp=1)
    cell2 = SimpleMorphCell(fname, ncomp=1)
    cell3 = SimpleMorphCell(fname, ncomp=1)

    # Try move_to on a cell
    cell1.move_to(10.0, 20.0, 30.0)
    assert cell1.xyzr[0][0, 0] == 10.0
    assert cell1.xyzr[0][0, 1] == 20.0

    net = jx.Network([cell1, cell2, cell3])
    # Try move_to on the network
    net.move_to(10.0, 20.0, 30.0)
    assert net.xyzr[0][0, 0] == 10.0
    assert net.xyzr[0][0, 1] == 20.0

    x_coords = np.array([20.0, 30.0, 40.0])
    y_coords = np.array([5.0, 15.0, 25.0])
    z_coords = np.array([1.0, 2.0, 3.0])

    net.move_to(x_coords, y_coords, z_coords)

    n_branches_cell1 = len(cell1.xyzr)
    n_branches_cell2 = len(cell2.xyzr)
    assert net.xyzr[0][0, 0] == 20.0
    assert net.xyzr[n_branches_cell1][0, 1] == 15.0
    assert net.xyzr[n_branches_cell1 + n_branches_cell2][0, 2] == 3.0

    # Test move_to on a subset of the cells
    sub_net = net.cell([1, 2])
    sub_net.move_to(30.0, 40.0, 50.0)
    assert net.xyzr[n_branches_cell1][0, 0] == 30.0
    assert net.xyzr[n_branches_cell1][0, 1] == 40.0

    x_coords = np.array([3.0, 4.0])
    y_coords = np.array([6.0, 7.0])
    z_coords = np.array([9.0, 10.0])
    sub_net.move_to(x_coords, y_coords, z_coords)
    assert net.xyzr[n_branches_cell1][0, 0] == 3.0
    assert net.xyzr[n_branches_cell1 + n_branches_cell2][0, 1] == 7.0
