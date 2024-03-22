import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jax import jit

import jaxley as jx


def test_move():
    nseg = 4
    length = 10.0

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell.compute_xyz()
    cell.move(20.0, 30.0, 5.0)
    assert cell.xyzr[0][0, 0] == 20.0
    assert cell.xyzr[0][0, 1] == 30.0
    assert cell.xyzr[0][0, 2] == 5.0

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell.compute_xyz()
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


def test_move_to():
    nseg = 4

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell.compute_xyz()
    cell.move_to(20.0, 30.0, 5.0)
    assert cell.xyzr[0][0, 0] == 20.0
    assert cell.xyzr[0][0, 1] == 30.0
    assert cell.xyzr[0][0, 2] == 5.0

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=nseg)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell.compute_xyz()
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
