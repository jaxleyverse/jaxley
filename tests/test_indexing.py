from copy import deepcopy

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.utils.cell_utils import index_of_loc, loc_of_index, flip_comp_indices


def test_flip_compartment_indices():
    nseg = 4

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg)
    cell1 = jx.Cell(branch, parents=[-1, 0, 0])
    cell2 = jx.Cell(branch, parents=[-1, 0, 0])

    indices = [0, 1, 2, 3]
    flipped_inds = flip_comp_indices(np.asarray(indices), nseg).tolist()
    radii = np.random.rand(nseg)

    for counter, i in enumerate(flipped_inds):
        cell1[1, i].set("radius", radii[counter])
    for counter, i in enumerate(indices):
        cell2[1, i].set("radius", np.flip(radii)[counter])
    assert all(cell1.nodes == cell2.nodes)
    

def test_getitem():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(3)], parents=jnp.asarray([-1, 0, 0]))
    net = jx.Network([cell for _ in range(3)])

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
    for cell in net:
        pass

    for cell in net:
        for branch in cell:
            for comp in branch:
                pass

    for comp in net[0, 0]:
        pass


def test_loc_v_comp():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])

    assert np.all(branch.comp(0).show() == branch.loc(0.0).show())
    assert np.all(branch.comp(3).show() == branch.loc(1.0).show())

    assert np.all(branch.loc(loc_of_index(2, 4)).show() == branch.comp(2).show())
    assert np.all(branch.comp(index_of_loc(0, 0.4, 4)).show() == branch.loc(0.4).show())


def test_shape():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(3)], parents=jnp.asarray([-1, 0, 0]))
    net = jx.Network([cell for _ in range(3)])

    assert net.shape == (3, 3, 4)
    assert cell.shape == (1, 3, 4)
    assert branch.shape == (1, 4)
    assert comp.shape == (1,)

    assert net.cell.shape == net.shape
    assert cell.branch.shape == cell.shape

    assert net.cell.shape == (3, 3, 4)
    assert net.cell.branch.shape == (3, 3, 4)
    assert net.cell.branch.comp.shape == (3, 3, 4)

    assert net.cell(0).shape == (1, 3, 4)
    assert net.cell(0).branch(0).shape == (1, 1, 4)
    assert net.cell(0).branch(0).comp(0).shape == (1, 1, 1)


def test_set_and_insert():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 0, 1, 1]))
    net = jx.Network([cell for _ in range(5)])
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
        i_delay=10.0, i_dur=80.0, i_amp=5.0, delta_t=0.025, t_max=100.0
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

    assert np.all(cell1.currents == cell2.currents)
    assert np.all(cell1.current_inds == cell2.current_inds)
    assert np.all(cell1.recordings == cell2.recordings)


def test_local_indexing():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 0, 1, 1]))
    net = jx.Network([cell for _ in range(2)])

    local_idxs = net[:]._get_local_indices()
    idx_cols = ["cell_index", "branch_index", "comp_index"]

    global_index = 0
    for cell_idx in range(2):
        for branch_idx in range(5):
            for comp_idx in range(4):
                compview = net[cell_idx, branch_idx, comp_idx].show()
                assert np.all(
                    compview[idx_cols].values == [cell_idx, branch_idx, comp_idx]
                )
                assert np.all(
                    local_idxs.iloc[global_index] == [cell_idx, branch_idx, comp_idx]
                )
                global_index += 1


def test_child_view():
    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(4)])
    cell = jx.Cell([branch for _ in range(5)], parents=jnp.asarray([-1, 0, 0, 1, 1]))
    net = jx.Network([cell for _ in range(2)])

    assert np.all(net._childview(0).show() == net.cell(0).show())
    assert np.all(cell._childview(0).show() == cell.branch(0).show())
    assert np.all(branch._childview(0).show() == branch.comp(0).show())

    assert np.all(
        net._childview(0)._childview(0).show() == net.cell(0).branch(0).show()
    )
    assert np.all(
        cell._childview(0)._childview(0).show() == cell.branch(0).comp(0).show()
    )
