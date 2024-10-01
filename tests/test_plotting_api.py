# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
from copy import deepcopy

import jax

from jaxley.connect import connect

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import numpy as np

import jaxley as jx
from jaxley.synapses import IonotropicSynapse


def test_cell():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph.swc")
    cell = jx.read_swc(fname, nseg=4)

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = cell.vis(ax=ax)
    ax = cell.branch([0, 1, 2]).vis(ax=ax, col="r")
    ax = cell.branch(1).loc(0.9).vis(ax=ax, col="b")

    # Plot 2.
    cell.branch(0).add_to_group("soma")
    cell.branch(1).add_to_group("soma")
    ax = cell.soma.vis()


def test_network():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph.swc")
    cell1 = jx.read_swc(fname, nseg=4)
    cell2 = jx.read_swc(fname, nseg=4)
    cell3 = jx.read_swc(fname, nseg=4)

    net = jx.Network([cell1, cell2, cell3])
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(2).branch(0).loc(0.0),
        IonotropicSynapse(),
    )

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = net.cell([0, 1]).vis(ax=ax)
    ax = net.cell(2).vis(ax=ax, col="r", type="line")
    ax = net.cell(2).vis(ax=ax, col="r", type="scatter")
    ax = net.cell(0).branch(np.arange(10).tolist()).vis(ax=ax, col="b")

    # Plot 2.
    ax = net.vis(detail="full", type="line")
    ax = net.vis(detail="full", type="scatter")

    # Plot 3.
    net.vis(detail="point")

    # Plot 4.
    net.vis(detail="point", layers=[2, 1])

    # Plot 5.
    net.vis(detail="full", layers=[2, 1])

    # Plot 5.
    net.cell(0).add_to_group("excitatory")
    net.cell(1).add_to_group("excitatory")
    ax = net.excitatory.vis()


def test_vis_networks_built_from_scartch():
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net = jx.Network([cell, cell])
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(2).loc(0.0),
        IonotropicSynapse(),
    )
    net.compute_xyz()

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = net.vis(detail="full", ax=ax)

    # Plot 2.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    net.cell(0).move(0, 100)
    ax = net.vis(detail="full", ax=ax)

    # Plot 3.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    comp.compute_xyz()
    ax = comp.vis(ax=ax)

    # Plot 4.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    branch.compute_xyz()
    ax = branch.vis(ax=ax)


def test_mixed_network():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", "morph.swc")
    cell1 = jx.read_swc(fname, nseg=4)

    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell2 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    net = jx.Network([cell1, cell2])
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(0).loc(0.0),
        IonotropicSynapse(),
    )
    connect(
        net.cell(0).branch(0).loc(0.0),
        net.cell(1).branch(1).loc(0.0),
        IonotropicSynapse(),
    )

    net.compute_xyz()
    net.cell(0).move(0, 800)
    net.cell(1).move(0, -800)
    net.rotate(180)

    before_xyzrs = deepcopy(net.xyzr[len(cell1.xyzr) :])
    net.cell(1).rotate(90)
    after_xyzrs = net.xyzr[len(cell1.xyzr) :]
    # Test that rotation worked as expected.
    for b, a in zip(before_xyzrs, after_xyzrs):
        assert np.allclose(b[:, 0], -a[:, 1], atol=1e-6)
        assert np.allclose(b[:, 1], a[:, 0], atol=1e-6)

    _ = net.vis(detail="full")


def test_volume_plotting():
    comp = jx.Compartment()
    comp.compute_xyz()
    branch = jx.Branch(comp, 4)
    branch.compute_xyz()
    cell = jx.Cell([branch] * 3, [-1, 0, 0])
    cell.compute_xyz()
    net = jx.Network([cell] * 4)
    net.compute_xyz()

    fig, ax = plt.subplots()
    for module in [comp, branch, cell, net]:
        module.vis(type="volume", ax=ax)
        if not isinstance(module, jx.Compartment):
            module[0].vis(type="volume", ax=ax)
    plt.close(fig)
