import os

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import numpy as np

import jaxley as jx
from jaxley.synapses import IonotropicSynapse


def test_cell():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "morph.swc")
    cell = jx.read_swc(fname, nseg=4)

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = cell.vis(ax=ax)
    ax = cell.branch([0, 1, 2]).vis(ax=ax, col="r")
    ax = cell.branch(1).comp(0.9).vis(ax=ax, col="b")

    # Plot 2.
    cell.branch(0).add_to_group("soma")
    cell.branch(1).add_to_group("soma")
    ax = cell.soma.vis()


def test_network():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "morph.swc")
    cell1 = jx.read_swc(fname, nseg=4)
    cell2 = jx.read_swc(fname, nseg=4)
    cell3 = jx.read_swc(fname, nseg=4)

    conns = [
        jx.Connectivity(
            IonotropicSynapse(),
            [jx.Connection(0, 0, 0.0, 1, 0, 0.0), jx.Connection(0, 0, 0.0, 2, 0, 0.0)],
        )
    ]
    net = jx.Network([cell1, cell2, cell3], conns)

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = net.cell([0, 1]).vis(ax=ax)
    ax = net.cell(2).vis(ax=ax, col="r")
    ax = net.cell(0).branch(np.arange(10).tolist()).vis(ax=ax, col="b")

    # Plot 2.
    ax = net.vis(detail="full")

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

    conns = [
        jx.Connectivity(
            IonotropicSynapse(),
            [jx.Connection(0, 0, 0.0, 1, 0, 0.0), jx.Connection(0, 0, 0.0, 1, 2, 0.0)],
        )
    ]
    net = jx.Network([cell, cell], conns)
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
    fname = os.path.join(dirname, "morph.swc")
    cell1 = jx.read_swc(fname, nseg=4)

    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell2 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    conns = [
        jx.Connectivity(
            IonotropicSynapse(),
            [jx.Connection(0, 0, 0.0, 1, 0, 0.0), jx.Connection(0, 0, 0.0, 1, 1, 0.0)],
        )
    ]

    net = jx.Network([cell1, cell2], conns)

    net.compute_xyz()
    net.cell(0).move(0, 800)
    net.cell(1).move(0, -800)
    net.rotate(180)
    net.cell(1).rotate(90)

    _ = net.vis(detail="full")
