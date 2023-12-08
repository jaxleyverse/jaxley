import os

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import numpy as np

import jaxley as jx
from jaxley.synapses import GlutamateSynapse


def test_cell():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "morph.swc")
    cell = jx.read_swc(fname, nseg=4)

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = cell.vis(detail="full", ax=ax)
    ax = cell.branch([0, 1, 2]).vis(detail="full", ax=ax, col="r")

    # Plot 2.
    cell.branch(0).add_to_group("soma")
    cell.branch(1).add_to_group("soma")
    ax = cell.soma.vis(detail="full")


def test_network():
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "morph.swc")
    cell1 = jx.read_swc(fname, nseg=4)
    cell2 = jx.read_swc(fname, nseg=4)
    cell3 = jx.read_swc(fname, nseg=4)

    conns = [
        jx.Connectivity(
            GlutamateSynapse(),
            [jx.Connection(0, 0, 0.0, 1, 0, 0.0), jx.Connection(0, 0, 0.0, 2, 0, 0.0)],
        )
    ]
    net = jx.Network([cell1, cell2, cell3], conns)

    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = net.cell([0, 1]).vis(detail="full", ax=ax)
    ax = net.cell(2).vis(detail="full", ax=ax, col="r")
    ax = net.cell(0).branch(np.arange(10).tolist()).vis(detail="full", ax=ax, col="b")

    # Plot 2.
    ax = net.vis(detail="full")

    # Plot 3.
    net.vis(detail="point")

    # Plot 4.
    net.vis(detail="point", layers=[2, 1])

    # Plot 5.
    net.cell(0).add_to_group("excitatory")
    net.cell(1).add_to_group("excitatory")
    ax = net.excitatory.vis(detail="full")
