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
import pytest

import jaxley as jx
from jaxley.synapses import IonotropicSynapse


@pytest.fixture(scope="module")
def comp() -> jx.Compartment:
    comp = jx.Compartment()
    comp.compute_xyz()
    return comp


@pytest.fixture(scope="module")
def branch(comp) -> jx.Branch:
    branch = jx.Branch(comp, 4)
    branch.compute_xyz()
    return branch


@pytest.fixture(scope="module")
def cell(branch) -> jx.Cell:
    cell = jx.Cell(branch, [-1, 0, 0, 1, 1])
    cell.branch(0).set_ncomp(3)
    cell.compute_xyz()
    return cell


@pytest.fixture(scope="module")
def simple_net(cell) -> jx.Network:
    net = jx.Network([cell] * 4)
    net.compute_xyz()
    return net


@pytest.fixture(scope="module")
def morph_cell() -> jx.Cell:
    morph_cell = jx.read_swc(
        os.path.join(os.path.dirname(__file__), "swc_files", "morph.swc"),
        nseg=1,
    )
    morph_cell.branch(0).set_ncomp(2)
    return morph_cell


def test_cell(morph_cell):
    # Plot 1.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = morph_cell.vis(ax=ax)
    ax = morph_cell.branch([0, 1, 2]).vis(ax=ax, col="r")
    ax = morph_cell.branch(1).loc(0.9).vis(ax=ax, col="b")

    # Plot 2.
    morph_cell.branch(0).add_to_group("soma")
    morph_cell.branch(1).add_to_group("soma")
    ax = morph_cell.soma.vis()


def test_network(morph_cell):
    net = jx.Network([morph_cell, morph_cell, morph_cell])
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


def test_vis_networks_built_from_scratch(comp, branch, cell):
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
    ax = comp.vis(ax=ax)

    # Plot 4.
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = branch.vis(ax=ax)


def test_mixed_network(morph_cell, cell):
    net = jx.Network([morph_cell, cell])
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

    before_xyzrs = deepcopy(net.xyzr[len(morph_cell.xyzr) :])
    net.cell(1).rotate(90)
    after_xyzrs = net.xyzr[len(morph_cell.xyzr) :]
    # Test that rotation worked as expected.
    for b, a in zip(before_xyzrs, after_xyzrs):
        assert np.allclose(b[:, 0], -a[:, 1], atol=1e-6)
        assert np.allclose(b[:, 1], a[:, 0], atol=1e-6)

    _ = net.vis(detail="full")


def test_volume_plotting_2d(comp, branch, cell, simple_net, morph_cell):
    fig, ax = plt.subplots()
    for module in [comp, branch, cell, simple_net, morph_cell]:
        module.vis(type="comp", ax=ax, morph_plot_kwargs={"resolution": 6})
    plt.close(fig)


def test_volume_plotting_3d(comp, branch, cell, simple_net, morph_cell):
    # test 3D plotting
    for module in [comp, branch, cell, simple_net, morph_cell]:
        module.vis(type="comp", dims=[0, 1, 2], morph_plot_kwargs={"resolution": 6})
    plt.close()


def test_morph_plotting(morph_cell):
    # test morph plotting (does not work if no radii in xyzr)
    morph_cell.vis(type="morph", morph_plot_kwargs={"resolution": 6})
    morph_cell.branch(1).vis(
        type="morph", dims=[0, 1, 2], morph_plot_kwargs={"resolution": 6}
    )  # plotting whole thing takes too long
    plt.close()
