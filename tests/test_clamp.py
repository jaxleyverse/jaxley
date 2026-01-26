# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

from jaxley.connect import connect
from jaxley.synapses.ionotropic import IonotropicSynapse

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH, CaL, CaT, Channel, K, Km, Leak, Na


def test_clamp_pointneuron(SimpleComp):
    comp = SimpleComp()
    comp.insert(HH())
    comp.record()
    comp.clamp("v", -50.0 * jnp.ones((1000,)))

    v = jx.integrate(comp, t_max=1.0)
    assert np.all(v[:, 1:] == -50.0)


def test_clamp_currents(SimpleComp):
    comp = SimpleComp()
    comp.insert(HH())
    comp.record("i_HH")

    # test clamp
    comp.clamp("i_HH", 1.0 * jnp.ones((1000,)))
    i1 = jx.integrate(comp, t_max=1.0)
    assert np.all(i1[:, 1:] == 1.0)

    # test data clamp
    data_clamps = None
    ipts = 1.0 * jnp.ones((1000,))
    data_clamps = comp.data_clamp("i_HH", ipts, data_clamps=data_clamps)

    i2 = jx.integrate(comp, data_clamps=data_clamps, t_max=1.0)
    assert np.all(i2[:, 1:] == 1.0)

    assert np.all(np.isclose(i1, i2))


def test_clamp_synapse(SimpleNet):
    net = SimpleNet(2, 1, 1)
    connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
    net.record("IonotropicSynapse_s")

    # test clamp
    net.clamp("IonotropicSynapse_s", 1.0 * jnp.ones((1000,)))
    s1 = jx.integrate(net, t_max=1.0)
    assert np.all(s1[:, 1:] == 1.0)

    net.delete_clamps()

    # test data clamp
    data_clamps = None
    ipts = 1.0 * jnp.ones((1000,))
    data_clamps = net.data_clamp("IonotropicSynapse_s", ipts, data_clamps=data_clamps)

    s2 = jx.integrate(net, data_clamps=data_clamps, t_max=1.0)
    assert np.all(s2[:, 1:] == 1.0)

    assert np.all(np.isclose(s1, s2))


def test_clamp_multicompartment(SimpleBranch):
    branch = SimpleBranch(4)
    branch.insert(HH())
    branch.record()
    branch.comp(0).clamp("v", -50.0 * jnp.ones((1000,)))

    v = jx.integrate(branch, t_max=1.0)

    # The clamped compartment should be fixed.
    assert np.all(v[0, 1:] == -50.0)

    # For other compartments, the voltage should have non-zero std.
    assert np.all(np.std(v[1:, 1:], axis=1) > 0.1)


def test_clamp_and_stimulate_api(SimpleCell):
    """Ensure proper behaviour when `.clamp()` and `.stimulate()` are combined."""
    cell1 = SimpleCell(1, 4)
    cell2 = SimpleCell(1, 4)
    net = jx.Network([cell1, cell2])

    net.insert(HH())
    net[0, 0, 0].record()
    net[1, 0, 0].record()

    net[0, 0, 0].clamp("v", 0.1 * jnp.ones((1000,)))
    net[1, 0, 0].stimulate(0.1 * jnp.ones((1000,)))

    vs1 = jx.integrate(net, t_max=1.0)

    cell1.insert(HH())
    cell1[0, 0].record()
    cell1[0, 0].clamp("v", 0.1 * jnp.ones((1000,)))
    vs21 = jx.integrate(cell1, t_max=1.0)

    cell2.insert(HH())
    cell2[0, 0].record()
    cell2[0, 0].stimulate(0.1 * jnp.ones((1000,)))
    vs22 = jx.integrate(cell2, t_max=1.0)

    vs2 = jnp.concatenate([vs21, vs22])
    assert np.max(np.abs(vs1 - vs2)) < 1e-8


def test_data_clamp(SimpleComp):
    """Data clamp with no stimuli or data_stimuli, and no t_max (should get defined by the clamp)."""
    comp = SimpleComp()
    comp.insert(HH())
    comp.record()
    clamp = -50.0 * jnp.ones((1000,))

    def provide_data(clamp):
        data_clamps = comp.data_clamp("v", clamp)
        return data_clamps

    def simulate(clamp):
        data_clamps = provide_data(clamp)
        return jx.integrate(comp, data_clamps=data_clamps)

    jitted_simulate = jax.jit(simulate)

    s = jitted_simulate(clamp)
    assert np.all(s[:, 1:] == -50.0)


def test_data_clamp_and_data_stimulate(SimpleComp):
    """In theory people shouldn't use these two together, but at least it shouldn't break."""
    comp = SimpleComp()
    comp.insert(HH())
    comp.record()
    clamp = -50.0 * jnp.ones((1000,))
    stim = 0.1 * jnp.ones((1000,))

    def provide_data(clamp, stim):
        data_clamps = comp.data_clamp("v", clamp)
        data_stims = comp.data_stimulate(stim)
        return data_clamps, data_stims

    def simulate(clamp, stim):
        data_clamps, data_stims = provide_data(clamp, stim)
        return jx.integrate(comp, data_clamps=data_clamps, data_stimuli=data_stims)

    jitted_simulate = jax.jit(simulate)

    s = jitted_simulate(clamp, stim)
    assert np.all(s[:, 1:] == -50.0)


def test_data_clamp_and_stimulate(SimpleComp):
    """Test that data clamp overrides a previously set stimulus."""
    comp = SimpleComp()
    comp.insert(HH())
    comp.record()
    clamp = -50.0 * jnp.ones((1000,))
    stim = 0.1 * jnp.ones((800,))
    t_max = clamp.shape[0] * 0.025  # make sure the stimulus gets padded
    comp.stimulate(stim)

    def simulate(clamp):
        data_clamps = comp.data_clamp("v", clamp)  # should override the stimulation
        return jx.integrate(comp, data_clamps=data_clamps, t_max=t_max)

    jitted_simulate = jax.jit(simulate)

    s = jitted_simulate(clamp)
    assert np.all(s[:, 1:] == -50.0)


def test_data_clamp_and_clamp(SimpleComp):
    """Test that data clamp can override (same loc.) and add (another loc.) to clamp."""
    comp = SimpleComp()
    comp.insert(HH())
    comp.record()
    clamp1 = -50.0 * jnp.ones((1000,))
    clamp2 = -60.0 * jnp.ones((1000,))
    comp.clamp("v", clamp1)

    def simulate(clamp):
        data_clamps = comp.data_clamp(
            "v", clamp, None
        )  # should override the first clamp
        return jx.integrate(comp, data_clamps=data_clamps)

    jitted_simulate = jax.jit(simulate)

    # Clamp2 should override clamp1 here
    s = jitted_simulate(clamp2)
    assert np.all(s[:, 1:] == -60.0)

    comp2 = SimpleComp()
    comp2.insert(HH())
    branch1 = jx.Branch(comp, 4)
    branch2 = jx.Branch(comp2, 4)
    cell = jx.Cell([branch1, branch2], [-1, 0])

    # Apply the clamp1 to the second branch via clamp
    cell[1, 0].clamp("v", clamp1)

    cell.delete_recordings()
    cell.branch(0).comp(0).record()
    cell.branch(1).comp(0).record()

    def simulate(clamp):
        data_clamps = cell.branch(0).comp(0).data_clamp("v", clamp, None)
        return jx.integrate(cell, data_clamps=data_clamps)

    jitted_simulate = jax.jit(simulate)

    # Apply clamp2 to the first branch via data_clamp
    s = jitted_simulate(clamp2)

    assert np.all(s[0, 1:] == -60.0)
    assert np.all(s[1, 1:] == -50.0)


def test_multiple_data_clamps(SimpleNet):
    """Test that data clamps can be added to different states."""
    net = SimpleNet(2, 1, 1)
    net.cell(0).insert(HH(name="cell1"))
    net.cell(1).insert(HH(name="cell2"))

    net.cell(0).record("cell1_m")
    net.cell(1).record("cell2_m")

    time = np.arange(200, step=0.1)
    m1 = np.zeros_like(time) + 0.2
    m2 = np.zeros_like(time) + 0.3

    def simulate(clamps):
        data_clamps = net.cell(0).data_clamp("cell1_m", clamps[0], None)
        data_clamps = net.cell(1).data_clamp("cell2_m", clamps[1], data_clamps)
        return jx.integrate(net, delta_t=0.1, data_clamps=data_clamps)

    jitted_simulate = jax.jit(simulate)
    soln = jitted_simulate((m1, m2))

    net2 = SimpleNet(2, 1, 1)
    net2.cell(0).insert(HH(name="cell1"))
    net2.cell(1).insert(HH(name="cell2"))

    net2.cell(0).record("cell1_m")
    net2.cell(1).record("cell2_m")

    net2.cell(0).clamp("cell1_m", m1)
    net2.cell(1).clamp("cell2_m", m2)
    soln2 = jx.integrate(net2, delta_t=0.1)

    assert np.all(soln == soln2)


def test_checkpointing_multiple_clamps(SimpleNet):
    """Test that checkpointing works with multiple data clamps of different shapes."""
    net = SimpleNet(3, 1, 1)
    net.insert(HH())

    time = np.arange(200, step=0.1)
    m_clamp = np.zeros_like(time) + 0.2
    h_clamp = np.zeros_like(time) + 0.4

    net.cell(0).record("HH_m")
    net.cell([1, 2]).record("HH_h")

    checkpoints = tuple(int(np.ceil(len(time) ** (1 / 5))) for _ in range(5))

    def simulate(clamps):
        data_clamps = net.cell(0).data_clamp("HH_m", clamps[0], None)
        data_clamps = net.cell([1, 2]).data_clamp("HH_h", clamps[1], data_clamps)
        return jx.integrate(
            net, delta_t=0.1, data_clamps=data_clamps, checkpoint_lengths=checkpoints
        )

    soln = simulate((m_clamp, h_clamp))
    assert np.all(soln[0, 1:] == 0.2)
    assert np.all(soln[1:, 1:] == 0.4)
