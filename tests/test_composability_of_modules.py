# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import jaxley as jx
from jaxley.channels import HH
from jaxley.pumps import CaPump


def test_compose_branch():
    """Test inserting to comp and composing to branch equals inserting to branch."""
    dt = 0.025
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    comp1 = jx.Compartment()
    comp1.insert(HH())
    comp1.insert(CaPump())
    comp2 = jx.Compartment()
    branch1 = jx.Branch([comp1, comp2])
    branch1.loc(0.0).record()
    branch1.loc(0.0).record("CaCon_i")
    branch1.loc(0.0).stimulate(current)

    comp = jx.Compartment()
    branch2 = jx.Branch(comp, ncomp=2)
    branch2.loc(0.0).insert(HH())
    branch2.loc(0.0).insert(CaPump())
    branch2.loc(0.0).record()
    branch2.loc(0.0).record("CaCon_i")
    branch2.loc(0.0).stimulate(current)

    assert branch1.pumped_ions == branch2.pumped_ions == ["CaCon_i"]
    recordings1 = jx.integrate(branch1, delta_t=dt)
    recordings2 = jx.integrate(branch2, delta_t=dt)

    assert jnp.max(jnp.abs(recordings1 - recordings2)) < 1e-8


def test_compose_cell():
    """Test inserting to branch and composing to cell equals inserting to cell."""
    ncomp_per_branch = 4
    dt = 0.025
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    comp = jx.Compartment()

    branch1 = jx.Branch(comp, ncomp_per_branch)
    branch1.insert(HH())
    branch1.insert(CaPump())
    branch2 = jx.Branch(comp, ncomp_per_branch)
    cell1 = jx.Cell([branch1, branch2], parents=[-1, 0])
    cell1.branch(0).loc(0.0).record()
    cell1.branch(0).loc(0.0).record("CaCon_i")
    cell1.branch(0).loc(0.0).stimulate(current)

    branch = jx.Branch(comp, ncomp_per_branch)
    cell2 = jx.Cell(branch, parents=[-1, 0])
    cell2.branch(0).insert(HH())
    cell2.branch(0).insert(CaPump())
    cell2.branch(0).loc(0.0).record()
    cell2.branch(0).loc(0.0).record("CaCon_i")
    cell2.branch(0).loc(0.0).stimulate(current)

    assert cell1.pumped_ions == cell2.pumped_ions == ["CaCon_i"]
    recordings1 = jx.integrate(cell1, delta_t=dt)
    recordings2 = jx.integrate(cell2, delta_t=dt)

    assert jnp.max(jnp.abs(recordings1 - recordings2)) < 1e-8


def test_compose_net():
    """Test inserting to cell and composing to net equals inserting to net."""
    ncomp_per_branch = 4
    dt = 0.025
    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp_per_branch)

    cell1 = jx.Cell(branch, parents=[-1, 0, 0])
    cell1.insert(HH())
    cell1.insert(CaPump())
    cell2 = jx.Cell(branch, parents=[-1, 0, 0])
    cell2.insert(CaPump())
    net1 = jx.Network([cell1, cell2])
    net1.cell(0).branch(0).loc(0.0).record()
    net1.cell(0).branch(0).loc(0.0).record("CaCon_i")
    net1.cell(0).branch(0).loc(0.0).stimulate(current)

    cell = jx.Cell(branch, parents=[-1, 0, 0])
    net2 = jx.Network([cell, cell])
    net2.cell(0).insert(HH())
    net2.insert(CaPump())
    net2.cell(0).branch(0).loc(0.0).record()
    net2.cell(0).branch(0).loc(0.0).record("CaCon_i")
    net2.cell(0).branch(0).loc(0.0).stimulate(current)

    assert net1.pumped_ions == net2.pumped_ions == ["CaCon_i"]
    recordings1 = jx.integrate(net1, delta_t=dt)
    recordings2 = jx.integrate(net2, delta_t=dt)

    assert jnp.max(jnp.abs(recordings1 - recordings2)) < 1e-8

    cell2.insert(CaPump("OtherCaPump"))
    assert jx.Network([cell1, cell2]).pumped_ions == ["CaCon_i"]
