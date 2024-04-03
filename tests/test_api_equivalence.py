import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.connection import connect
from jaxley.synapses import IonotropicSynapse


def test_api_equivalence_morphology():
    """Test the API for how one can build morphologies from scratch."""
    nseg_per_branch = 2
    depth = 2
    dt = 0.025

    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)
    num_branches = len(parents)

    comp = jx.Compartment().initialize()

    branch1 = jx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell1 = jx.Cell(
        [branch1 for _ in range(num_branches)], parents=parents
    ).initialize()

    branch2 = jx.Branch(comp, nseg=nseg_per_branch).initialize()
    cell2 = jx.Cell(branch2, parents=parents).initialize()

    cell1.branch(2).loc(0.4).record()
    cell2.branch(2).loc(0.4).record()

    current = jx.step_current(0.5, 1.0, 1.0, dt, 3.0)
    cell1.branch(1).loc(1.0).stimulate(current)
    cell2.branch(1).loc(1.0).stimulate(current)

    voltages1 = jx.integrate(cell1, delta_t=dt)
    voltages2 = jx.integrate(cell2, delta_t=dt)
    assert (
        jnp.max(jnp.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between morphology APIs."


def test_api_equivalence_synapses():
    """Test whether ways of adding synapses are equivalent."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell1 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
    cell2 = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])

    conns = [
        jx.Connectivity(
            IonotropicSynapse(),
            [jx.Connection(0, 0, 1.0, 1, 4, 1.0), jx.Connection(1, 1, 0.8, 0, 4, 0.1)],
        )
    ]
    net1 = jx.Network([cell1, cell2], conns)

    net2 = jx.Network([cell1, cell2])
    pre = net2.cell(0).branch(0).loc(1.0)
    post = net2.cell(1).branch(4).loc(1.0)
    connect(pre, post, IonotropicSynapse())

    pre = net2.cell(1).branch(1).loc(0.8)
    post = net2.cell(0).branch(4).loc(0.1)
    connect(pre, post, IonotropicSynapse())

    for net in [net1, net2]:
        current = jx.step_current(0.5, 1.0, 0.5, 0.025, 5.0)
        net.cell(0).branch(0).loc(0.0).stimulate(current)
        net.cell(0).branch(0).loc(0.5).record()
        net.cell(1).branch(4).loc(0.5).record()

    voltages1 = jx.integrate(net1)
    voltages2 = jx.integrate(net2)

    assert (
        np.max(np.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between synapse APIs."
