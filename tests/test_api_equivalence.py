import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import jaxley as jx


def test_api_equivalence():
    """Test the API for recording and stimulating."""
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

    cell1.branch(2).comp(0.4).record()
    cell2.branch(2).comp(0.4).record()

    current = jx.step_current(0.5, 1.0, 1.0, dt, 3.0)
    cell1.branch(1).comp(1.0).stimulate(current)
    cell2.branch(1).comp(1.0).stimulate(current)

    voltages1 = jx.integrate(cell1, delta_t=dt)
    voltages2 = jx.integrate(cell2, delta_t=dt)
    assert (
        jnp.max(jnp.abs(voltages1 - voltages2)) < 1e-8
    ), "Voltages do not match between APIs."
