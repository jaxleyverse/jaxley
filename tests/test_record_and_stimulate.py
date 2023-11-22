import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import jaxley as jx


def test_record_and_stimulate_api():
    """Test the API for recording and stimulating."""
    nseg_per_branch = 2
    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch(comp, nseg_per_branch).initialize()
    cell = jx.Cell(branch, parents=parents).initialize()

    cell.branch(0).comp(0.0).record()
    cell.branch(1).comp(1.0).record()

    current = jx.step_current(0.0, 1.0, 1.0, 0.025, 3.0)
    cell.branch(1).comp(1.0).stimulate(current)

    cell.delete_recordings()
    cell.delete_stimuli()


def test_record_shape():
    """Test the API for recording and stimulating."""
    nseg_per_branch = 2
    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch(comp, nseg_per_branch).initialize()
    cell = jx.Cell(branch, parents=parents).initialize()

    current = jx.step_current(0.0, 1.0, 1.0, 0.025, 3.0)
    cell.branch(1).comp(1.0).stimulate(current)

    cell.branch(0).comp(0.0).record()
    cell.branch(1).comp(1.0).record()
    cell.branch(0).comp(1.0).record()
    cell.delete_recordings()
    cell.branch(2).comp(0.5).record()
    cell.branch(1).comp(0.1).record()

    voltages = jx.integrate(cell)
    assert (
        voltages.shape[0] == 2
    ), f"Shape of recordings ({voltages.shape}) is not right."
