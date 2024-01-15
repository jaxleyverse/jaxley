import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import GlutamateSynapse, TestSynapse

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


def test_record_diverse_states():
    """Tests recording of membrane and synaptic states."""

    _ = np.random.seed(0)  # Seed because connectivity is at random postsyn locs.

    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(3)])
    net.insert(HH())

    net.cell([0]).fully_connect(net.cell([1]), GlutamateSynapse())
    net.cell([1]).fully_connect(net.cell([2]), TestSynapse())
    net.cell([2]).fully_connect(net.cell([0]), GlutamateSynapse())

    current = jx.step_current(1.0, 80.0, 0.02, 0.025, 100.0)
    net.cell(0).branch(0).comp(0.0).stimulate(current)

    net.cell(2).branch(0).comp(0.0).record("voltages")
    net.GlutamateSynapse(1).record("s")
    net.cell(1).branch(0).comp(0.0).record("voltages")
    net.TestSynapse(0).record("c")

    recs = jx.integrate(net)

    # Loop over first two recorings and then the second two recordings.
    for index in [0, 2]:
        # Local maxima of voltage trace.
        y = recs[index]
        maxima_1 = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
        max_vals = recs[index][maxima_1]
        condition = max_vals > 10.0
        maxima_1 = maxima_1[condition]
        
        # Local maxima of synaptic state.
        y = recs[index+1]
        maxima_2 = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1

        offset = 10  # On average the synaptic trace spikes around 10 steps after voltage.
        assert np.all(np.abs(maxima_2 - maxima_1 - offset)) < 5.0
