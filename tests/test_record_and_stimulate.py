# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse


def test_record_and_stimulate_api():
    """Test the API for recording and stimulating."""
    nseg_per_branch = 2
    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    cell = jx.Cell(branch, parents=parents)

    cell.branch(0).loc(0.0).record()
    cell.branch(1).loc(1.0).record()

    current = jx.step_current(0.0, 1.0, 1.0, 0.025, 3.0)
    cell.branch(1).loc(1.0).stimulate(current)

    cell.delete_recordings()
    cell.delete_stimuli()


def test_record_shape():
    """Test the API for recording and stimulating."""
    nseg_per_branch = 2
    depth = 2
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg_per_branch)
    cell = jx.Cell(branch, parents=parents)

    current = jx.step_current(0.0, 1.0, 1.0, 0.025, 3.0)
    cell.branch(1).loc(1.0).stimulate(current)

    cell.branch(0).loc(0.0).record()
    cell.branch(1).loc(1.0).record()
    cell.branch(0).loc(1.0).record()
    cell.delete_recordings()
    cell.branch(2).loc(0.5).record()
    cell.branch(1).loc(0.1).record()

    voltages = jx.integrate(cell)
    assert (
        voltages.shape[0] == 2
    ), f"Shape of recordings ({voltages.shape}) is not right."


def test_record_synaptic_and_membrane_states():
    """Tests recording of synaptic and membrane states.

    Tests are functional, not just API. They test whether the voltage and synaptic
    states spike at (almost) the same times.
    """

    _ = np.random.seed(0)  # Seed because connectivity is at random postsyn locs.

    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, parents=[-1])
    net = jx.Network([cell for _ in range(3)])
    net.insert(HH())

    fully_connect(net.cell([0]), net.cell([1]), IonotropicSynapse())
    fully_connect(net.cell([1]), net.cell([2]), TestSynapse())
    fully_connect(net.cell([2]), net.cell([0]), IonotropicSynapse())

    current = jx.step_current(1.0, 80.0, 0.02, 0.025, 100.0)
    net.cell(0).branch(0).loc(0.0).stimulate(current)

    net.cell(2).branch(0).loc(0.0).record("v")
    net.IonotropicSynapse(1).record("IonotropicSynapse_s")
    net.cell(2).branch(0).loc(0.0).record("HH_m")
    net.cell(1).branch(0).loc(0.0).record("v")
    net.TestSynapse(0).record("TestSynapse_c")
    net.cell(1).branch(0).loc(0.0).record("HH_m")

    recs = jx.integrate(net)

    # Loop over first two recorings and then the second two recordings.
    for index in [0, 3]:
        # Local maxima of voltage trace.
        y = recs[index]
        maxima_1 = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
        max_vals = recs[index][maxima_1]
        condition = max_vals > 10.0
        maxima_1 = maxima_1[condition]

        # Local maxima of synaptic state.
        y = recs[index + 1]
        maxima_2 = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1

        # Local maxima of membrane channel trace.
        y = recs[index + 2]
        maxima_3 = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
        max_vals = recs[index + 2][maxima_3]
        condition = max_vals > 0.3
        maxima_3 = maxima_3[condition]

        # On average the synaptic trace spikes around 10 steps after voltage.
        offset_syn = 10
        assert np.all(np.abs(maxima_2 - maxima_1 - offset_syn)) < 5.0

        # On average the membrane trace spikes around 0 steps after voltage.
        offset_mem = 0
        assert np.all(np.abs(maxima_3 - maxima_1 - offset_mem)) < 5.0
