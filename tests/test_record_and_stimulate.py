# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import fully_connect
from jaxley.synapses import IonotropicSynapse, TestSynapse


def test_record_and_stimulate_api(SimpleCell):
    """Test the API for recording and stimulating."""
    cell = SimpleCell(3, 2)

    cell.branch(0).loc(0.0).record()
    cell.branch(1).loc(1.0).record()

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    cell.branch(1).loc(1.0).stimulate(current)

    cell.delete_recordings()
    cell.delete_stimuli()


def test_record_shape(SimpleCell):
    """Test the API for recording and stimulating."""
    cell = SimpleCell(3, 2)

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
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


def test_record_synaptic_and_membrane_states(SimpleNet):
    """Tests recording of synaptic and membrane states.

    Tests are functional, not just API. They test whether the voltage and synaptic
    states spike at (almost) the same times.
    """

    _ = np.random.seed(0)  # Seed because connectivity is at random postsyn locs.

    net = SimpleNet(3, 1, 4)
    net.insert(HH())

    fully_connect(net.cell([0]), net.cell([1]), IonotropicSynapse())
    fully_connect(net.cell([1]), net.cell([2]), TestSynapse())
    fully_connect(net.cell([2]), net.cell([0]), IonotropicSynapse())

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    net.cell(0).branch(0).loc(0.0).stimulate(current)

    # Invoke recording of voltage and synaptic states.
    net.cell(2).branch(0).loc(0.0).record("v")
    net.IonotropicSynapse.edge(1).record("IonotropicSynapse_s")
    net.cell(2).branch(0).loc(0.0).record("HH_m")
    net.cell(1).branch(0).loc(0.0).record("v")
    net.TestSynapse.edge(0).record("TestSynapse_c")
    net.cell(1).branch(0).loc(0.0).record("HH_m")
    net.cell(1).branch(0).loc(0.0).record("i_HH")
    net.IonotropicSynapse.edge(1).record("i_IonotropicSynapse")

    # Advanced synapse indexing for recording.
    net.copy_node_property_to_edges("global_cell_index")
    # Record currents from specific post synaptic cells.
    df = net.edges
    df = df.query("pre_global_cell_index in [0, 1]")
    net.select(edges=df.index).record("i_IonotropicSynapse")
    # Record currents from specific synapse types
    df = net.edges
    df = df.query("type == 'TestSynapse'")
    net.select(edges=df.index).record("i_TestSynapse")

    recs = jx.integrate(net)

    # Loop over first two recordings and then the second two recordings.
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


def test_empty_recordings(SimpleComp):
    # Create an empty compartment
    comp = SimpleComp()

    # Check if a ValueError is raised when integrating an empty compartment
    with pytest.raises(ValueError):
        v = jx.integrate(comp, delta_t=0.025, t_max=10.0)
