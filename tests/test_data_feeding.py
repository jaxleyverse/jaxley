# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit

import jaxley as jx
from tests import builders as build


@pytest.mark.parametrize("with_stimulus", [False, True])
def test_data_stimuli_add_up(with_stimulus):
    """Two data stimuli in a jitted simulation (optionally on top of a regular
    stimulus) act like one stimulus with the summed amplitude.

    With the regular stimulus, the cell also has a trainable parameter (#467).
    """
    cell = build.cell(nbranches=3, ncomp=2)
    cell.branch(0).loc(0.0).record("v")
    i_amp_const = 0.1 if with_stimulus else 0.0
    if with_stimulus:
        cell.make_trainable("radius")
        current = jx.step_current(0.5, 1.0, i_amp_const, 0.025, 5.0)
        cell.branch(1).loc(0.6).stimulate(current)

    @jit
    def simulate(i_amps):
        currents = jx.datapoint_to_step_currents(0.5, 1.0, i_amps, 0.025, 5.0)
        data_stimuli = cell.branch(1).loc(0.6).data_stimulate(currents[0], None)
        data_stimuli = cell.branch(1).loc(0.6).data_stimulate(currents[1], data_stimuli)
        return jx.integrate(cell, data_stimuli=data_stimuli)

    i_amps_data = jnp.asarray([0.01, 0.005])
    v_data = simulate(i_amps_data)

    cell.delete_stimuli()
    i_amp_summed = i_amp_const + jnp.sum(i_amps_data)
    cell.branch(1).loc(0.6).stimulate(
        jx.step_current(0.5, 1.0, i_amp_summed, 0.025, 5.0)
    )
    v_stim = jx.integrate(cell)
    np.testing.assert_allclose(v_stim, v_data, atol=1e-8, rtol=0)
