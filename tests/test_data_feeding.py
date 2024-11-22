# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jax import jit

import jaxley as jx
from jaxley.channels import HH


def test_constant_and_data_stimulus(SimpleCell):
    cell = SimpleCell(3, 2)
    cell.branch(0).loc(0.0).record("v")

    # test data_stimulate and jit works with trainable parameters see #467
    cell.make_trainable("radius")

    i_amp_const = 0.1
    i_amps_data = jnp.asarray([0.01, 0.005])

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=i_amp_const, delta_t=0.025, t_max=5.0
    )
    cell.branch(1).loc(0.6).stimulate(current)

    def provide_data(i_amps):
        current = jx.datapoint_to_step_currents(
            i_delay=0.5, i_dur=1.0, i_amp=i_amps, delta_t=0.025, t_max=5.0
        )
        data_stimuli = None
        data_stimuli = cell.branch(1).loc(0.6).data_stimulate(current[0], data_stimuli)
        data_stimuli = cell.branch(1).loc(0.6).data_stimulate(current[1], data_stimuli)
        return data_stimuli

    def simulate(i_amps):
        data_stimuli = provide_data(i_amps)
        return jx.integrate(cell, data_stimuli=data_stimuli)

    jitted_simulate = jit(simulate)
    v_data = jitted_simulate(i_amps_data)

    cell.delete_stimuli()
    i_amp_summed = i_amp_const + jnp.sum(i_amps_data)
    current_sum = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=i_amp_summed, delta_t=0.025, t_max=5.0
    )
    cell.branch(1).loc(0.6).stimulate(current_sum)

    v_stim = jx.integrate(cell)

    diff = np.abs(v_stim - v_data)
    assert np.max(diff) < 1e-8


def test_data_vs_constant_stimulus(SimpleCell):
    cell = SimpleCell(3, 2)
    cell.branch(0).loc(0.0).record("v")

    i_amps_data = jnp.asarray([0.01, 0.005])

    def provide_data(i_amps):
        current = jx.datapoint_to_step_currents(0.5, 1.0, i_amps, 0.025, 5.0)
        data_stimuli = None
        data_stimuli = cell.branch(1).loc(0.6).data_stimulate(current[0], data_stimuli)
        data_stimuli = cell.branch(1).loc(0.6).data_stimulate(current[1], data_stimuli)
        return data_stimuli

    def simulate(i_amps):
        data_stimuli = provide_data(i_amps)
        return jx.integrate(cell, data_stimuli=data_stimuli)

    jitted_simulate = jit(simulate)
    v_data = jitted_simulate(i_amps_data)

    cell.delete_stimuli()
    i_amp_summed = jnp.sum(i_amps_data)
    current_sum = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=i_amp_summed, delta_t=0.025, t_max=5.0
    )
    cell.branch(1).loc(0.6).stimulate(current_sum)

    v_stim = jx.integrate(cell)

    diff = np.abs(v_stim - v_data)
    assert np.max(diff) < 1e-8
