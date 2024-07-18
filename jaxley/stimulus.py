# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax.numpy as jnp


def step_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    delta_t: float,
    t_max: float,
    i_offset: float = 0.0,
):
    """
    Return step current in unit nA.

    Unlike the `datapoint_to_step()` method, this takes a single value for the amplitude
    and returns a single step current. The output of this function can be passed to
    `.stimulate()`, but not to `integrate(..., currents=)`.
    """
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = jnp.zeros((time_steps,)) + i_offset
    return current.at[window_start:window_end].set(i_amp)


def datapoint_to_step_currents(
    i_delay: float,
    i_dur: float,
    i_amp: jnp.asarray,
    delta_t: float,
    t_max: float,
    i_offset: float = 0.0,
):
    """
    Return several step currents in unit nA.

    Unlike the `step_current()` method, this takes a vector of amplitude and returns
    a step current for each value in the vector. The output of this function can be
    passed to `integrate(..., currents=)`, but can not be passed to `.stimulate()`.
    """
    dim = len(i_amp)
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = jnp.zeros((time_steps, dim)) + i_offset
    return current.at[window_start:window_end, :].set(i_amp).T
