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
    """Return time series of a step current.

    Unlike the `datapoint_to_step()` method, this takes a single value for the amplitude
    and returns a single step current. The output of this function can be passed to
    `.stimulate()`, but not to `integrate(..., currents=)`.

    Args:
        i_delay: Delay in ms until the stimulus turns on.
        i_dur: Duration of the stimulus in ms.
        i_amp: Stimulus amplitude.
        delta_t: Time step in ms.
        t_max: Maximal time.
        i_offset: An offset that is added to the baseline current.

    Returns:
        A tuple of a time vector and a stimulus.

    Example Usage:
    ^^^^^^^^^^^^^^

    ::

        import jaxley as jx

        time, current = jx.step_current(10.0, 80.0, 0.1, 0.025, 100.0)
        plt.plot(time, current)

        cell = jx.Cell()
        cell.stimulate((time, current))
    """
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = jnp.zeros((time_steps,)) + i_offset

    time_vec = jnp.arange(0, t_max + dt, dt)
    return time_vec, current.at[window_start:window_end].set(i_amp)


def datapoint_to_step_currents(
    i_delay: float,
    i_dur: float,
    i_amp: jnp.asarray,
    delta_t: float,
    t_max: float,
    i_offset: float = 0.0,
):
    """Return time series of a several step currents with different amplitudes.

    Unlike the `step_current()` method, this takes a vector of amplitude and returns
    a step current for each value in the vector. The output of this function can be
    passed to `integrate(..., currents=)`, but can not be passed to `.stimulate()`.
    
    Args:
        i_delay: Delay in ms until the stimulus turns on.
        i_dur: Duration of the stimulus in ms.
        i_amp: An array of N stimulus amplitudes.
        delta_t: Time step in ms.
        t_max: Maximal time.
        i_offset: An offset that is added to the baseline current.

    Returns:
        A tuple of a time vector (shape (T,)) and N stimuli (shape (N, T)).

    Example Usage:
    ^^^^^^^^^^^^^^

    ::

        import jaxley as jx

        time, currents = jx.datapoint_to_step_currents(
            10.0, 80.0, np.asarray([0.1, 2.0]), 0.025, 100.0
        )
        plt.plot(time, currents.T)

        comp = jx.Compartment()
        branch = jx.Branch(comp, 2)
        branch.stimulate((time, currents))
    """
    dim = len(i_amp)
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = jnp.zeros((time_steps, dim)) + i_offset

    time_vec = jnp.arange(0, t_max + dt, dt)
    return time_vec, current.at[window_start:window_end, :].set(i_amp).T
