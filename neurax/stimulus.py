from math import pi

import jax.numpy as jnp
from jax import lax


class Stimulus:
    def __init__(self, branch_ind, loc, current: jnp.ndarray):
        """
        Args:
            current: Time series of the current.
        """
        self.branch_ind = branch_ind
        self.loc = loc
        self.current = current


def step_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    time_vec: jnp.asarray,
):
    zero_vec = jnp.zeros_like(time_vec)
    stim_on = jnp.greater_equal(time_vec, i_delay)
    stim_off = jnp.less_equal(time_vec, i_delay + i_dur)
    protocol_on = jnp.logical_and(stim_on, stim_off)
    return zero_vec.at[protocol_on].set(i_amp)


def get_external_input(
    voltages: jnp.ndarray,
    i_inds: jnp.ndarray,
    i_stim: jnp.ndarray, 
    radius: float,
    length_single_compartment: float,
):
    """
    Compute external input to each compartment.
    """
    zero_vec = jnp.zeros_like(voltages)
    current = i_stim / 2 / pi / radius / length_single_compartment
    stim_at_timestep = zero_vec.at[i_inds].set(current)
    return stim_at_timestep
