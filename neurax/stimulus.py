from math import pi

import jax.numpy as jnp
from jax import lax


class Stimulus:
    def __init__(self, cell_ind, branch_ind, loc, current: jnp.ndarray):
        """
        Args:
            current: Time series of the current.
        """
        self.cell_ind = cell_ind
        self.branch_ind = branch_ind
        self.loc = loc
        self.current = current


def step_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    time_vec: jnp.asarray,
):
    """
    Return step current in unit nA.
    """
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
    Return external input to each compartment in uA / cm^2.
    """
    zero_vec = jnp.zeros_like(voltages)
    # `radius`: um
    # `length_single_compartment`: um
    # `i_stim`: nA
    current = i_stim / 2 / pi / radius / length_single_compartment  # nA / um^2
    current *= 100_000  # Convert (nA / um^2) to (uA / cm^2)
    stim_at_timestep = zero_vec.at[i_inds].set(current)
    return stim_at_timestep
