from typing import List, Optional
from math import pi

import jax.numpy as jnp
from jax import lax
from neurax.utils.cell_utils import index_of_loc


class Stimulus:
    """A single stimulus to the network."""

    def __init__(
        self, cell_ind, branch_ind, loc, current: Optional[jnp.ndarray] = None
    ):
        """
        Args:
            current: Time series of the current.
        """
        self.cell_ind = cell_ind
        self.branch_ind = branch_ind
        self.loc = loc
        self.current = current


class Stimuli:
    """Several stimuli to the network.

    Here, the properties of all individual stimuli already get vectorized and put
    into arrays for speed.
    """

    def __init__(self, stims: List[Stimulus], nseg_per_branch: int):
        self.branch_inds = jnp.asarray(
            [index_of_loc(s.branch_ind, s.loc, nseg_per_branch) for s in stims]
        )
        self.cell_inds = jnp.asarray([s.cell_ind for s in stims])
        self.currents = jnp.asarray([s.current for s in stims]).T  # nA

    def set_currents(self, currents: float):
        """Rescale the current of the stimulus with a constant value over time."""
        self.currents = currents
        return self


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
    current = (
        i_stim / 2 / pi / radius[i_inds] / length_single_compartment[i_inds]
    )  # nA / um^2
    current *= 100_000  # Convert (nA / um^2) to (uA / cm^2)
    stim_at_timestep = zero_vec.at[i_inds].set(current)
    return stim_at_timestep
