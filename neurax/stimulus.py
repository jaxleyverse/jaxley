from math import pi

import jax.numpy as jnp
from jax import lax


class Stimulus:
    def __init__(self, i_delay, i_dur, i_amp):
        self.i_delay = i_delay
        self.i_dur = i_dur
        self.i_amp = i_amp


def get_external_input(
    voltages: jnp.ndarray,
    t: float,
    i_delay: float,
    i_dur: float,
    i_amp: float,
    radius: float,
    length_single_compartment: float,
    nseg_per_branch: int,
):
    """
    Compute external input to each compartment.
    """
    zero_vec = jnp.zeros_like(voltages)
    stim_on = jnp.greater_equal(t, i_delay)
    stim_off = jnp.less_equal(t, i_delay + i_dur)
    stim_ = jnp.logical_and(stim_on, stim_off)
    current_in_comp = i_amp / 2 / pi / radius / length_single_compartment
    external_currents = lax.cond(
        stim_,
        lambda x: x.at[nseg_per_branch - 1].set(current_in_comp),
        lambda x: x,
        zero_vec,
    )
    return external_currents
