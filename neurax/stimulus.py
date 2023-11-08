from math import pi
from typing import List, Optional

import jax.numpy as jnp
from jax.lax import ScatterDimensionNumbers, scatter_add

from neurax.utils.cell_utils import index_of_loc


def step_current(
    i_delay: float,
    i_dur: float,
    i_amp: float,
    time_vec: jnp.asarray,
    i_offset: float = 0.0,
):
    """
    Return step current in unit nA.
    """
    zero_vec = jnp.zeros_like(time_vec) + i_offset
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

    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    stim_at_timestep = scatter_add(zero_vec, i_inds[:, None], current, dnums)
    return stim_at_timestep
