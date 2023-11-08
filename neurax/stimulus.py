from math import pi
from typing import List, Optional

import jax.numpy as jnp
from jax.lax import ScatterDimensionNumbers, scatter_add

from neurax.utils.cell_utils import index_of_loc


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
    """
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)
    time_steps = int(t_max // dt) + 2
    current = jnp.zeros((time_steps,)) + i_offset
    return current.at[window_start:window_end].set(i_amp)


def step_dataset(
    i_delay: float,
    i_dur: float,
    i_amp: jnp.asarray,
    delta_t: float,
    t_max: float,
    i_offset: float = 0.0,
):
    """
    Return several step currents in unit nA.
    """
    dim = len(i_amp)
    dt = delta_t
    window_start = int(i_delay / dt)
    window_end = int((i_delay + i_dur) / dt)

    time_steps = int(t_max // dt) + 2
    current = jnp.zeros((time_steps, dim)) + i_offset
    return current.at[window_start:window_end, :].set(i_amp).T


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
