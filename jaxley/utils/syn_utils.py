from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from jax.lax import ScatterDimensionNumbers, scatter_add

from jaxley.utils.cell_utils import index_of_loc


def gather_synapes(
    number_of_compartments: jnp.ndarray,
    post_syn_comp_inds: np.ndarray,
    current_each_synapse_voltage_term: jnp.ndarray,
    current_each_synapse_constant_term: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute current at the post synapse.
    
    All this does it that it sums the synaptic currents that come into a particular
    compartment. It returns an array of as many elements as there are compartments.
    """
    incoming_currents_voltages = jnp.zeros((number_of_compartments,))
    incoming_currents_contant = jnp.zeros((number_of_compartments,))

    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    incoming_currents_voltages = scatter_add(
        incoming_currents_voltages,
        post_syn_comp_inds[:, None],
        current_each_synapse_voltage_term,
        dnums,
    )
    incoming_currents_contant = scatter_add(
        incoming_currents_contant,
        post_syn_comp_inds[:, None],
        current_each_synapse_constant_term,
        dnums,
    )
    return incoming_currents_voltages, incoming_currents_contant


def prepare_syn(
    conns: List["jx.Connection"], nseg_per_branch: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare synapses by computing the pre and post compartment within each cell."""
    pre_syn_inds = [
        index_of_loc(c.pre_branch_ind, c.pre_loc, nseg_per_branch) for c in conns
    ]
    pre_syn_inds = jnp.asarray(pre_syn_inds)
    pre_syn_cell_inds = jnp.asarray([c.pre_cell_ind for c in conns])

    post_syn_inds = [
        index_of_loc(c.post_branch_ind, c.post_loc, nseg_per_branch) for c in conns
    ]
    post_syn_inds = jnp.asarray(post_syn_inds)
    post_syn_cell_inds = jnp.asarray([c.post_cell_ind for c in conns])

    return pre_syn_cell_inds, pre_syn_inds, post_syn_cell_inds, post_syn_inds
