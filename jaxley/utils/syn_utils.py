from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
from jax.lax import ScatterDimensionNumbers, scatter_add
from jax import vmap

from jaxley.utils.cell_utils import index_of_loc


def postsyn_voltage_updates(
    voltages: jnp.ndarray,
    post_syn_comp_inds: np.ndarray,
    current_each_synapse: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute current at the post synapse."""
    incoming_currents = jnp.zeros_like(voltages)

    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    incoming_current_at_each_comp = scatter_add(
        incoming_currents, post_syn_comp_inds[:, None], current_each_synapse, dnums
    )

    return incoming_current_at_each_comp


vmapped_postsyn_updates = vmap(postsyn_voltage_updates, in_axes=(0, None, 0))


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
