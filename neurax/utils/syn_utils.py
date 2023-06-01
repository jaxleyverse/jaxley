from typing import Tuple, List
import jax.numpy as jnp
from neurax.utils.cell_utils import index_of_loc
from jax.lax import ScatterDimensionNumbers, scatter_add


def postsyn_voltage_updates(
    voltages,
    post_syn_comp_inds,
    non_zero_voltage_term,
    non_zero_constant_term,
):
    voltage_term = jnp.zeros_like(voltages)
    constant_term = jnp.zeros_like(voltages)

    dnums = ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    voltage_term = scatter_add(
        voltage_term,
        post_syn_comp_inds[:, None],
        non_zero_voltage_term,
        dnums,
    )
    constant_term = scatter_add(
        constant_term, post_syn_comp_inds[:, None], non_zero_constant_term, dnums
    )

    return voltage_term, constant_term


def prepare_syn(conns, nseg_per_branch):
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
