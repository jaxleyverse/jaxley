from typing import Tuple, List
import jax.numpy as jnp
from neurax.utils.cell_utils import index_of_loc


def prepare_synapses(conns, nseg_per_branch):
    """Group the synapses by the number of inputs into each compartment.

    This is a separate function that has to be run by the user because `jnp.unique`
    does not support jit-compilation.
    """
    post_syn_inds = [
        index_of_loc(c.post_branch_ind, c.post_loc, nseg_per_branch) for c in conns
    ]
    post_syn_inds = jnp.asarray(post_syn_inds)
    post_syn_cell_inds = jnp.asarray([c.post_cell_ind for c in conns])

    post_syn = jnp.stack([post_syn_cell_inds, post_syn_inds]).T
    grouped_post_syn_inds, grouped_post_syns, _ = group_by_num_occurences_and_vals(
        post_syn
    )
    return grouped_post_syn_inds, grouped_post_syns


def group_by_num_occurences_and_vals(
    arr: jnp.ndarray,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], jnp.ndarray]:
    """Group an array by the number of each element and the value of the element itself.

    Args:
        arr: The array. Should be 2D.

    Returns:
        - List of arrays which correspond to the grouped indizes. The n-th element
        in the list occurs a particular number of times, specified by the n-th
        element of `num_occ` (see third returned arg).
        - List of arrays which corresponds to the respective elements.
        - List of number of occurances.

    Example:
    ```
    np.random.seed(0)
    nvals = 6
    arr = jnp.asarray(np.random.randint(0, 3, (nvals, 2)))

    grouped, vals, num_occ = group_by_num_occurences_and_vals(a)
    # -> grouped:
    # [
    #     Array([[4], [2]]),
    #     Array([[0, 1], [3, 5]]),
    # ]
    # -> vals:
    # [
    #     Array([[0, 0], [1, 2]]),
    #     Array([[0, 1], [0, 2]]),
    # ]
    # -> num_occ:
    # Array([1, 2])
    ```

    This result can be interpreted as follows:
    - the value `[0, 0]` occurs once, at index 4.
    - the value `[1, 2]` occurs once, at index 2.
    - the value `[0, 1]` occurs twice, at indizes 0 and 1.
    - the value `[0, 2]` occurs twice, at indizes 3 and 5.
    """
    unique_arr, counts = jnp.unique(
        arr,
        axis=0,
        return_counts=True,
    )
    grouped_inds = []
    vals = []
    unique_counts = jnp.unique(counts)
    for count in unique_counts:
        vals_to_match = unique_arr[counts == count]
        inds_n_times = _inds_that_match_vals(arr, vals_to_match)
        grouped_inds.append(inds_n_times)
        vals.append(vals_to_match)
    return grouped_inds, vals, unique_counts


def _inds_that_match_vals(arr, vals_to_match):
    grouped_inds = []
    for val in vals_to_match:
        ind_arr_matches_val = jnp.where(jnp.all(arr == val, axis=1))[0]
        grouped_inds.append(ind_arr_matches_val)
    return jnp.asarray(grouped_inds)
