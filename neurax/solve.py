from typing import List, Optional
from math import prod

import jax.numpy as jnp
from neurax.utils.jax_utils import nested_checkpoint_scan

from neurax.stimulus import Stimulus
from neurax.recording import Recording
from neurax.modules import Module
from neurax.utils.cell_utils import index_of_loc


def solve(
    module: Module,
    stimuli: List[Stimulus],
    recordings: List[Recording],
    delta_t: float = 0.025,
    checkpoint_lengths: Optional[List[int]] = None,
) -> jnp.ndarray:
    """Solve the ODE and return recorded voltages."""
    assert module.initialized, "Module is not initialized, run `.initialize()`."

    nseg = module.nseg
    cumsum_nbranches = module.cumsum_nbranches

    i_current, i_inds = prepare_stim(stimuli, nseg, cumsum_nbranches)
    rec_inds = prepare_recs(recordings, nseg, cumsum_nbranches)

    def _body_fun(state, i_stim):
        state = module.step(state, delta_t, i_inds, i_stim)
        return state, state["voltages"][rec_inds]

    nsteps_to_return = len(i_current)
    init_recording = jnp.expand_dims(module.states["voltages"][rec_inds], axis=0)

    # If necessary, pad the stimulus with zeros in order to simulate sufficiently long.
    # The total simulation length will be `prod(checkpoint_lengths)`. At the end, we
    # return only the first `nsteps_to_return` elements (plus the initial state).
    if checkpoint_lengths is None:
        checkpoint_lengths = [len(i_current)]
        length = len(i_current)
    else:
        length = prod(checkpoint_lengths)
        assert (
            len(i_current) <= length
        ), "The external current is longer than `prod(nested_length)`."
        size_difference = length - len(i_current)
        dummy_stimulus = jnp.zeros((size_difference, i_current.shape[1]))
        i_current = jnp.concatenate([i_current, dummy_stimulus])

    _, recordings = nested_checkpoint_scan(
        _body_fun,
        module.states,
        i_current,
        length=length,
        nested_lengths=checkpoint_lengths,
    )
    return jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T


def prepare_recs(recordings: List[Recording], nseg: int, cumsum_nbranches):
    """Prepare recordings."""
    rec_comp_inds = [index_of_loc(r.branch_ind, r.loc, nseg) for r in recordings]
    rec_comp_inds = jnp.asarray(rec_comp_inds)
    rec_branch_inds = jnp.asarray([r.cell_ind for r in recordings])
    rec_branch_inds = nseg * cumsum_nbranches[rec_branch_inds]
    return rec_branch_inds + rec_comp_inds


def prepare_stim(stimuli: List[Stimulus], nseg: int, cumsum_nbranches):
    """Prepare stimuli."""
    i_comp_inds = [index_of_loc(s.branch_ind, s.loc, nseg) for s in stimuli]
    i_comp_inds = jnp.asarray(i_comp_inds)

    i_branch_inds = jnp.asarray([s.cell_ind for s in stimuli])
    i_branch_inds = cumsum_nbranches[i_branch_inds] * nseg
    i_ext = jnp.asarray([s.current for s in stimuli]).T  # nA
    return i_ext, i_branch_inds + i_comp_inds
