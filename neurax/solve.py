from typing import List

import jax.numpy as jnp
from jax.lax import fori_loop

from neurax.stimulus import Stimulus
from neurax.recording import Recording
from neurax.modules import Module
from neurax.utils.cell_utils import index_of_loc


def solve(
    module: Module,
    stimuli: List[Stimulus],
    recordings: List[Recording],
    dt: float = 0.025,
    nsteps: int = 300,
) -> jnp.ndarray:
    """Solve the ODE and return recorded voltages."""
    assert module.initialized, "Module is not initialized, run `.initialize()`."

    nseg = module.nseg
    cumsum_nbranches = module.cumsum_nbranches

    i_current, i_inds = prepare_stim(stimuli, nseg, cumsum_nbranches)
    rec_inds = prepare_recs(recordings, nseg, cumsum_nbranches)
    saveat = jnp.zeros((nsteps, len(recordings)))

    u = module.states

    def body_fun(i, state):
        u, saveat = state
        u = module.step(u, dt, i_inds, i_current[i])
        saveat = saveat.at[i, :].set(u["voltages"][rec_inds])
        return (u, saveat)

    final_state = fori_loop(0, 300, body_fun, (u, saveat))
    return final_state[-1]


def prepare_recs(recordings: List[Recording], nseg: int, cumsum_nbranches):
    rec_comp_inds = [index_of_loc(r.branch_ind, r.loc, nseg) for r in recordings]
    rec_comp_inds = jnp.asarray(rec_comp_inds)
    rec_branch_inds = jnp.asarray([r.cell_ind for r in recordings])
    rec_branch_inds = nseg * cumsum_nbranches[rec_branch_inds]
    return rec_branch_inds + rec_comp_inds


def prepare_stim(stimuli: List[Stimulus], nseg: int, cumsum_nbranches):
    i_comp_inds = [index_of_loc(s.branch_ind, s.loc, nseg) for s in stimuli]
    i_comp_inds = jnp.asarray(i_comp_inds)

    i_branch_inds = jnp.asarray([s.cell_ind for s in stimuli])
    i_branch_inds = cumsum_nbranches[i_branch_inds] * nseg
    i_ext = jnp.asarray([s.current for s in stimuli]).T  # nA
    return i_ext, i_branch_inds + i_comp_inds
