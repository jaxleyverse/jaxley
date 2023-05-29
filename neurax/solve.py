import numpy as np
import jax.numpy as jnp
import jax
from jax.lax import fori_loop
from jax import jit
from neurax.modules import Module
from neurax.utils.cell_utils import index_of_loc


def solve(module: Module, stimuli, recordings):
    """Solve the ODE and return recorded voltages."""
    # Prepare recordings.
    rec_comp_inds = [
        index_of_loc(r.branch_ind, r.loc, NSEG_PER_BRANCH) for r in recordings
    ]
    rec_comp_inds = jnp.asarray(rec_branch_inds)
    rec_branch_inds = jnp.asarray([r.cell_ind for r in recordings])
    rec_branch_inds = NSEG_PER_BRANCH * CUMSUM_NUM_BRANCHES[rec_branch_inds]

    # Prepare stimuli.
    stim_branch_inds = [
        index_of_loc(s.branch_ind, s.loc, NSEG_PER_BRANCH) for s in stimuli
    ]
    i_cell_inds = jnp.asarray([s.cell_ind for s in stimuli])
    i_branch_inds = jnp.asarray(stim_branch_inds)
    i_inds = CUMSUM_NUM_BRANCHES[i_cell_inds] * NSEG_PER_BRANCH + i_branch_inds
    i_ext = jnp.asarray([s.current for s in stimuli]).T  # nA

    u = module.states
    stim = jnp.zeros((1, 8))
    stim = stim.at[0, 0].set(3.0)

    nsteps = 300
    saveat = jnp.zeros((nsteps, 8))

    def body_fun(i, state):
        u, dt, stim, saveat = state
        u = module.step(u, 0.025, stim)
        saveat = saveat.at[i, :].set(u["voltages"][rec_branch_inds, rec_comp_inds])
        return (u, dt, stim, saveat)

    final_state = fori_loop(0, 300, body_fun, (u, 0.025, stim, saveat))
    return final_state[-1]
