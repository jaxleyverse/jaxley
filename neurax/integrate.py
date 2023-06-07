from math import prod
from typing import Dict, List, Optional, Union

import jax.numpy as jnp

from neurax.modules import Module
from neurax.recording import Recording
from neurax.stimulus import Stimulus, Stimuli
from neurax.utils.cell_utils import index_of_loc
from neurax.utils.jax_utils import nested_checkpoint_scan


def integrate(
    module: Module,
    stimuli: Union[List[Stimulus], Stimuli],
    recordings: List[Recording],
    params: List[Dict[str, jnp.ndarray]] = [],
    t_max: Optional[float] = None,
    delta_t: float = 0.025,
    solver: str = "bwd_euler",
    tridiag_solver: str = "stone",
    checkpoint_lengths: Optional[List[int]] = None,
) -> jnp.ndarray:
    """
    Solves ODE and simulates neuron model.

    Args:
        t_max: Duration of the simulation in milliseconds. If `None`, the duration is
            inferred from the duration of the stimulus. If it is larger than the
            duration of the stimulus, the stimulus is padded with zeros at the end.
        delta_t: Time step of the solver in milliseconds.
        solver: Which ODE solver to use. Either of ["fwd_euler", "bwd_euler", "cranck"].
        tridiag_solver: Algorithm to solve tridiagonal systems. The  different options
            only affect `bwd_euler` and `cranck` solvers. Either of ["stone",
            "thomas"], where `stone` is much faster on GPU for long branches
            with many compartments and `thomas` is slightly faster on CPU (`thomas` is
            used in NEURON).
        checkpoint_lengths: Number of timesteps at every level of checkpointing. The
            `prod(checkpoint_lengths)` must be larger or equal to the desired number of
            simulated timesteps. Warning: the simulation is run for
            `prod(checkpoint_lengths)` timesteps, and the result is posthoc truncated
            to the desired simulation length. Therefore, a poor choice of
            `checkpoint_lengths` can lead to longer simulation time. If `None`, no
            checkpointing is applied.
    """

    assert module.initialized, "Module is not initialized, run `.initialize()`."

    nseg = module.nseg
    cumsum_nbranches = module.cumsum_nbranches

    i_current, i_inds = prepare_stim(stimuli, nseg, cumsum_nbranches)
    rec_inds = prepare_recs(recordings, nseg, cumsum_nbranches)

    # Shorten or pad stimulus depending on `t_max`.
    if t_max is not None:
        t_max_steps = int(t_max // delta_t + 1)
        if t_max_steps > i_current.shape[0]:
            i_current = jnp.zeros((t_max_steps, i_current.shape[1]))
        else:
            i_current = i_current[:t_max_steps, :]

    # Run `init_conds()` and return every parameter that is needed to solve the ODE.
    # This includes conductances, radiuses, lenghts, axial_resistivities, but also
    # coupling conductances.
    all_params = module.get_all_parameters(params)

    def _body_fun(state, i_stim):
        state = module.step(
            state,
            delta_t,
            i_inds,
            i_stim,
            params=all_params,
            solver=solver,
            tridiag_solver=tridiag_solver,
        )
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
        ), "The desired simulation duration is longer than `prod(nested_length)`."
        size_difference = length - len(i_current)
        dummy_stimulus = jnp.zeros((size_difference, i_current.shape[1]))
        i_current = jnp.concatenate([i_current, dummy_stimulus])

    # Join node and edge states.
    states = {}
    for key in module.states:
        states[key] = module.states[key]
    for key in module.syn_states:
        states[key] = module.syn_states[key]

    _, recordings = nested_checkpoint_scan(
        _body_fun, states, i_current, length=length, nested_lengths=checkpoint_lengths
    )
    return jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T


def prepare_recs(recordings: List[Recording], nseg: int, cumsum_nbranches):
    """Prepare recordings."""
    rec_comp_inds = [index_of_loc(r.branch_ind, r.loc, nseg) for r in recordings]
    rec_comp_inds = jnp.asarray(rec_comp_inds)
    rec_branch_inds = jnp.asarray([r.cell_ind for r in recordings])
    rec_branch_inds = nseg * cumsum_nbranches[rec_branch_inds]
    return rec_branch_inds + rec_comp_inds


def prepare_stim(stimuli: Union[List[Stimulus], Stimuli], nseg: int, cumsum_nbranches):
    """Prepare stimuli."""
    if isinstance(stimuli, Stimuli):
        # Indexing.
        i_comp_inds = stimuli.comp_inds
        i_branch_inds = stimuli.branch_inds

        # Currents.
        i_ext = stimuli.currents  # nA
    else:
        # Indexing.
        i_comp_inds = [index_of_loc(s.branch_ind, s.loc, nseg) for s in stimuli]
        i_comp_inds = jnp.asarray(i_comp_inds)
        i_branch_inds = jnp.asarray([s.cell_ind for s in stimuli])
        i_branch_inds = cumsum_nbranches[i_branch_inds] * nseg

        # Currents.
        i_ext = jnp.asarray([s.current for s in stimuli]).T  # nA

    return i_ext, i_branch_inds + i_comp_inds
