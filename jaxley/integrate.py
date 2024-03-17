from math import prod
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import pandas as pd

from jaxley.modules import Module
from jaxley.utils.cell_utils import params_to_pstate
from jaxley.utils.jax_utils import nested_checkpoint_scan


def integrate(
    module: Module,
    params: List[Dict[str, jnp.ndarray]] = [],
    *,
    param_state: Optional[List[Dict]] = None,
    data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
    t_max: Optional[float] = None,
    delta_t: float = 0.025,
    solver: str = "bwd_euler",
    tridiag_solver: str = "stone",
    checkpoint_lengths: Optional[List[int]] = None,
) -> jnp.ndarray:
    """
    Solves ODE and simulates neuron model.

    Args:
        params: Trainable parameters returned by `get_parameters()`.
        param_state: Parameters returned by `data_set`.
        data_stimuli: Outputs of `.data_stimulate()`, only needed if stimuli change
            across function calls.
        t_max: Duration of the simulation in milliseconds. If `t_max` is greater than
            the length of the stimulus input, the stimulus will be padded at the end
            with zeros. If `t_max` is smaller, then the stimulus with be truncated.
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
    module.to_jax()  # Creates `.jaxnodes` from `.nodes` and `.jaxedges` from `.edges`.

    # At least one stimulus was inserted.
    if module.currents is not None or data_stimuli is not None:
        if module.currents is not None:
            i_current = module.currents  # Shape `(num_stimuli, time)`.
            i_inds = module.current_inds.comp_index.to_numpy()

            if data_stimuli is not None:
                # Append stimuli from `data_stimuli`.
                i_current = jnp.concatenate([i_current, data_stimuli[0]])
                i_inds = jnp.concatenate(
                    [i_inds, data_stimuli[1].comp_index.to_numpy()]
                )
        else:
            i_current = data_stimuli[0]  # Shape `(num_stimuli, time)`
            i_inds = data_stimuli[1].comp_index.to_numpy()

        i_current = i_current.T  # Shape `(time, num_stimuli)`.
    else:
        # No stimulus was inserted.
        i_current = jnp.asarray([[]]).astype("int32")
        i_inds = jnp.asarray([]).astype("int32")
        assert (
            t_max is not None
        ), "If no stimulus is inserted that you have to specify the simulation duration at `jx.integrate(..., t_max=)`."

    rec_inds = module.recordings.rec_index.to_numpy()
    rec_states = module.recordings.state.to_numpy()

    # Shorten or pad stimulus depending on `t_max`.
    if t_max is not None:
        t_max_steps = int(t_max // delta_t + 1)
        if t_max_steps > i_current.shape[0]:
            pad = jnp.zeros((t_max_steps - i_current.shape[0], i_current.shape[1]))
            i_current = jnp.concatenate((i_current, pad))
        else:
            i_current = i_current[:t_max_steps, :]

    # Make the `trainable_params` of the same shape as the `param_state`, such that they
    # can be processed together by `get_all_parameters`.
    pstate = params_to_pstate(params, module.indices_set_by_trainables)

    # Gather parameters from `make_trainable` and `data_set` into a single list.
    if param_state is not None:
        pstate += param_state

    # Run `init_conds()` and return every parameter that is needed to solve the ODE.
    # This includes conductances, radiuses, lenghts, axial_resistivities, but also
    # coupling conductances.
    all_params = module.get_all_parameters(pstate)

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
        recs = jnp.asarray(
            [
                state[rec_state][rec_ind]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ]
        )
        return state, recs

    # If necessary, pad the stimulus with zeros in order to simulate sufficiently long.
    # The total simulation length will be `prod(checkpoint_lengths)`. At the end, we
    # return only the first `nsteps_to_return` elements (plus the initial state).
    nsteps_to_return = len(i_current)
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

    # Join node and edge states into a single state dictionary.
    states = {"v": module.jaxnodes["v"]}
    for channel in module.channels:
        for channel_states in list(channel.channel_states.keys()):
            states[channel_states] = module.jaxnodes[channel_states]
    for synapse_states in module.synapse_state_names:
        states[synapse_states] = module.jaxedges[synapse_states]

    # Override with the initial states set by `.make_trainable()`.
    for inds, set_param in zip(module.indices_set_by_trainables, params):
        for key in set_param.keys():
            if key in list(states.keys()):  # Only initial states, not parameters.
                states[key] = states[key].at[inds].set(set_param[key])

    # Add to the states the initial current through every channel.
    states, _ = module._channel_currents(
        states, delta_t, module.channels, module.nodes, all_params
    )

    # Add to the states the initial current through every synapse.
    states, _ = module._synapse_currents(
        states, module.synapses, all_params, delta_t, module.edges
    )

    # Record the initial state.
    init_recs = jnp.asarray(
        [states[rec_state][rec_ind] for rec_state, rec_ind in zip(rec_states, rec_inds)]
    )
    init_recording = jnp.expand_dims(init_recs, axis=0)

    # Run simulation.
    _, recordings = nested_checkpoint_scan(
        _body_fun, states, i_current, length=length, nested_lengths=checkpoint_lengths
    )
    return jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T
