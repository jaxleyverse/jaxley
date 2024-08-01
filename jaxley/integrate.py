# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

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
    voltage_solver: str = "jaxley.stone",
    checkpoint_lengths: Optional[List[int]] = None,
    all_states: Optional[Dict] = None,
    return_states: bool = False,
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
        all_states: An optional initial state that was returned by a previous
            `jx.integrate(..., return_states=True)` run. Overrides potentially
            trainable initial states.
        return_states: If True, it returns all states such that the current state of
            the `Module` can be set with `set_states`.
    """

    assert module.initialized, "Module is not initialized, run `.initialize()`."
    module.to_jax()  # Creates `.jaxnodes` from `.nodes` and `.jaxedges` from `.edges`.

    # Initialize the external inputs and their indices.
    externals = module.externals.copy()
    external_inds = module.external_inds.copy()

    # If stimulus is inserted, add it to the external inputs.
    if "i" in module.externals.keys() or data_stimuli is not None:
        if "i" in module.externals.keys():
            if data_stimuli is not None:
                externals["i"] = jnp.concatenate([externals["i"], data_stimuli[0]])
                external_inds["i"] = jnp.concatenate(
                    [external_inds["i"], data_stimuli[1].comp_index.to_numpy()]
                )
        else:
            externals["i"] = data_stimuli[0]
            external_inds["i"] = data_stimuli[1].comp_index.to_numpy()
    else:
        externals["i"] = jnp.asarray([[]]).astype("float")
        external_inds["i"] = jnp.asarray([]).astype("int32")

    if not externals.keys():
        # No stimulus was inserted and no clamp was set.
        assert (
            t_max is not None
        ), "If no stimulus or clamp are inserted you have to specify the simulation duration at `jx.integrate(..., t_max=)`."

    for key in externals.keys():
        externals[key] = externals[key].T  # Shape `(time, num_stimuli)`.

    rec_inds = module.recordings.rec_index.to_numpy()
    rec_states = module.recordings.state.to_numpy()

    # Shorten or pad stimulus depending on `t_max`.
    if t_max is not None:
        t_max_steps = int(t_max // delta_t + 1)

        # Pad or truncate the stimulus.
        if "i" in externals.keys() and t_max_steps > externals["i"].shape[0]:
            pad = jnp.zeros(
                (t_max_steps - externals["i"].shape[0], externals["i"].shape[1])
            )
            externals["i"] = jnp.concatenate((externals["i"], pad))

        for key in externals.keys():
            if t_max_steps > externals[key].shape[0]:
                raise NotImplementedError(
                    "clamp must be at least as long as simulation."
                )
            else:
                externals[key] = externals[key][:t_max_steps, :]

    # Make the `trainable_params` of the same shape as the `param_state`, such that they
    # can be processed together by `get_all_parameters`.
    pstate = params_to_pstate(params, module.indices_set_by_trainables)

    # Gather parameters from `make_trainable` and `data_set` into a single list.
    if param_state is not None:
        pstate += param_state

    all_params = module.get_all_parameters(pstate)
    all_states = (
        module.get_all_states(pstate, all_params, delta_t)
        if all_states is None
        else all_states
    )

    def _body_fun(state, externals):
        state = module.step(
            state,
            delta_t,
            external_inds,
            externals,
            params=all_params,
            solver=solver,
            voltage_solver=voltage_solver,
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
    example_key = list(externals.keys())[0]
    nsteps_to_return = len(externals[example_key])
    if checkpoint_lengths is None:
        checkpoint_lengths = [len(externals[example_key])]
        length = len(externals[example_key])
    else:
        length = prod(checkpoint_lengths)
        size_difference = length - len(externals[example_key])
        dummy_external = jnp.zeros((size_difference, externals[example_key].shape[1]))
        assert (
            len(externals[example_key]) <= length
        ), "The desired simulation duration is longer than `prod(nested_length)`."
        for key in externals.keys():
            externals[key] = jnp.concatenate([externals[key], dummy_external])

    # Record the initial state.
    init_recs = jnp.asarray(
        [
            all_states[rec_state][rec_ind]
            for rec_state, rec_ind in zip(rec_states, rec_inds)
        ]
    )
    init_recording = jnp.expand_dims(init_recs, axis=0)

    # Run simulation.
    all_states, recordings = nested_checkpoint_scan(
        _body_fun,
        all_states,
        externals,
        length=length,
        nested_lengths=checkpoint_lengths,
    )
    recs = jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T
    return (recs, all_states) if return_states else recs
