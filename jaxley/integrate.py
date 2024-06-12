from math import prod
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import pandas as pd

from jaxley.modules import Module
from jaxley.utils.cell_utils import flip_comp_indices, params_to_pstate
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

    # Initialize the external inputs and their indices.
    externals = {}
    external_inds = {}

    # If stimulus is inserted, add it to the external inputs.
    if "i" in module.externals.keys() or data_stimuli is not None:
        if "i" in module.externals.keys():
            externals["i"] = module.externals["i"]
            external_inds["i"] = module.external_inds["i"]

            if data_stimuli is not None:
                externals["i"] = jnp.concatenate([externals["i"], data_stimuli[0]])
                external_inds["i"] = jnp.concatenate(
                    [external_inds["i"], data_stimuli[1].comp_index.to_numpy()]
                )
        else:
            externals["i"] = data_stimuli[0]
            external_inds["i"] = data_stimuli[1].comp_index.to_numpy()
    else:
        externals = {"i": jnp.asarray([[]]).astype("float")}
        external_inds = {"i": jnp.asarray([]).astype("int32")}

    # Add the rest of the external and indices from .clamp().
    for key in module.externals.keys():
        externals[key] = module.externals[key]
        external_inds[key] = module.external_inds[key]

    if not externals.keys():
        # No stimulus was inserted and no clamp was set.
        assert (
            t_max is not None
        ), "If no stimulus or clamp are inserted you have to specify the simulation duration at `jx.integrate(..., t_max=)`."

    for key in externals.keys():
        externals[key] = externals[key].T  # Shape `(time, num_stimuli)`.
        external_inds[key] = flip_comp_indices(
            external_inds[key], module.nseg
        )  # See #305

    rec_inds = module.recordings.rec_index.to_numpy()
    rec_inds = flip_comp_indices(rec_inds, module.nseg)  # See #305
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

    # Run `init_conds()` and return every parameter that is needed to solve the ODE.
    # This includes conductances, radiuses, lenghts, axial_resistivities, but also
    # coupling conductances.
    all_params = module.get_all_parameters(pstate)

    def _body_fun(state, externals):
        state = module.step(
            state,
            delta_t,
            external_inds,
            externals,
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
    example_key = list(externals.keys())[0]
    nsteps_to_return = len(externals[example_key])
    if checkpoint_lengths is None:
        checkpoint_lengths = [len(externals[example_key])]
        length = len(externals[example_key])
    else:
        length = prod(checkpoint_lengths)
        dummy_external = jnp.zeros((size_difference, externals[example_key].shape[1]))
        assert (
            len(externals[example_key]) <= length
        ), "The desired simulation duration is longer than `prod(nested_length)`."
        size_difference = length - len(externals[example_key])
        for key in externals.keys():
            externals[key] = jnp.concatenate([externals[key], dummy_external])

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
                if key not in module.synapse_state_names:
                    inds = flip_comp_indices(inds, module.nseg)  # See 305
                # `inds` is of shape `(num_params, num_comps_per_param)`.
                # `set_param` is of shape `(num_params,)`
                # We need to unsqueeze `set_param` to make it `(num_params, 1)` for the
                # `.set()` to work. This is done with `[:, None]`.
                states[key] = states[key].at[inds].set(set_param[key][:, None])

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
        _body_fun,
        states,
        externals,
        length=length,
        nested_lengths=checkpoint_lengths,
    )
    return jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T
