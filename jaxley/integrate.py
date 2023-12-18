from math import prod
from typing import Dict, List, Optional, Union

import jax.numpy as jnp

from jaxley.modules import Module
from jaxley.utils.jax_utils import nested_checkpoint_scan


def integrate(
    module: Module,
    params: List[Dict[str, jnp.ndarray]] = [],
    currents: Optional[jnp.ndarray] = None,
    *,
    t_max: Optional[float] = None,
    delta_t: float = 0.025,
    solver: str = "bwd_euler",
    tridiag_solver: str = "stone",
    checkpoint_lengths: Optional[List[int]] = None,
) -> jnp.ndarray:
    """
    Solves ODE and simulates neuron model.

    Args:
        t_max: Duration of the simulation in milliseconds.
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
    module.to_jax()  # Creates `.jaxnodes` from `.nodes`.

    if module.currents is not None:
        # At least one stimulus was inserted.
        i_current = currents.T if currents is not None else module.currents.T
        i_inds = module.current_inds.comp_index.to_numpy()
    else:
        # No stimulus was inserted.
        i_current = jnp.asarray([[]]).astype("int")
        i_inds = jnp.asarray([]).astype("int")
        assert (
            t_max is not None
        ), "If no stimulus is inserted that you have to specify the simulation duration at `jx.integrate(..., t_max=)`."

    rec_inds = module.recordings.comp_index.to_numpy()

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
    init_recording = jnp.expand_dims(module.jaxnodes["voltages"][rec_inds], axis=0)

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
    states = {"voltages": module.jaxnodes["voltages"]}
    for channel in module.channels:
        for channel_states in list(channel.channel_states.keys()):
            states[channel_states] = module.jaxnodes[channel_states]

    # Override with the initial states set by `.make_trainable()`.
    for inds, set_param in zip(module.indices_set_by_trainables, params):
        for key in set_param.keys():
            if key in list(states.keys()):  # Only initial states, not parameters.
                states[key] = states[key].at[inds].set(set_param[key])

    # Write synaptic states. TODO move above when new interface for synapses.
    for key in module.syn_states:
        states[key] = module.syn_states[key]

    # Run simulation.
    _, recordings = nested_checkpoint_scan(
        _body_fun, states, i_current, length=length, nested_lengths=checkpoint_lengths
    )
    return jnp.concatenate([init_recording, recordings[:nsteps_to_return]], axis=0).T
