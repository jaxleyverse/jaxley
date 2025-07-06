# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from math import prod
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pandas as pd

from jaxley.modules import Module
from jaxley.utils.cell_utils import params_to_pstate
from jaxley.utils.jax_utils import nested_checkpoint_scan


def build_init_and_step_fn(
    module: Module,
    voltage_solver: str = "jaxley.dhs",
    solver: str = "bwd_euler",
) -> Tuple[Callable, Callable]:
    """Return ``init_fn`` and ``step_fn`` which initialize modules and run update steps.

    This method can be used to gain additional control over the simulation workflow.
    It exposes the ``step`` function, which can be used to perform step-by-step updates
    of the differential equations.

    Args:
        module: A `Module` object that e.g. a cell.
        voltage_solver: Voltage solver used in step. Defaults to "jaxley.stone".
        solver: ODE solver. Defaults to "bwd_euler".

    Returns:
        init_fn, step_fn: Functions that initialize the state and parameters, and
            perform a single integration step, respectively.

    Example usage
    ^^^^^^^^^^^^^

    The following allows you to perform a step-by-step update of the differential
    equations.

    ::

        import jax.numpy as jnp
        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn

        t_max = 3.0
        delta_t = 0.025

        cell = jx.Cell()
        cell.record()
        cell.stimulate(jx.step_current(0, 1, 2, delta_t, t_max))
        params = cell.get_parameters()

        cell.to_jax()
        rec_inds = cell.recordings.rec_index.to_numpy()
        rec_states = cell.recordings.state.to_numpy()
        externals = cell.externals.copy()
        external_inds = cell.external_inds.copy()

        # Uncomment this line if `data_stimuli` is not `None`.
        # externals, external_inds = add_stimuli(externals, external_inds, data_stimuli)

        # Uncomment this line if `data_clamps` is not `None`.
        # externals, external_inds = add_clamps(externals, external_inds, data_clamps)

        # Initialize.
        init_fn, step_fn = build_init_and_step_fn(cell)
        states, params = init_fn(params)
        recordings = [
            states[rec_state][rec_ind][None]
            for rec_state, rec_ind in zip(rec_states, rec_inds)
        ]

        # Loop over the ODE. The `step_fn` can be jitted for improving speed.
        steps = int(t_max / delta_t)  # Steps to integrate
        for step in range(steps):
            # Get externals at current timestep.
            externals_now = {}
            for key in externals.keys():
                externals_now[key] = externals[key][:, step]

            states = step_fn(
                states, params, externals_now, external_inds, delta_t=delta_t
            )
            recs = jnp.asarray(
                [
                    states[rec_state][rec_ind]
                    for rec_state, rec_ind in zip(rec_states, rec_inds)
                ]
            )
            recordings.append(recs)

        rec = jnp.stack(recordings, axis=0).T

    """
    # Initialize the external inputs and their indices.
    external_inds = module.external_inds.copy()

    def init_fn(
        params: List[Dict[str, jnp.ndarray]],
        all_states: Optional[Dict] = None,
        param_state: Optional[List[Dict]] = None,
        delta_t: float = 0.025,
    ) -> Tuple[Dict, Dict]:
        """Initializes the parameters and states of the neuron model.

        Args:
            params: List of trainable parameters.
            all_states: State if already initialized. Defaults to None.
            param_state: Parameters returned by `data_set`.. Defaults to None.
            delta_t: Step size. Defaults to 0.025.

        Returns:
            All states and parameters.
        """
        # Make the `trainable_params` of the same shape as the `param_state`, such that
        # they can be processed together by `get_all_parameters`.
        pstate = params_to_pstate(params, module.indices_set_by_trainables)
        if param_state is not None:
            pstate += param_state

        all_params = module.get_all_parameters(pstate)
        all_states = (
            module.get_all_states(pstate, all_params, delta_t)
            if all_states is None
            else all_states
        )
        return all_states, all_params

    def step_fn(
        all_states: Dict,
        all_params: Dict,
        externals: Dict,
        external_inds: Dict = external_inds,
        delta_t: float = 0.025,
    ) -> Dict:
        """Performs a single integration step with step size delta_t.

        Args:
            all_states: Current state of the neuron model.
            all_params: Current parameters of the neuron model.
            externals: External inputs.
            external_inds: External indices. Defaults to `module.external_inds`.
            delta_t: Time step. Defaults to 0.025.

        Returns:
            Updated states.
        """
        state = all_states
        state = module.step(
            state,
            delta_t,
            external_inds,
            externals,
            params=all_params,
            solver=solver,
            voltage_solver=voltage_solver,
        )
        return state

    return init_fn, step_fn


def add_stimuli(
    externals: Dict,
    external_inds: Dict,
    data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
) -> Tuple[Dict, Dict]:
    """Extends the external inputs with the stimuli.

    Args:
        externals: Current external inputs.
        external_inds: Current external indices.
        data_stimuli: Additional data stimuli. Defaults to None.

    Returns:
        Updated external inputs and indices.
    """
    # If stimulus is inserted, add it to the external inputs.
    if "i" in externals.keys() or data_stimuli is not None:
        if "i" in externals.keys():
            if data_stimuli is not None:
                externals["i"] = jnp.concatenate([externals["i"], data_stimuli[1]])
                external_inds["i"] = jnp.concatenate(
                    [external_inds["i"], data_stimuli[2].index.to_numpy()]
                )
        else:
            externals["i"] = data_stimuli[1]
            external_inds["i"] = data_stimuli[2].index.to_numpy()

    return externals, external_inds


def add_clamps(
    externals: Dict,
    external_inds: Dict,
    data_clamps: Optional[Tuple[str, jnp.ndarray, pd.DataFrame]] = None,
) -> Tuple[Dict, Dict]:
    """Adds clamps to the external inputs.

    Args:
        externals: Current external inputs.
        external_inds: Current external indices.
        data_clamps: Additional data clamps. Defaults to None.

    Returns:
        Updated external inputs and indices.
    """
    # If a clamp is inserted, add it to the external inputs.
    if data_clamps is not None:
        state_name, clamps, inds = data_clamps
        if state_name in externals.keys():
            externals[state_name] = jnp.concatenate([externals[state_name], clamps])
            external_inds[state_name] = jnp.concatenate(
                [external_inds[state_name], inds.index.to_numpy()]
            )
        else:
            externals[state_name] = clamps
            external_inds[state_name] = inds.index.to_numpy()

    return externals, external_inds


def integrate(
    module: Module,
    params: List[Dict[str, jnp.ndarray]] = [],
    *,
    param_state: Optional[List[Dict]] = None,
    data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
    data_clamps: Optional[Tuple[str, jnp.ndarray, pd.DataFrame]] = None,
    t_max: Optional[float] = None,
    delta_t: float = 0.025,
    solver: str = "bwd_euler",
    voltage_solver: str = "jaxley.dhs",
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
        data_clamps: Outputs of `.data_clamp()`, only needed if clamps change across
            function calls.
        t_max: Duration of the simulation in milliseconds. If `t_max` is greater than
            the length of the stimulus input, the stimulus will be padded at the end
            with zeros. If `t_max` is smaller, then the stimulus with be truncated.
        delta_t: Time step of the solver in milliseconds.
        solver: Which ODE solver to use. Either of ["fwd_euler", "bwd_euler",
            "crank_nicolson"].
        voltage_solver: Algorithm to solve quasi-tridiagonal linear system describing
            the voltage equations. The different options only take effect when
            `solver` is either `bwd_euler` or `crank_nicolson`. The options for
            `voltage_solver` are `jaxley.dhs` and `jax.sparse`.
            For unbranched cables, we also support `jaxley.stone` (which has good
            performance for unbranched cables on GPU).
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

    Example usage
    ^^^^^^^^^^^^^

    The most simple usage is the following:

    ::

        cell = jx.Cell()
        v = jx.integrate(cell, t_max=10.0)

    If ``t_max`` is not passed, then you must have inserted a stimulus, and ``t_max``
    will match the stimulus length.

    Customizing the solver
    ^^^^^^^^^^^^^^^^^^^^^^

    If you use ``jx.integrate(..., voltage_solver="jaxley.dhs")``, we automatically
    choose between a CPU and a GPU optimized version. If you manually want to run the
    CPU-optimized version on GPU, do:

    ::

        cell._init_morph_jaxley_dhs_solve(allowed_nodes_per_level=1)
        v = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")

    To run the GPU-opotimized version on CPU, do:

    ::

        cell._init_morph_jaxley_dhs_solve(allowed_nodes_per_level=16)
        v = jx.integrate(cell, voltage_solver="jaxley.dhs.gpu")

    """
    if voltage_solver == "jaxley.dhs":
        # Automatically infer the voltage solver.
        if module._solver_device in ["gpu", "tpu"]:
            voltage_solver = "jaxley.dhs.gpu"
        else:
            voltage_solver = "jaxley.dhs.cpu"

    assert module.initialized, "Module is not initialized, run `._initialize()`."
    module.to_jax()  # Creates `.jaxnodes` from `.nodes` and `.jaxedges` from `.edges`.

    # Initialize the external inputs and their indices.
    externals = module.externals.copy()
    external_inds = module.external_inds.copy()

    # If stimulus is inserted, add it to the external inputs.
    externals, external_inds = add_stimuli(externals, external_inds, data_stimuli)

    # If a clamp is inserted, add it to the external inputs.
    externals, external_inds = add_clamps(externals, external_inds, data_clamps)

    if not externals.keys():
        # No stimulus was inserted and no clamp was set.
        assert t_max is not None, (
            "If no stimulus or clamp are inserted you have to specify the simulation"
            "duration at `jx.integrate(..., t_max=)`."
        )

    for key in externals.keys():
        externals[key] = externals[key].T  # Shape `(time, num_stimuli)`.

    if module.recordings.empty:
        raise ValueError("No recordings are set. Please set them.")
    rec_inds = module.recordings.rec_index.to_numpy()
    rec_states = module.recordings.state.to_numpy()

    # Shorten or pad stimulus depending on `t_max`.
    if t_max is not None:
        t_max_steps = int(t_max // delta_t + 1)

        # Pad or truncate the stimulus.
        for key in externals.keys():
            if t_max_steps > externals[key].shape[0]:
                if key == "i":
                    pad = jnp.zeros(
                        (t_max_steps - externals["i"].shape[0], externals["i"].shape[1])
                    )
                    externals["i"] = jnp.concatenate((externals["i"], pad))
                else:
                    raise NotImplementedError(
                        "clamp must be at least as long as simulation."
                    )
            else:
                externals[key] = externals[key][:t_max_steps, :]

    init_fn, step_fn = build_init_and_step_fn(
        module, voltage_solver=voltage_solver, solver=solver
    )
    all_states, all_params = init_fn(params, all_states, param_state, delta_t)

    def _body_fun(state, externals):
        state = step_fn(state, all_params, externals, external_inds, delta_t)
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
    if externals:
        example_key = list(externals.keys())[0]
        nsteps_to_return = len(externals[example_key])
    else:
        nsteps_to_return = t_max_steps

    if checkpoint_lengths is None:
        checkpoint_lengths = [nsteps_to_return]
        length = nsteps_to_return
    else:
        length = prod(checkpoint_lengths)
        size_difference = length - nsteps_to_return
        assert (
            nsteps_to_return <= length
        ), "The desired simulation duration is longer than `prod(nested_length)`."
        if externals:
            dummy_external = jnp.zeros(
                (size_difference, externals[example_key].shape[1])
            )
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
