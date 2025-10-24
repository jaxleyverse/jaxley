# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.tree import leaves
from jax.tree_util import tree_map

from jaxley.utils.cell_utils import params_to_pstate
from jaxley.integrate import build_init_and_step_fn


def _remove_currents_from_states(states: dict[str, Array], current_keys: list[str]):
    """Remove the currents through channels and synapses from the states.
    
    Args:
        states: States (including currents) of the system.
        current_keys: The names of all channel currents.
    """
    for key in current_keys:
        del states[key]
    return states


def build_utils_for_dynamic_states(module) -> Tuple[Callable, Callable, Callable]:
    r"""Returns three utility functions which can be used to convert states to a vector.

    These utility functions are meant to be used together with 
    ``jx.integrate.build_init_and_step_fn``. The ``init_fn`` returned by
    ``build_init_and_step_fn`` returns an ``all_states``, which is a dictionary
    of all states, including the voltages at branchpoints and the channel and synapse
    currents. The utility functions returned by ``build_utils_for_dynamic_states()``
    modify the ``all_states`` as follows:

    - They remove all channel currents, syanpse currents, and branchpoint voltages
      (which can be computed from compartment voltages). As such, only "true" dynamic
      states remain.
    - They return the states as a flat array. This allows easier interoperability
      with frameworks such as ``dynamax``.

    Args:
        module: A ``Module`` object that e.g. a cell.
        params: Trainable parameters of the neuron model.
        param_state: Parameters returned by ``data_set``. Defaults to None.
        voltage_solver: Voltage solver used in step.
        solver: ODE solver.
        delta_t: Time step.

    Returns:

        * **states_to_pytree** -  Function to convert the state vector back to a pytree.
        * **states_to_full_pytree** -  Function to convert the state vector back to a
          pytree and restore observables.
        * **full_pytree_to_states** -  Function to convert the full state pytree to a
          vector and filter observables.

    Example usage
    ^^^^^^^^^^^^^

    Example 1: Use `full_pytree_to_states` to build a vector of dynamics states. Use
    `states_to_full_pytree` to convert the vector back to the `states` dictionary.

    ::

        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn
        from jaxley.utils.dynamics import build_step_dynamics_fn

        cell = jx.Cell()
        params = cell.get_parameters()

        build_step_dynamics_fn()
        build_init_and_step_fn()

        all_states, all_params = init_fn(params)

        dynamic_states = full_pytree_to_states(all_states)
        recovered_all_states = states_to_pytree(dynamic_states)

    Example 2: Use `states_to_pytree` to idetify the names of states in the vector
    valued dynamic states.

    ::

        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn
        from jaxley.utils.dynamics import build_step_dynamics_fn

        cell = jx.Cell()
        params = cell.get_parameters()

        build_step_dynamics_fn()
        build_init_and_step_fn()

        all_states, all_params = init_fn(params)

        dynamic_states = full_pytree_to_states(full_states_pytree)

        # Recover the names of states in the `dynamic_states`.
        states_pytree = states_to_pytree(dynamic_states)

    Example 3: Build a `step_dynamics` function and use it to compute the Jacobian
    of a single step.

    ::

        from jax import jacfwd
        import jaxley as jx
        from jaxley.utils.dynamics import build_step_dynamics_fn
        from jaxley.channels import Leak

        comp = jx.Compartment()
        branch = jx.Branch(comp, 8)
        cell = jx.Cell(branch, parents=[-1, 0, 0])
        cell.insert(Leak())
        cell.to_jax()

        states_vec, step_dynamics_fn, _, _, _ = build_step_dynamics_fn(
            cell, solver="bwd_euler", delta_t=0.025
        )
        jacobian = jacfwd(step_dynamics_fn)(states_vec)

    Example 4: Build a loss function based on input and parameters.

    ::

        import jax.numpy as jnp

        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn
        from jaxley.utils.dynamics import build_step_dynamics_fn
        from jaxley.channels import Leak

        cell = jx.Cell()
        cell.insert(Leak())
        t_max = 3.0
        delta_t = 0.025

        cell.record()
        cell.stimulate(jx.step_current(0, 1, 2, delta_t, t_max))

        rec_inds = cell.recordings.rec_index.to_numpy()
        rec_states = cell.recordings.state.to_numpy()
        externals = cell.externals.copy()
        external_inds = cell.external_inds.copy()

        cell.make_trainable("radius")
        params = cell.get_parameters()

        init_fn, step_fn = build_init_and_step_fn(cell)
        full_pytree_to_states, states_to_full_pytree, states_to_pytree = build_step_dynamics_fn(cell)

        def init_dynamics(params, param_state):
            all_states, all_params = init_fn(params, None, param_state)
            recordings = [
                all_states[rec_state][rec_ind][None]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ]
            dynamic_states = full_pytree_to_states(all_states)
            return dynamic_states, all_params, recordings

        def step_dynamics(dynamic_states, all_params, externals, external_inds):
            all_states = states_to_full_pytree(dynamic_states, all_params, 0.025)
            all_states = step_fn(
                all_states, all_params, externals, external_inds, delta_t=delta_t
            )
            recs = jnp.asarray(
                [
                    all_states[rec_state][rec_ind]
                    for rec_state, rec_ind in zip(rec_states, rec_inds)
                ]
            )
            dynamic_states = full_pytree_to_states(all_states)
            return dynamic_states, recs

        def loss_fn(params, param_state_value):
            param_state = cell.data_set("Leak_gLeak", param_state_value, None)
            cell.to_jax()
            dynamic_states, all_params, recordings = init_dynamics(params, param_state)
            steps = int(t_max / delta_t)
            for step in range(steps):
                externals_now = {}
                for key in externals.keys():
                    externals_now[key] = externals[key][:, step]
                dynamic_states, recs = step_dynamics(dynamic_states, all_params, externals_now, external_inds)
                recordings.append(recs)
            return jnp.mean(jnp.stack(recordings, axis=0).T)

        loss = loss_fn(params, 1e-4)
    """

    # # Get the names of the currents that are artifially added.§
    all_states = module.get_all_states([])
    # base_keys = list(all_states.keys())

    # init_fn, step_fn = build_init_and_step_fn(
    #     module, voltage_solver=voltage_solver, solver=solver)
    # all_states = init_fn()
    # added_keys = [k for k in all_states.keys() if k not in base_keys]
    added_keys = ["i_Leak"]

    # Remove branchpoints if needed
    original_length = len(leaves(all_states)[0])

    if hasattr(module, "_branchpoints") and len(module._branchpoints.index) > 0:
        filter_indices = jnp.array(module._branchpoints.index.to_numpy(), dtype=int)
        all_indices = jnp.arange(original_length)
        keep_indices = jnp.setdiff1d(all_indices, filter_indices, assume_unique=True)
        branch_filter_applied = True
    else:
        keep_indices = jnp.arange(original_length)
        branch_filter_applied = False

    all_states = tree_map(lambda x: jnp.take(x, keep_indices, axis=0), all_states)
    filtered_length = len(leaves(all_states)[0])

    # Remove NaNs (appear if some states are not defined on all compartments)
    nan_mask_tree = tree_map(jnp.isnan, all_states)
    nan_indices_tree = tree_map(lambda m: jnp.where(~m)[0], nan_mask_tree)

    def take_by_idx(x, idx):
        if getattr(x, "ndim", None) is None or x.ndim == 0:
            return x
        return jnp.take(x, idx, axis=0)

    all_states_no_nans = tree_map(take_by_idx, all_states, nan_indices_tree)

    # Flatten to a vector
    states_vec, states_to_pytree = ravel_pytree(all_states_no_nans)

    # Now we can create functions that convert between the full state pytree
    # and the filtered state vector
    # ----------------------------------------------------------

    # Ravel from pytree (post-step) to vector
    def full_pytree_to_states(states):
        filtered_states = _remove_currents_from_states(states, added_keys)
        filtered_states = tree_map(
            lambda x: jnp.take(x, keep_indices, axis=0), filtered_states
        )
        filtered_states = tree_map(take_by_idx, filtered_states, nan_indices_tree)
        filtered_states_vec, _ = ravel_pytree(filtered_states)
        return filtered_states_vec

    # Unravel from vector to full restored state pytree
    def states_to_full_pytree(states_vec, all_params, delta_t):
        all_states_no_nans = states_to_pytree(states_vec)

        def restore_leaf(filtered_array, nan_indices_leaf):
            restored_array = jnp.full(filtered_length, jnp.nan)
            restored_array = restored_array.at[nan_indices_leaf].set(filtered_array)
            return restored_array

        all_states_with_nans = tree_map(
            restore_leaf, all_states_no_nans, nan_indices_tree
        )
        if branch_filter_applied:

            def restore_branch_leaf(leaf):
                restored_array = jnp.full(original_length, -1.0)
                restored_array = restored_array.at[keep_indices].set(leaf)
                return restored_array

            restored_states = tree_map(restore_branch_leaf, all_states_with_nans)
        else:
            restored_states = all_states_with_nans

        restored_states = module.append_channel_currents_to_states(
            restored_states, all_params, delta_t=delta_t
        )
        return restored_states

    return full_pytree_to_states, states_to_full_pytree, states_to_pytree
