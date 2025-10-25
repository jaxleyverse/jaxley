# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.tree import leaves
from jax.tree_util import tree_map


def _remove_currents_from_states(states: dict[str, Array], current_keys: list[str]):
    """Remove the currents through channels and synapses from the states.
    
    Args:
        states: States (including currents) of the system.
        current_keys: The names of all channel currents.
    """
    for key in current_keys:
        del states[key]
    return states


def build_dynamic_state_utils(module) -> Tuple[Callable, Callable, Callable, Callable]:
    r"""Return functions which extract the dynamic (ODE) states of a ``jx.Module``.

    These utility functions are meant to be used together with 
    ``jx.integrate.build_init_and_step_fn``. The ``init_fn`` returned by
    ``build_init_and_step_fn`` returns an ``all_states``, which is a dictionary
    of all states, including the voltages at branchpoints and the channel and synapse
    currents. The utility functions returned by ``build_utils_for_dynamic_states()``
    modify the ``all_states`` as follows:

    - They remove all channel currents, syanpse currents, and branchpoint voltages
      (which can be computed from compartment voltages). As such, only "true" dynamic
      states remain. This is handled by the returned functions ``remove_spurious`` and
      ``add_spurious``.
    - They return the states as a flat array. This allows easier interoperability
      with frameworks such as ``dynamax``. This is handled by the returned functions
      ``flatten`` and ``unflatten``.

    Args:
        module: A ``Module`` object that e.g. a cell.
        params: Trainable parameters of the neuron model.
        param_state: Parameters returned by ``data_set``. Defaults to None.
        voltage_solver: Voltage solver used in step.
        solver: ODE solver.
        delta_t: Time step.

    Returns:

        * **remove_spurious** - Callable which removes the membrane currents, synaptic
          currents, and branchpoint voltages from the states dict. The returned
          states only include true "dynamic" states.

          * Args: ``all_states`` (Dict[str, Array]). All states of the system which can
            be recorded.

          * Returns: Dynamic states of the system (Dict[str, Array]).

        * **add_spurious** - Callable which adds membrane currents, synaptic currents,
          and branchpoint voltages to the states dictionary.

          * Args: ``dynamic_states`` (Dict[str, Array]),
            ``all_params`` (Dict[str, Array]), ``delta_t`` (float).

          * Returns: All states of the system which can be recorded (Dict[str, Array]).

        * **flatten** - Callable which flattens states as a pytree into a jnp.Array.

          * Args: ``dynamic_states`` (Dict[str, Array]). Contains all dynamic states.

          * Returns: Dynamic states of the system as a flattened Array (Array).

        * **unflatten** - Callable which converts the state vector back to a pytree.

          * Args: ``flat_dynamic_states`` (Array). A flattened array including the
            dynamic states of the system.

          * Returns: Dynamic states as a dict of Arrays (Dict[str, Array]).

    Example usage
    ^^^^^^^^^^^^^

    Example 1: Use the functions returned by `build_dynamic_state_utils` to build a
    vector of dynamics states. Then convert the vector back to the
    `all_states` dictionary.

    ::

        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn
        from jaxley.utils.dynamics import build_dynamic_state_utils

        cell = jx.Cell()
        params = cell.get_parameters()

        init_fn, step_fn = build_init_and_step_fn(cell)
        remove_spurious, add_spurious, flatten, unflatten = build_dynamic_state_utils(cell)

        all_states, all_params = init_fn(params)

        dynamic_states = flatten(remove_spurious(all_states))
        recovered_all_states = add_spurious(unflatten(dynamic_states), all_params, delta_t=0.025)

    Example 2: Build a `step_dynamics` function and use it to compute the Jacobian
    of a single step.

    ::

        from jax import jacfwd

        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn
        from jaxley.utils.dynamics import build_dynamic_state_utils
        from jaxley.channels import Leak

        comp = jx.Compartment()
        branch = jx.Branch(comp, 2)
        cell = jx.Cell(branch, parents=[-1, 0, 0])
        cell.insert(Leak())
        params = cell.get_parameters()

        externals = cell.externals.copy()
        external_inds = cell.external_inds.copy()

        init_fn, step_fn = build_init_and_step_fn(cell)
        remove_spurious, add_spurious, flatten, unflatten = build_dynamic_state_utils(cell)

        all_states, all_params = init_fn(params)
        dynamic_states = flatten(remove_spurious(all_states))

        def step_dynamics(dynamic_states, all_params, externals, external_inds, delta_t):
            all_states = add_spurious(unflatten(dynamic_states), all_params, delta_t)
            all_states = step_fn(
                all_states, all_params, externals, external_inds, delta_t=delta_t
            )
            dynamic_states = flatten(remove_spurious(all_states))
            return dynamic_states

        jacobian = jacfwd(step_dynamics)(dynamic_states, all_params, externals, external_inds, delta_t=0.025)

    Example 3: Build a loss function based on input and parameters.

    ::

        import jax.numpy as jnp

        import jaxley as jx
        from jaxley.integrate import build_init_and_step_fn
        from jaxley.utils.dynamics import build_dynamic_state_utils
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
        remove_spurious, add_spurious, flatten, unflatten = build_dynamic_state_utils(cell)

        def init_dynamics(params, param_state):
            all_states, all_params = init_fn(params, None, param_state)
            recordings = [
                all_states[rec_state][rec_ind][None]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ]
            dynamic_states = flatten(remove_spurious(all_states))
            return dynamic_states, all_params, recordings

        def step_dynamics(dynamic_states, all_params, externals, external_inds):
            all_states = add_spurious(unflatten(dynamic_states), all_params, 0.025)
            all_states = step_fn(
                all_states, all_params, externals, external_inds, delta_t=delta_t
            )
            recs = jnp.asarray(
                [
                    all_states[rec_state][rec_ind]
                    for rec_state, rec_ind in zip(rec_states, rec_inds)
                ]
            )
            dynamic_states = flatten(remove_spurious(all_states))
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

    all_states = module.get_all_states([])
    added_keys = module.membrane_current_names + module.synapse_current_names

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
    _, unflatten = ravel_pytree(all_states_no_nans)

    def flatten(all_states_no_nans: dict[str, Array]) -> Array:
        """Convert the state vector back to a pytree.

        Args: Dynamic states as dict of jnp.Arrays. Contains all dynamic states.

        Returns: Flattened dynamic states as an jnp.Array.
        """
        states_vec, _ = ravel_pytree(all_states_no_nans)
        return states_vec

    # Now we can create functions that convert between the full state pytree
    # and the filtered state vector
    # ----------------------------------------------------------

    # Ravel from pytree (post-step) to vector
    def remove_spurious(states: (dict[str, Array])) -> dict[str, Array]:
        r"""Remove the membrane currents, synaptic currents, and branchpoint voltages.
         
        Thus, the returned states only include true "dynamic" states.

        Args:
            all_states:. All states of the system which can
            be recorded.

        Returns: All dynamic states of the system.
        """
        filtered_states = _remove_currents_from_states(states, added_keys)
        filtered_states = tree_map(
            lambda x: jnp.take(x, keep_indices, axis=0), filtered_states
        )
        filtered_states = tree_map(take_by_idx, filtered_states, nan_indices_tree)
        return filtered_states

    # Unravel from vector to full restored state pytree
    def add_spurious(
            all_states_no_nans: dict[str, Array],
            all_params: dict[str, Array],
            delta_t: float
        ) -> dict[str, Array]:
        """Add membrane currents, synaptic currents, and branchpoint voltages to states.

        Args:
            dynamic_states:
            all_params:
            delta_t: 

        Returns:
            ``all_states`` which can be passed to the ``step_fn`` (returned by
            ``jx.integrate.build_init_and_step_fn``).
        """
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

    return remove_spurious, add_spurious, flatten, unflatten
