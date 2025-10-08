from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.tree import leaves
from jax.tree_util import tree_map

from jaxley.modules import Module
from jaxley.utils.cell_utils import params_to_pstate


def add_currents_to_states(module, states, delta_t, all_params):
    """Add the currents through channels and synapses to the states."""

    # Add to the states the initial current through every channel.
    states, _ = module.base._channel_currents(
        states, delta_t, module.channels + module.pumps, module.nodes, all_params
    )

    # Add to the states the initial current through every synapse.
    states, _ = module.base._synapse_currents(
        states,
        module.synapses,
        all_params,
        delta_t,
        module.edges,
    )
    return states


def remove_currents_from_states(states, current_keys):
    """Remove the currents through channels and synapses from the states."""
    for key in current_keys:
        del states[key]
    return states


def get_all_states_no_currents(module, pstate):
    """Get all states from the module, excluding currents"""
    states = module._get_states_from_nodes_and_edges()
    # Override with the initial states set by `.make_trainable()`.

    for parameter in pstate:
        key = parameter["key"]
        inds = parameter["indices"]
        set_param = parameter["val"]
        if key in states:  # Only initial states, not parameters.
            # `inds` is of shape `(num_params, num_comps_per_param)`.
            # `set_param` is of shape `(num_params,)`
            # We need to unsqueeze `set_param` to make it `(num_params, 1)` for the
            # `.set()` to work. This is done with `[:, None]`.
            states[key] = states[key].at[inds].set(set_param[:, None])
    return states


def build_step_dynamics_fn(
    module: Module,
    voltage_solver: str = "jaxley.dhs",
    solver: str = "bwd_euler",
    delta_t: float = 0.025,
    params: Dict = {},
    param_state: Dict = None,
) -> Tuple[Callable, Callable]:
    """Return ``init_fn`` and ``step_fn`` which initialize modules and run update steps.

    This method can be used to gain additional control over the simulation workflow.
    It exposes the ``step`` function, which can be used to perform step-by-step updates
    of the differential equations.

    Crucially this function returns a vectorized state representation containting
    only true dynamical states (e.g. voltages and gating variables), and no observables
    (e.g. currents).

    Args:
        module: A `Module` object that e.g. a cell.
        voltage_solver: Voltage solver used in step. Defaults to "jaxley.stone".
        solver: ODE solver. Defaults to "bwd_euler".

    Returns:
        init_fn, step_fn: Functions that initialize the state and parameters, and
            perform a single integration step, respectively.
    """

    # Initialize the external inputs and their indices.
    external_inds = module.external_inds.copy()

    
    # Get the full parameter state including observables
    # ----------------------------------------------------------
    pstate = params_to_pstate(params, module.indices_set_by_trainables)
    if param_state is not None:
        pstate += param_state

    all_params = module.get_all_parameters(pstate)
    all_states = get_all_states_no_currents(module, pstate)

    base_keys = list(all_states.keys())
    all_states = add_currents_to_states(module, all_states, delta_t, all_params)
    added_keys = [k for k in all_states.keys() if k not in base_keys]

    # Remove observables from states
    # ----------------------------------------------------------

    # first remove currents
    all_states = remove_currents_from_states(all_states, added_keys)

    # remove branchpoints if needed
    original_length = len(leaves(all_states)[0])

    if hasattr(module, "_branchpoints") and len(module._branchpoints.index) > 0:
        filter_indices = jnp.array(module._branchpoints.index.to_numpy(), dtype=int)
        all_indices = jnp.arange(original_length)
        keep_indices = jnp.setdiff1d(
            all_indices, filter_indices, assume_unique=True
        )
        branch_filter_applied = True
    else:
        keep_indices = jnp.arange(original_length)
        branch_filter_applied = False

    all_states = tree_map(lambda x: jnp.take(x, keep_indices, axis=0), all_states)
    filtered_length = len(leaves(all_states)[0])

    # remove NaNs (appear if some states are not defined on all compartments)
    nan_mask_tree = tree_map(jnp.isnan, all_states)
    nan_indices_tree = tree_map(lambda m: jnp.where(~m)[0], nan_mask_tree)

    def take_by_idx(x, idx):
        if getattr(x, "ndim", None) is None or x.ndim == 0:
            return x
        return jnp.take(x, idx, axis=0)

    all_states_no_nans = tree_map(take_by_idx, all_states, nan_indices_tree)

    # flatten to a vector
    states_vec, unravel_fn = ravel_pytree(all_states_no_nans)

    # Now we can create functions that convert between the full state pytree
    # and the filtered state vector
    # ----------------------------------------------------------

    # ravel from pytree (post-step) to vector
    def ravel_filter_fn(states):
        filtered_states = remove_currents_from_states(states, added_keys)
        filtered_states = tree_map(
            lambda x: jnp.take(x, keep_indices, axis=0), filtered_states
        )
        filtered_states = tree_map(take_by_idx, filtered_states, nan_indices_tree)
        filtered_states_vec, _ = ravel_pytree(filtered_states)
        return filtered_states_vec

    # unravel from vector to full restored state pytree
    def unravel_restore_fn(states_vec):
        all_states_no_nans = unravel_fn(states_vec)

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

        restored_states = add_currents_to_states(
            module, restored_states, delta_t, all_params
        )
        return restored_states

    
    def step_dynamics_fn(
        states_vec: Array,
        params: Dict,
        externals: Dict,
        external_inds: Dict = external_inds,
        delta_t: float = 0.025,
    ) -> Array:
        """Performs a single integration step with step size delta_t.

        Args:
            states_vec: Current state of the neuron model vectorised.
            all_params: Current parameters of the neuron model.
            externals: External inputs.
            external_inds: External indices. Defaults to `module.external_inds`.
            delta_t: Time step. Defaults to 0.025.
            unravel_restore_fn: Function to convert the state vector back to a pytree and restore observables.
            ravel_filter_fn: Function to convert the full state pytree to a vector and filter observables.
        Returns:
            Updated states vectorised.
        """

        state = unravel_restore_fn(states_vec)
        
        # to do: add params to all_params

        state = module.step(
            state,
            delta_t,
            external_inds,
            externals,
            params=all_params,
            solver=solver,
            voltage_solver=voltage_solver,
        )
        states_vec = ravel_filter_fn(state)
        return states_vec

    return states_vec, step_dynamics_fn
