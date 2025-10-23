# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.tree import leaves
from jax.tree_util import tree_map

from jaxley.modules import Module
from jaxley.utils.cell_utils import params_to_pstate


def get_all_params_param_state(module, params, param_state):
    """Convenience functions to combine params and param_state to get all_params."""
    pstate = params_to_pstate(params, module.indices_set_by_trainables)
    if param_state is not None:
        pstate += param_state
    all_params = module.get_all_parameters(pstate)
    return all_params, pstate


def remove_currents_from_states(states, current_keys):
    """Remove the currents through channels and synapses from the states."""
    for key in current_keys:
        del states[key]
    return states


def build_step_dynamics_fn(
    module: Module,
    params: list[dict[str, Array]] | None = None,
    param_state: list[dict] | None = None,
    voltage_solver: str = "jaxley.dhs",
    solver: str = "bwd_euler",
    delta_t: float = 0.025,
) -> Tuple[Array, Callable, Callable, Callable, Callable]:
    """Initialises and returns ``step_dynamics`` which takes a vector-valued state 
    and returns the state vector at the next time step.

    This can be used used to perform step-by-step updates where states are vector
    representations of the full state pytree of the module, and only contain
    true dynamical states (e.g. voltages and gating variables), and no observables
    (e.g. currents). This is useful for e.g. computing Jacobians and Kalman filters.

    Args:
        module: A `Module` object that e.g. a cell.
        params: Trainable parameters of the neuron model.
        param_state: Parameters returned by `data_set`.. Defaults to None.
        voltage_solver: Voltage solver used in step. Defaults to "jaxley.stone".
        solver: ODE solver. Defaults to "bwd_euler".
        delta_t: Time step. Defaults to 0.025.

    Returns:
        states_vec: Initial state of the neuron model vectorised.
        step_dynamics_fn: Function that performs a single integration step with step size delta_t.
        states_to_pytree: Function to convert the state vector back to a pytree.
        states_to_full_pytree: Function to convert the state vector back to a pytree and restore observables.
        full_pytree_to_states: Function to convert the full state pytree to a vector and filter observables.

    Example usage
    ^^^^^^^^^^^^^
    # We can build a step dynamics function for a simple cell as follows. 
    ::
        # Build a simple cell
        comp = jx.Compartment()
        branch = jx.Branch(comp, 8)
        cell = jx.Cell(branch, parents=[-1, 0, 0])
        cell.insert(Leak())

        # states_vec, step_dynamics_fn, states_to_pytree, states_to_full_pytree, full_pytree_to_states = build_step_dynamics_fn(
                cell, solver="bwd_euler", delta_t=0.025
            )

        # we can now step through the dynamics
        states_vec_next = step_dynamics_fn(states_vec)

        # we can use this function to conveniently calculate jacobians
        from jax import jacfwd
        jacobian = jacfwd(step_dynamics_fn)(states_vec)

        # we can convert the state vector back to a pytree
        states_pytree = states_to_pytree(states_vec_next)

        # or to a pytree that includes observables like currents and branchpoints
        full_states_pytree = states_to_full_pytree(states_vec_next)

        # we can also convert a full pytree back to a state vector
        states_vec_restored = full_pytree_to_states(full_states_pytree)

        
    
    We can also add inputs, jit, and compute gradients w.r.t. parameters

    ::

        # Build a simple cell
        comp = jx.Compartment()
        branch = jx.Branch(comp, 8)
        cell = jx.Cell(branch, parents=[-1, 0, 0])
        cell.insert(HH())
        cell.insert(Leak())

        # make some parameters trainable
        cell.make_trainable("Leak_gLeak")
        cell.make_trainable("v")
        params = cell.get_parameters()
        # Define parameter transform and apply it to the parameters.
        transform = jx.ParamTransform(
            [
                {"Leak_gLeak": jt.SigmoidTransform(0.00001, 0.0002)},
                {"v": jt.SigmoidTransform(-100, -30)},
            ]
        )
        opt_params = transform.inverse(params)
        params = transform.forward(opt_params)
        cell.to_jax()

        # add some inputs
        externals = cell.externals.copy()
        external_inds = cell.external_inds.copy()
        current = jx.step_current(
            i_delay=1.0, i_dur=2.0, i_amp=0.08, delta_t=0.025, t_max=0.075
        )
        data_stimuli = None
        data_stimuli = cell.branch(0).comp(0).data_stimulate(current, data_stimuli)
        externals, external_inds = add_stimuli(externals, external_inds, data_stimuli)

        def get_externals_now(externals, step):
            externals_now = {}
            for key in externals.keys():
                externals_now[key] = externals[key][:, step]
            return externals_now

        target_voltage = -60.0
        state_idx = -30

        # Define the optimizer
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(opt_params)

        def loss(opt_params):
            params = transform.forward(opt_params)

            # initialise and build the step function
            states_vec, step_dynamics_fn, _, _, _ = build_step_dynamics_fn(
                cell, solver="bwd_euler", delta_t=0.025, params=params
            )

            # JIT the step function for speed
            @jit
            def step_fn_vec_to_vec(states_vec, externals_now, params=None):
                states_vec = step_dynamics_fn(
                    states_vec,
                    params,
                    externals=externals_now,
                    external_inds=external_inds,
                    delta_t=0.025,
                )
                return states_vec

            states_vecs = [states_vec]

            # Simulate the model
            for step in range(3):
                # Get inputs at this time step
                externals_now = get_externals_now(externals, step)
                # Step the ODE
                states_vec = step_fn_vec_to_vec(states_vec, externals_now, params)
                # Store the state
                states_vecs.append(states_vec)
            # Compute the loss at the last time step
            loss = jnp.mean((states_vecs[-1][state_idx] - target_voltage) ** 2)
            return loss

        # Compute the gradient of the loss with respect to the parameters
        grad_loss = value_and_grad(loss, argnums=0)
        value, gradient = grad_loss(opt_params)

        updates, opt_state = optimizer.update(gradient, opt_state)
    """

    # Initialize the external inputs and their indices.
    external_inds = module.external_inds.copy()

    # Get the full parameter state including observables
    # ----------------------------------------------------------
    if params is None:
        params = {}

    all_params, pstate = get_all_params_param_state(module, params, param_state)

    all_states = module.get_all_states(pstate)

    base_keys = list(all_states.keys())

    all_states = module.append_channel_currents_to_states(
        all_states, all_params, delta_t=delta_t
    )

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
        keep_indices = jnp.setdiff1d(all_indices, filter_indices, assume_unique=True)
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
    states_vec, states_to_pytree = ravel_pytree(all_states_no_nans)

    # Now we can create functions that convert between the full state pytree
    # and the filtered state vector
    # ----------------------------------------------------------

    # ravel from pytree (post-step) to vector
    def full_pytree_to_states(states):
        filtered_states = remove_currents_from_states(states, added_keys)
        filtered_states = tree_map(
            lambda x: jnp.take(x, keep_indices, axis=0), filtered_states
        )
        filtered_states = tree_map(take_by_idx, filtered_states, nan_indices_tree)
        filtered_states_vec, _ = ravel_pytree(filtered_states)
        return filtered_states_vec

    # unravel from vector to full restored state pytree
    def states_to_full_pytree(states_vec):
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

    def step_dynamics(
        states_vec: Array,
        params: list[dict[str, Array]] | None = None,
        param_state: dict[str, Array] | None = None,
        externals: dict[str, Array] | None = None,
        external_inds: dict[str, Array] = external_inds,
        delta_t: float = 0.025,
    ) -> Array:
        """Performs a single integration step with step size delta_t.

        Args:
            states_vec: Current state of the neuron model vectorised.
            params: trainable params of the neuron model.
            param_state: Parameters returned by `data_set`.. Defaults to None.
            externals: External inputs.
            external_inds: External indices. Defaults to `module.external_inds`.
            delta_t: Time step. Defaults to 0.025.
        Returns:
            states_vec at next time step.
        """
        if externals is None:  # saver than {} default argument
            externals = {}

        if params is None:
            params = {}

        # restore full state pytree from vector
        state = states_to_full_pytree(states_vec)

        # add params to all_params
        all_params, _ = get_all_params_param_state(module, params, param_state)

        # step the dynamics
        state = module.step(
            state,
            delta_t,
            external_inds,
            externals,
            params=all_params,
            solver=solver,
            voltage_solver=voltage_solver,
        )

        # convert back to vector and filter out observables
        states_vec = full_pytree_to_states(state)
        return states_vec

    return (
        states_vec,
        step_dynamics,
        states_to_pytree,
        states_to_full_pytree,
        full_pytree_to_states,
    )
