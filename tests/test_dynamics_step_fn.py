# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

import jaxley as jx
import jaxley.optimize.transforms as jt
from jaxley.channels import Leak
from jaxley.channels.hh import HH
from jaxley.integrate import add_stimuli
from jaxley.utils.dynamics import build_step_dynamics_fn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import jit, value_and_grad


@pytest.fixture()
def hh_cell():
    """Fixture to build a simple HH+Leak cell"""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 3)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(HH())
    cell.insert(Leak())
    cell.to_jax()
    return cell


def test_cycle_consistency(hh_cell):
    """Ensure that ravel/unravel of state vectors is consistent"""
    cell = hh_cell
    params = []

    states_vec, _, _, states_to_full_pytree, full_pytree_to_states = (
        build_step_dynamics_fn(cell, solver="bwd_euler", delta_t=0.025, params=params)
    )

    restored = states_to_full_pytree(states_vec)
    reraveled = full_pytree_to_states(restored)

    assert np.allclose(reraveled, states_vec)


def test_jit(hh_cell):
    """Verify that the JIT-compiled step function runs without errors"""
    cell = hh_cell
    params = []
    states_vec, step_dynamics, _, _, _ = build_step_dynamics_fn(
        cell, solver="bwd_euler", delta_t=0.025, params=params
    )

    @jit
    def step_once(states_vec):
        return step_dynamics(
            states_vec, params, externals={}, external_inds={}, delta_t=0.025
        )

    result = step_once(states_vec)
    assert result.shape == states_vec.shape


def test_jit_and_grad(hh_cell):
    """Test jitting the dynamics step function and gradients"""
    cell = hh_cell

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

    target_voltage = -50.0
    state_idx = -30

    # Define the optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(opt_params)

    def loss(opt_params):
        params = transform.forward(opt_params)

        # initialise and build the step function
        states_vec, step_dynamics, _, _, _ = build_step_dynamics_fn(
            cell, solver="bwd_euler", delta_t=0.025, params=params
        )

        # JIT the step function for speed
        @jit
        def step_fn_vec_to_vec(states_vec, externals_now, params=None):
            states_vec = step_dynamics(
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

    assert np.all(abs(gradient[0]["Leak_gLeak"]) > 0)
    assert np.all(abs(gradient[1]["v"]) > 0)


def test_jit_and_grad_pstate(hh_cell):
    """Test jitting the dynamics step function and gradients"""
    cell = hh_cell

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

    target_voltage = -50.0
    state_idx = -30

    # Define the optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(opt_params)

    def loss(opt_params):
        params = transform.forward(opt_params)

        # initialise and build the step function
        states_vec, step_dynamics, _, _, _ = build_step_dynamics_fn(
            cell, solver="bwd_euler", delta_t=0.025, params=params
        )

        # JIT the step function for speed
        @jit
        def step_fn_vec_to_vec(states_vec, externals_now, params=None):
            states_vec = step_dynamics(
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

    assert np.all(abs(gradient[0]["Leak_gLeak"]) > 0)
    assert np.all(abs(gradient[1]["v"]) > 0)


def test_jit_and_grad_pstate(hh_cell):
    """Test jitting the dynamics step function and gradients wrt pstate"""
    cell = hh_cell

    params = None  # no trainable params as we are going to use pstate
    pstate_values = jnp.array([-60, 0.0001]) # initial v and Leak_gLeak

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

    target_voltage = -50.0
    state_idx = -30

    # Define the optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(pstate_values)

    def loss(pstate_values):

        # initialise and build the step function
        pstate = cell.data_set("Leak_gLeak", pstate_values[1], None)
        pstate = cell.data_set("v", pstate_values[0], pstate)
        (
            states_vec,
            step_dynamics_fn,
            _,
            _,
            _,
        ) = build_step_dynamics_fn(
            cell, solver="bwd_euler", delta_t=0.025, params=params, param_state=pstate
        )

        # JIT the step function for speed
        @jit
        def step_fn_vec_to_vec(
            states_vec, externals_now, params=None, pstate=pstate_values
        ):
            pstate = cell.data_set(
                "Leak_gLeak", pstate_values[1], None
            )  # only update leak_gLeak
            states_vec = step_dynamics_fn(
                states_vec,
                params,
                param_state=pstate,
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
            states_vec = step_fn_vec_to_vec(
                states_vec, externals_now, params, pstate_values
            )
            # Store the state
            states_vecs.append(states_vec)
        # Compute the loss at the last time step
        loss = jnp.mean((states_vecs[-1][state_idx] - target_voltage) ** 2)
        return loss

    # Compute the gradient of the loss with respect to the parameters
    grad_loss = value_and_grad(loss, argnums=0)
    value, gradient = grad_loss(pstate_values)

    updates, opt_state = optimizer.update(gradient, opt_state)
    assert np.all(abs(gradient[0]) > 0)
    assert np.all(abs(gradient[1]) > 0)


@pytest.mark.parametrize(
    "branchpoint", [True, False], ids=["branchpoint", "no_branchpoint"]
)
def test_build_step_dynamics_fn_branchpoints(branchpoint):
    """Check state vector length changes with/without branchpoints"""
    comp = jx.Compartment()
    branch = jx.Branch(comp, 8)
    parents = [-1, 0, 0] if branchpoint else [-1]
    cell = jx.Cell(branch, parents=parents)
    cell.insert(HH())
    cell.insert(Leak())
    cell.to_jax()
    params = []

    states_vec, _, states_to_pytree, states_to_full_pytree, _ = build_step_dynamics_fn(
        cell, solver="bwd_euler", delta_t=0.025, params=params
    )

    v_len_full = len(states_to_full_pytree(states_vec)["v"])
    v_len = len(states_to_pytree(states_vec)["v"])
    i_hh_len = len(states_to_full_pytree(states_vec)["i_HH"])

    if branchpoint:
        assert v_len == 24
        assert v_len_full == 25  # should be n_branches * ncomp_per_branch + 1
        assert i_hh_len == 25
    else:
        assert v_len == 8
        assert v_len_full == 8
        assert i_hh_len == 8
