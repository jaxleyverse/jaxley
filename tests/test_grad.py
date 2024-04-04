import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import pytest
from jax import value_and_grad

import jaxley as jx
from jaxley.channels import HH


@pytest.mark.parametrize("key", ["HH_m", "v"])
def test_grad_against_finite_diff_initial_state(key):
    def simulate():
        return jnp.sum(jx.integrate(comp))

    def simulate_with_params(params):
        return jnp.sum(jx.integrate(comp, params=params))

    comp = jx.Compartment()
    comp.insert(HH())
    comp.record()
    comp.stimulate(jx.step_current(0.1, 0.2, 0.1, 0.025, 5.0))

    val = 0.2 if key == "HH_m" else -70.0
    step_size = 0.01

    # Finite differences.
    comp.set(key, val)
    r1 = simulate()
    comp.set(key, val + step_size)
    r2 = simulate()
    finitediff_grad = (r2 - r1) / step_size

    # Autodiff gradient.
    grad_fn = value_and_grad(simulate_with_params)
    comp.set(key, val)
    comp.make_trainable(key)
    params = comp.get_parameters()
    v, g = grad_fn(params)
    autodiff_grad = g[0][key]

    # Less than 5% error on the gradient difference.
    assert jnp.abs(autodiff_grad - finitediff_grad) / autodiff_grad < 0.05


@pytest.mark.parametrize("key", ["HH_m", "v"])
def test_branch_grad_against_finite_diff_initial_state(key):
    def simulate():
        return jnp.sum(jx.integrate(branch))

    def simulate_with_params(params):
        return jnp.sum(jx.integrate(branch, params=params))

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)
    branch.loc(0.0).record()
    branch.loc(0.0).stimulate(jx.step_current(0.1, 0.2, 0.1, 0.025, 5.0))
    branch.loc(0.0).insert(HH())

    val = 0.2 if key == "HH_m" else -70.0
    step_size = 0.01

    # Finite differences.
    branch.loc(0.0).set(key, val)
    r1 = simulate()
    branch.loc(0.0).set(key, val + step_size)
    r2 = simulate()
    finitediff_grad = (r2 - r1) / step_size

    # Autodiff gradient.
    grad_fn = value_and_grad(simulate_with_params)
    branch.loc(0.0).set(key, val)
    branch.loc(0.0).make_trainable(key)
    params = branch.get_parameters()
    v, g = grad_fn(params)
    autodiff_grad = g[0][key]

    # Less than 6% error on the gradient difference, error for HH_m is around 5.2%.
    assert jnp.abs(autodiff_grad - finitediff_grad) / autodiff_grad < 0.06
