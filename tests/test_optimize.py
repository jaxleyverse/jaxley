import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.optimize import TypeOptimizer


def test_type_optimizer():
    """Tests whether optimization recovers a ground truth parameter set."""
    comp = jx.Compartment()
    comp.insert(HH())
    comp.record()
    comp.stimulate(jx.step_current(0.1, 3.0, 0.1, 0.025, 5.0))

    comp.set("HH_gNa", 0.4)
    comp.set("radius", 30.0)

    def simulate(params):
        return jx.integrate(comp, params=params)

    observation = simulate([{}])
    comp.set("HH_gNa", 0.3)
    comp.make_trainable("HH_gNa")

    comp.set("radius", 20.0)
    comp.make_trainable("radius")

    def loss_fn(params):
        res = simulate(params)
        return jnp.mean((observation - res) ** 2)

    grad_fn = jit(value_and_grad(loss_fn))

    # Diverse lr -> good learning.
    opt_params = comp.get_parameters()
    lrs = {"HH_gNa": 0.01, "radius": 1.0}
    optimizer = TypeOptimizer(optax.adam, lrs, opt_params)
    opt_state = optimizer.init(opt_params)

    for i in range(500):
        l, grad = grad_fn(opt_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)
    assert l < 1e-5, "Loss should be low if a diverse lr is used."

    # Too low lr -> poor learning.
    opt_params = comp.get_parameters()
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(opt_params)

    for i in range(500):
        l, grad = grad_fn(opt_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

    assert l > 3.0, "Loss should be high if a uniformly low lr is used."

    # Too high lr -> poor learning.
    opt_params = comp.get_parameters()
    optimizer = optax.adam(1.0)
    opt_state = optimizer.init(opt_params)

    # Run two epochs to ensure everything also works after a full run-through.
    for i in range(500):
        l, grad = grad_fn(opt_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

    assert l > 30.0, "Loss should be high if a uniformly high lr is used."