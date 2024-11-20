# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

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
from jaxley.optimize.utils import l2_norm


def test_type_optimizer_api(SimpleComp):
    """Tests whether optimization recovers a ground truth parameter set."""
    comp = SimpleComp(copy=True)
    comp.insert(HH())
    comp.record()
    current = jx.step_current(
        i_delay=0.1, i_dur=3.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    comp.stimulate(current)

    def simulate(params):
        return jx.integrate(comp, params=params)

    comp.make_trainable("HH_gNa")
    comp.make_trainable("radius")

    def loss_fn(params):
        return jnp.mean(simulate(params))

    grad_fn = jit(value_and_grad(loss_fn))

    # Diverse lr -> good learning.
    opt_params = comp.get_parameters()
    lrs = {"HH_gNa": [0.01, 0.1], "radius": [1.0, 0.2]}
    optimizer = TypeOptimizer(
        lambda args: optax.sgd(args[0], momentum=args[1]), lrs, opt_params
    )
    opt_state = optimizer.init(opt_params)

    _, grad = grad_fn(opt_params)
    updates, opt_state = optimizer.update(grad, opt_state)
    opt_params = optax.apply_updates(opt_params, updates)


def test_type_optimizer(SimpleComp):
    """Tests whether optimization recovers a ground truth parameter set."""
    comp = SimpleComp(copy=True)
    comp.insert(HH())
    comp.record()
    current = jx.step_current(
        i_delay=0.1, i_dur=3.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    comp.stimulate(current)

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


def test_l2_norm_utility():
    true_norm = np.sqrt(np.sum(np.asarray([0.01, 0.003, 0.05, 0.006, 0.07, 0.04]) ** 2))
    pytree = [
        {"a": 0.01},
        {"b": jnp.asarray([[0.003, 0.05]])},
        {"c": jnp.asarray([[0.006, 0.07]])},
        0.04,
    ]
    assert l2_norm(pytree).item() == true_norm
