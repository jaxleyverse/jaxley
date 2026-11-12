# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import jax
import jax.numpy as jnp
import pytest
from jax import value_and_grad

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse
from tests import builders as build


@pytest.mark.parametrize("key", ["HH_m", "v"])
@pytest.mark.parametrize("module_type", ["comp", "branch"])
def test_grad_against_finite_diff_initial_state(module_type, key):
    """The gradient w.r.t. an initial state matches central finite differences."""
    module = build.comp() if module_type == "comp" else build.branch(ncomp=4)

    def site():  # A new view each time, so that it sees inserted channels.
        return module if module_type == "comp" else module.loc(0.0)

    site().record()
    site().stimulate(
        jx.step_current(i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0)
    )
    site().insert(HH())
    site().set(key, 0.2 if key == "HH_m" else -70.0)
    site().make_trainable(key)
    params = module.get_parameters()

    # One compiled function for the gradient and both finite-difference evaluations.
    loss_and_grad = jax.jit(
        value_and_grad(lambda p: jnp.sum(jx.integrate(module, params=p)))
    )
    _, grad = loss_and_grad(params)

    # Central differences: forward differences are off by ~7% for `HH_m`.
    step = 0.01
    loss_plus, _ = loss_and_grad(jax.tree.map(lambda x: x + step, params))
    loss_minus, _ = loss_and_grad(jax.tree.map(lambda x: x - step, params))
    finite_diff = (loss_plus - loss_minus) / (2 * step)

    # Less than 1% relative error on the gradient.
    autodiff = grad[0][key]
    assert jnp.all(jnp.abs(autodiff - finite_diff) / jnp.abs(autodiff) < 0.01)


@pytest.mark.parametrize("key", ["radius", "length", "HH_gNa", "IonotropicSynapse_gS"])
def test_grad_against_finite_diff_parameters(key):
    """The gradient w.r.t. a parameter matches central finite differences.

    The parameter is scaled by `1 +- eps`, which is compared with the gradient in the
    direction of the parameter itself."""
    net = build.net(ncells=2, nbranches=2, ncomp=2)
    net.insert(HH())
    connect(
        net.cell(0).branch(0).comp(0),
        net.cell(1).branch(1).comp(1),
        IonotropicSynapse(),
    )
    net.cell(0).branch(0).comp(0).stimulate(
        jx.step_current(i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=10.0)
    )
    net.cell(1).branch(1).comp(1).record()
    net.make_trainable(key)
    params = net.get_parameters()

    loss_and_grad = jax.jit(
        value_and_grad(lambda p: jnp.sum(jx.integrate(net, params=p)))
    )
    _, grad = loss_and_grad(params)
    directional = sum(jnp.sum(g[key] * p[key]) for g, p in zip(grad, params))

    eps = 1e-4
    loss_plus, _ = loss_and_grad(jax.tree.map(lambda x: x * (1 + eps), params))
    loss_minus, _ = loss_and_grad(jax.tree.map(lambda x: x * (1 - eps), params))
    finite_diff = (loss_plus - loss_minus) / (2 * eps)

    assert jnp.abs(directional - finite_diff) / jnp.abs(directional) < 0.01
