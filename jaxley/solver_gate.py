# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import jax.numpy as jnp
from jax import custom_gradient
from jax.typing import ArrayLike


def save_exp(x, max_value: float = 20.0):
    """Clip the input to a maximum value and return its exponential."""
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)


def solve_gate_implicit(
    gating_state: ArrayLike,
    dt: float,
    alpha: ArrayLike,
    beta: ArrayLike,
):
    a_m = gating_state + dt * alpha
    b_m = 1.0 + dt * alpha + dt * beta

    return a_m / b_m


def solve_gate_exponential(
    x: ArrayLike,
    dt: float,
    alpha: ArrayLike,
    beta: ArrayLike,
):
    tau = 1 / (alpha + beta)
    xinf = alpha * tau
    return exponential_euler(x, dt, xinf, tau)


def exponential_euler(
    x: ArrayLike,
    dt: float,
    x_inf: ArrayLike,
    x_tau: ArrayLike,
):
    """An exact solver for the linear dynamical system `dx = -(x - x_inf) / x_tau`."""
    exp_term = save_exp(-dt / x_tau)
    return x * exp_term + x_inf * (1.0 - exp_term)


def heaviside(x: ArrayLike, at_zero: ArrayLike = 1.0, grad_scale: float = 10.0):
    """Compute the heaviside step function with a custom derivative.

    Jaxley implementation of ``jax.numpy.heaviside``, which includes a custom
    derivative.

    The custom derivative is $\\frac{1}{(g|x| + 1)^2}$ where g is ``grad_scale``.
    If you experience exploding or vanishing derivatives when using this function,
    try to change the value of `grad_scale` to remedy the problem.

    Note while this function works for ``x`` and ``at_zero`` being jax arrays,
    you can only take the gradient of this function when both are scalar values.

    Args:
        x: Input array or scalar. ``complex`` dtype are not supported.
        at_zero: Scalar or array. Specifies the return values when ``x`` is ``0``.
            ``complex`` dtype are not supported. ``x`` and ``at_zero`` must either
            have same shape or broadcast compatible.
        grad_scale: Specifies the flatness of the gradient curve. Larger values
            correspond to being closer to the 'real' gradient, however makes function
            more susceptible to exploding/vanishing gradients.

    Returns:
        An array containing the heaviside step function of ``x``, promoting to
        inexact dtype.
    """

    @custom_gradient
    def _heaviside_custom(x):
        return (
            jnp.heaviside(x, at_zero),
            lambda g: (g / (grad_scale * jnp.abs(x) + 1.0) ** 2),
        )

    return _heaviside_custom(x)
