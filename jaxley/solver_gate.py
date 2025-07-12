# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import jax.numpy as jnp
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


def solve_inf_gate_exponential(
    x: ArrayLike,
    dt: float,
    s_inf: ArrayLike,
    tau_s: ArrayLike,
):
    """solves dx/dt = (s_inf - x) / tau_s
    via exponential Euler

    Args:
        x (ArrayLike): gate variable
        dt (float): time_delta
        s_inf (ArrayLike): _description_
        tau_s (ArrayLike): _description_

    Returns:
        _type_: updated gate
    """
    slope = -1.0 / tau_s
    exp_term = save_exp(slope * dt)
    return x * exp_term + s_inf * (1.0 - exp_term)
