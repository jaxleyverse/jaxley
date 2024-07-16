# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax.numpy as jnp


def save_exp(x, max_value: float = 20.0):
    """Clip the input to a maximum value and return its exponential."""
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)


def solve_gate_implicit(
    gating_state: jnp.ndarray,
    dt: float,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
):
    a_m = gating_state + dt * alpha
    b_m = 1.0 + dt * alpha + dt * beta

    return a_m / b_m


def solve_gate_exponential(
    x: jnp.ndarray,
    dt: float,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
):
    tau = 1 / (alpha + beta)
    xinf = alpha * tau
    return exponential_euler(x, dt, xinf, tau)


def exponential_euler(
    x: jnp.ndarray,
    dt: float,
    x_inf: jnp.ndarray,
    x_tau: jnp.ndarray,
):
    """An exact solver for the linear dynamical system `dx = -(x - x_inf) / x_tau`."""
    exp_term = save_exp(-dt / x_tau)
    return x * exp_term + x_inf * (1.0 - exp_term)


def solve_inf_gate_exponential(
    x: jnp.ndarray,
    dt: float,
    s_inf: jnp.ndarray,
    tau_s: jnp.ndarray,
):
    """solves dx/dt = (s_inf - x) / tau_s
    via exponential Euler

    Args:
        x (jnp.ndarray): gate variable
        dt (float): time_delta
        s_inf (jnp.ndarray): _description_
        tau_s (jnp.ndarray): _description_

    Returns:
        _type_: updated gate
    """
    slope = -1.0 / tau_s
    exp_term = save_exp(slope * dt)
    return x * exp_term + s_inf * (1.0 - exp_term)
