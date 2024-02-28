import jax.numpy as jnp


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
    exp_term = jnp.exp(-dt / x_tau)
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

    exp_term = jnp.exp(slope * dt)

    return x * exp_term + s_inf * (1.0 - exp_term)
