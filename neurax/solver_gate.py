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
    slope = -(alpha + beta)
    xinf = -alpha / slope

    exp_term = jnp.exp(slope * dt)
    return x * exp_term + xinf * (1.0 - exp_term)
