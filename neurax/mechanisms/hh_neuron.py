import jax.numpy as jnp


def m_gate(v):
    alpha = 0.1 * vtrap(-(v + 40), 10)
    beta = 4.0 * jnp.exp(-(v + 65) / 18)
    return alpha, beta


def h_gate(v):
    alpha = 0.07 * jnp.exp(-(v + 65) / 20)
    beta = 1.0 / (jnp.exp(-(v + 35) / 10) + 1)
    return alpha, beta


def n_gate(v):
    alpha = 0.01 * vtrap(-(v + 55), 10)
    beta = 0.125 * jnp.exp(-(v + 65) / 80)
    return alpha, beta


def vtrap(x, y):
    return x / (jnp.exp(x / y) - 1.0)