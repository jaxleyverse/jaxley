import jax.numpy as jnp


def m_gate(v):
    V_T = -63.0
    alpha = -0.32 * (v - V_T - 13) / (jnp.exp(-(v - V_T - 13) / 4) - 1)
    beta = 0.28 * (v - V_T - 40) / (jnp.exp((v - V_T - 40) / 5) - 1)
    return alpha, beta


def h_gate(v):
    V_T = -63.0
    alpha = 0.128 * jnp.exp(-(v - V_T - 17) / 18)
    beta = 4 / (1 + jnp.exp(-(v - V_T - 40) / 5))
    return alpha, beta


def n_gate(v):
    V_T = -63.0
    alpha = -0.032 * (v - V_T - 15) / (jnp.exp(-(v - V_T - 15) / 5) - 1)
    beta = 0.5 * jnp.exp(-(v - V_T - 10) / 40)
    return alpha, beta
