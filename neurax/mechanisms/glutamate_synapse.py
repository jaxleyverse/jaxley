import jax.numpy as jnp


def glutamate(voltages, ss, pre_inds, pre_cell_inds, dt, synaptic_conds):
    """
    Compute membrane current and update gating variables with Hodgkin-Huxley equations.
    """

    e_syn = 0.0
    v_th = -35.0
    delta = 10.0
    k_minus = 1.0 / 40.0

    s_bar = 1.0 / (1.0 + jnp.exp((v_th - voltages[pre_cell_inds, pre_inds]) / delta))
    tau_s = (1.0 - s_bar) / k_minus

    s_inf = s_bar
    slope = -1.0 / tau_s
    exp_term = jnp.exp(slope * dt)
    new_s = ss * exp_term + s_inf * (1.0 - exp_term)

    non_zero_voltage_term = synaptic_conds * ss
    non_zero_constant_term = synaptic_conds * ss * e_syn

    return (non_zero_voltage_term, non_zero_constant_term), new_s
