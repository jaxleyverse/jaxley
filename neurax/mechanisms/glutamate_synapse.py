import jax.numpy as jnp


def glutamate(
    voltages, ss, pre_inds, pre_cell_inds, grouped_inds, post_syn, dt, synaptic_conds
):
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

    voltage_term = jnp.zeros_like(voltages)
    constant_term = jnp.zeros_like(voltages)

    for g, p in zip(grouped_inds, post_syn):
        summed_volt = jnp.sum(non_zero_voltage_term[g], axis=1)
        summed_const = jnp.sum(non_zero_constant_term[g], axis=1)

        voltage_term = voltage_term.at[p[:, 0], p[:, 1]].set(summed_volt)
        constant_term = constant_term.at[p[:, 0], p[:, 1]].set(summed_const)

    return (voltage_term, constant_term), (new_s,)
