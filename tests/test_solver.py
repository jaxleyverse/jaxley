# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.solver_gate import exponential_euler
from jaxley.solver_voltage import _make_dhs_solve
from jaxley.synapses import IonotropicSynapse


@pytest.mark.parametrize("x_inf", [3.0, 30.0])
def test_exp_euler(x_inf):
    """Test whether `dx = -(x - x_inf) / x_tau` is solved correctly."""
    x = 20.0
    x_tau = 12.0
    dt = 0.1
    vectorfield = -(x - x_inf) / x_tau
    fwd_euler = x + dt * vectorfield
    exp_euler = exponential_euler(x, dt, x_inf, x_tau)
    assert np.abs(fwd_euler - exp_euler) / np.abs(fwd_euler) < 1e-4


def test_fwd_euler_and_crank_nicolson(SimpleNet):
    """Compare accuracies of different solvers."""
    net = SimpleNet(2, 3, 4, connect=True)

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    net.cell(0).branch(0).comp(0).stimulate(current)
    net.cell(1).branch(0).comp(3).record()

    net.IonotropicSynapse.set("IonotropicSynapse_gS", 0.01)
    net.insert(HH())

    # As expected, using significantly shorter compartments or lower r_a leads to NaN
    # in forward Euler.
    net.set("length", 50.0)
    net.set("axial_resistivity", 200.0)

    bwd_v = jx.integrate(net, solver="bwd_euler")
    fwd_v = jx.integrate(net, solver="fwd_euler")
    crank_v = jx.integrate(net, solver="crank_nicolson")
    exp_v = jx.integrate(net, solver="exp_euler")

    # These allowed voltage differences may appear large, but due to the steepness
    # of a spike these values actually correspond to sub-millisecond spike time
    # differences.
    assert np.max(np.abs(bwd_v - fwd_v)) < 12.0
    assert np.max(np.abs(bwd_v - crank_v)) < 6.0
    assert np.max(np.abs(bwd_v - exp_v)) < 4.0


def test_exp_euler_solver_customization(SimpleCell):
    cell = SimpleCell(2, 3)
    delta_t = 0.025

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=delta_t, t_max=5.0
    )
    cell.branch(0).comp(0).stimulate(current)
    cell.branch(0).comp(1).record()
    cell.insert(HH())

    cell.customize_solver_exp_euler(
        exp_euler_transition=cell.build_exp_euler_transition_matrix(delta_t)
    )
    v = jx.integrate(cell, solver="exp_euler")
    assert np.invert(np.any(np.isnan(v)))


@pytest.mark.parametrize("optimize_for_gpu", [False, True])
def test_dhs_solve_handles_ragged_grouped_edges(optimize_for_gpu):
    solve_indexer = {
        "node_order_grouped": [
            np.asarray([[1, 0], [2, 0]], dtype=int),
            np.asarray([[3, 1]], dtype=int),
        ],
        "all_children": np.asarray([1, 2, 3], dtype=int),
        "all_parents": np.asarray([0, 0, 1], dtype=int),
        "parent_lookup": np.asarray([-1, 0, 0, 1, -1], dtype=int),
    }
    solve = _make_dhs_solve(
        solve_indexer=solve_indexer,
        optimize_for_gpu=optimize_for_gpu,
        n_nodes=4,
    )

    diags = jnp.asarray([4.0, 5.0, 6.0, 7.0, 1.0])
    lowers = jnp.asarray([0.0, -0.4, -0.3, -0.2, 0.0])
    uppers = jnp.asarray([0.0, -0.5, -0.1, -0.6, 0.0])
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0, 0.0])

    matrix = np.diag(np.asarray(diags))
    for child, parent in zip(
        solve_indexer["all_children"], solve_indexer["all_parents"]
    ):
        matrix[parent, child] = float(uppers[child])
        matrix[child, parent] = float(lowers[child])

    expected = np.linalg.solve(matrix, np.asarray(rhs))
    actual = np.asarray(jax.jit(solve)(diags, lowers, uppers, rhs))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    grad_fn = jax.jit(jax.grad(lambda b: jnp.sum(solve(diags, lowers, uppers, b))))
    actual_grad = np.asarray(grad_fn(rhs))
    expected_grad = np.linalg.solve(matrix.T, np.ones_like(expected))
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)


def test_dhs_vmap_closed_over_rhs_matches_arg():
    """``jit(vmap(DHS))`` must agree for argument vs closed-over multi-column RHS.

    On jax/jaxlib in roughly ``[0.5.3, 0.8.1]``, the old RHS form
    ``v[i] + dt * c[i]`` inside ``step_voltage_implicit_with_dhs_solve`` returned
    wrong primals when the RHS matrix was a compile-time constant (jax#33479 /
    xla#34260). This regression guards the scale-before-gather rewrite.
    """
    from jaxley.channels import Leak
    from jaxley.integrate import build_init_and_step_fn
    from jaxley.solver_voltage import step_voltage_implicit_with_dhs_solve

    branch = jx.Branch(jx.Compartment(), ncomp=4)
    cell = jx.Cell([branch] * 3, parents=[-1, 0, 0])
    cell.insert(Leak())
    cell.to_jax()
    build_init_and_step_fn(cell, voltage_solver="jaxley.dhs.cpu")

    N = len(cell.nodes)
    n_nodes = int(cell._n_nodes)
    dt = 0.025
    idxr = cell._dhs_solve_indexer
    internal = cell._internal_node_inds
    sinks = np.asarray(cell._comp_edges["sink"].to_list())
    ap = cell.get_all_parameters([])
    axial = jnp.asarray(ap["axial_conductances"]["v"])
    g_leak = jnp.asarray(ap["Leak_gLeak"][:N])
    cm = jnp.asarray(ap["capacitance"][:N])
    vt = jnp.zeros((n_nodes,)).at[:N].set(g_leak / cm)
    B = jax.random.normal(jax.random.PRNGKey(0), (N, 8))

    def solve_col(rhs):
        voltages = jnp.zeros((n_nodes,)).at[:N].set(rhs)
        out = step_voltage_implicit_with_dhs_solve(
            voltages,
            vt,
            jnp.zeros_like(voltages),
            axial,
            internal,
            sinks,
            n_nodes,
            idxr,
            False,
            dt,
        )
        return out[:N]

    def solve_mat(M):
        return jax.vmap(solve_col, in_axes=1, out_axes=1)(M)

    eager = solve_mat(B)
    as_arg = jax.jit(solve_mat)(B)
    closed = jax.jit(lambda: solve_mat(B))()
    np.testing.assert_allclose(as_arg, eager, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(closed, eager, rtol=1e-10, atol=1e-10)
