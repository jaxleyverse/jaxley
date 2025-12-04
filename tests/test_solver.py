# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.solver_gate import exponential_euler
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
