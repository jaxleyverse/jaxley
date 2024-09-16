# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np
import pytest
from jaxley_mech.channels.l5pc import CaNernstReversal, CaPump

import jaxley as jx
from jaxley.channels import HH, CaL, CaT, Channel, K, Km, Leak, Na
from jaxley.solver_gate import save_exp, solve_inf_gate_exponential


def test_channel_set_name():
    # default name is the class name
    assert Na().name == "Na"

    # channel name can be set in the constructor
    na = Na(name="NaPospischil")
    assert na.name == "NaPospischil"
    assert "NaPospischil_gNa" in na.channel_params.keys()
    assert "eNa" in na.channel_params.keys()
    assert "NaPospischil_h" in na.channel_states.keys()
    assert "NaPospischil_m" in na.channel_states.keys()
    assert "NaPospischil_vt" not in na.channel_params.keys()
    assert "vt" in na.channel_params.keys()

    # channel name can not be changed directly
    k = K()
    with pytest.raises(AttributeError):
        k.name = "KPospischil"
    assert "KPospischil_gNa" not in k.channel_params.keys()
    assert "eNa" not in k.channel_params.keys()
    assert "KPospischil_h" not in k.channel_states.keys()
    assert "KPospischil_m" not in k.channel_states.keys()


def test_channel_change_name():
    # channel name can be changed with change_name method
    # (and only this way after initialization)
    na = Na().change_name("NaPospischil")
    assert na.name == "NaPospischil"
    assert "NaPospischil_gNa" in na.channel_params.keys()
    assert "eNa" in na.channel_params.keys()
    assert "NaPospischil_h" in na.channel_states.keys()
    assert "NaPospischil_m" in na.channel_states.keys()
    assert "NaPospischil_vt" not in na.channel_params.keys()
    assert "vt" in na.channel_params.keys()


def test_integration_with_renamed_channels():
    neuron_hh = HH().change_name("NeuronHH")
    standard_hh = HH()

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)

    branch.loc(0.0).insert(standard_hh)
    branch.insert(neuron_hh)

    branch.loc(1.0).record()
    v = jx.integrate(branch, t_max=1.0)

    # Test if voltage is `NaN` which happens when channels get mixed up.
    assert np.invert(np.any(np.isnan(v)))


def test_init_states():
    """Functional test for `init_states()`.

    Checks whether, if everything is initialized in its steady state, the voltage
    after 10ms is almost exactly the same as after 0ms.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, 4)
    cell = jx.Cell(branch, [-1, 0])
    cell.branch(0).loc(0.0).record()

    cell.branch(0).insert(Na())
    cell.branch(1).insert(K())
    cell.branch(1).loc(0.0).insert(Km())
    cell.branch(0).loc(1.0).insert(CaT())
    cell.insert(CaL())
    cell.insert(Leak())

    cell.insert(HH())

    cell.set("v", -62.0)  # At -70.0 there is a rebound spike.
    cell.init_states()
    v = jx.integrate(cell, t_max=20.0)

    last_voltage = v[0, -1]
    cell.set("v", last_voltage)
    cell.init_states()

    v = jx.integrate(cell, t_max=10.0)
    assert np.abs(v[0, 0] - v[0, -1]) < 0.02


class KCA11(Channel):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_q10_ch": 3,
            f"{prefix}_q10_ch0": 22,
            "celsius": 22,
        }
        self.channel_states = {f"{prefix}_m": 0.02, "CaCon_i": 1e-4}
        self.current_name = f"i_K"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update state."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        q10 = params[f"{prefix}_q10_ch"] ** (
            (params["celsius"] - params[f"{prefix}_q10_ch0"]) / 10
        )
        cai = states["CaCon_i"]
        new_m = solve_inf_gate_exponential(m, dt, *self.m_gate(v, cai, q10))
        return {f"{prefix}_m": new_m}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current."""
        prefix = self._name
        m = states[f"{prefix}_m"]
        g = 0.03 * m * 1000  # mS/cm^2
        return g * (v + 80.0)

    def init_state(self, states, v, params, dt):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        q10 = params[f"{prefix}_q10_ch"] ** (
            (params["celsius"] - params[f"{prefix}_q10_ch0"]) / 10
        )
        cai = states["CaCon_i"]
        m_inf, _ = self.m_gate(v, cai, q10)
        return {
            f"{prefix}_m": m_inf,
        }

    @staticmethod
    def m_gate(v, cai, q10):
        cai = cai * 1e3
        v_half = -66 + 137 * save_exp(-0.3044 * cai) + 30.24 * save_exp(-0.04141 * cai)
        alpha = 25.0

        beta = 0.075 / save_exp((v - v_half) / 10)
        m_inf = alpha / (alpha + beta)
        tau_m = 1.0 * q10
        return m_inf, tau_m


def test_init_states_complex_channel():
    """Test for `init_states()` with a more complicated channel model.

    The channel model used for this test uses the `states` in `init_state` and it also
    uses `q10`. The model inserts the channel only is some branches. This test follows
    an issue I had with Jaxley in v0.2.0 (fixed in v0.2.1).
    """
    ## Create cell
    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=1)
    cell = jx.Cell(branch, parents=[-1, 0, 0])

    # CA channels.
    cell.branch([0, 1]).insert(CaNernstReversal())
    cell.branch([0, 1]).insert(CaPump())
    cell.branch([0, 1]).insert(KCA11())

    cell.init_states()

    current = jx.step_current(1.0, 1.0, 0.1, 0.025, 3.0)
    cell.branch(2).comp(0).stimulate(current)
    cell.branch(2).comp(0).record()
    voltages = jx.integrate(cell)
    assert np.invert(np.any(np.isnan(voltages))), "NaN voltage found"


def test_multiple_channel_currents():
    """Test whether all channels can"""

    class User(Channel):
        """The channel which uses currents of Dummy1 and Dummy2 to update its states."""

        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {"cumulative": 0.0}
            self.current_name = f"i_User"

        def update_states(self, states, dt, v, params):
            state = states["cumulative"]
            state += states["i_Dummy"] * 0.001
            return {"cumulative": state}

        def compute_current(self, states, v, params):
            return 0.01 * jnp.ones_like(v)

    class Dummy1(Channel):
        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {}
            self.current_name = f"i_Dummy"

        def update_states(self, states, dt, v, params):
            return {}

        def compute_current(self, states, v, params):
            return 0.01 * jnp.ones_like(v)

    class Dummy2(Channel):
        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {}
            self.current_name = f"i_Dummy"

        def update_states(self, states, dt, v, params):
            return {}

        def compute_current(self, states, v, params):
            return 0.01 * jnp.ones_like(v)

    dt = 0.025  # ms
    t_max = 10.0  # ms
    comp = jx.Compartment()
    branch = jx.Branch(comp, 1)
    cell = jx.Cell(branch, parents=[-1])
    cell.branch(0).loc(0.0).stimulate(jx.step_current(1.0, 2.0, 0.1, dt, t_max))

    cell.insert(User())
    cell.insert(Dummy1())
    cell.insert(Dummy2())
    cell.branch(0).loc(0.0).record("cumulative")

    s = jx.integrate(cell, delta_t=dt)

    num_channels = 2
    target = (t_max // dt + 2) * 0.001 * 0.01 * num_channels
    assert np.abs(target - s[0, -1]) < 1e-8
