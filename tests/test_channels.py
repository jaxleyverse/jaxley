# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from math import pi
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import (
    HH,
    CaL,
    CaT,
    Channel,
    Fire,
    Izhikevich,
    K,
    Km,
    Leak,
    Na,
    Rate,
)
from jaxley.solver_gate import exponential_euler, save_exp, solve_inf_gate_exponential


class CaPump(Channel):
    """Calcium dynamics tracking inside calcium concentration, modeled after Destexhe et al. 1994."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered)
            f"{self._name}_decay": 80,  # Rate of removal of calcium in ms
            f"{self._name}_depth": 0.1,  # Depth of shell in um
            f"{self._name}_minCai": 1e-4,  # Minimum intracellular calcium concentration in mM
        }
        self.channel_states = {
            f"CaCon_i": 5e-05,  # Initial internal calcium concentration in mM
        }
        self.current_name = f"i_Ca"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["i_Ca"] / 1_000.0
        cai = u["CaCon_i"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole

        # Calculate the contribution of calcium currents to cai change
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)

        cai_tau = decay
        cai_inf = minCai + decay * drive_channel
        new_cai = exponential_euler(cai, dt, cai_inf, cai_tau)

        return {f"CaCon_i": new_cai}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaNernstReversal(Channel):
    """Compute Calcium reversal from inner and outer concentration of calcium."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_constants = {
            "F": 96485.3329,  # C/mol (Faraday's constant)
            "T": 279.45,  # Kelvin (temperature)
            "R": 8.314,  # J/(mol K) (gas constant)
        }
        self.channel_params = {}
        self.channel_states = {"eCa": 0.0, "CaCon_i": 5e-05, "CaCon_e": 2.0}
        self.current_name = f"i_Ca"

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Cai = u["CaCon_i"]
        Cao = u["CaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vCa = C * jnp.log(Cao / Cai)
        return {"eCa": vCa, "CaCon_i": Cai, "CaCon_e": Cao}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


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
    branch = jx.Branch(comp, ncomp=4)

    branch.loc(0.0).insert(standard_hh)
    branch.insert(neuron_hh)

    branch.loc(1.0).record()
    v = jx.integrate(branch, t_max=1.0)

    # Test if voltage is `NaN` which happens when channels get mixed up.
    assert np.invert(np.any(np.isnan(v)))


@pytest.mark.slow
def test_init_states(SimpleCell):
    """Functional test for `init_states()`.

    Checks whether, if everything is initialized in its steady state, the voltage
    after 10ms is almost exactly the same as after 0ms.
    """
    cell = SimpleCell(2, 4)
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
        self.current_is_in_mA_per_cm2 = True
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
        return {f"{prefix}_m": m_inf}

    @staticmethod
    def m_gate(v, cai, q10):
        cai = cai * 1e3
        v_half = -66 + 137 * save_exp(-0.3044 * cai) + 30.24 * save_exp(-0.04141 * cai)
        alpha = 25.0

        beta = 0.075 / save_exp((v - v_half) / 10)
        m_inf = alpha / (alpha + beta)
        tau_m = 1.0 * q10
        return m_inf, tau_m


def test_init_states_complex_channel(SimpleCell):
    """Test for `init_states()` with a more complicated channel model.

    The channel model used for this test uses the `states` in `init_state` and it also
    uses `q10`. The model inserts the channel only is some branches. This test follows
    an issue I had with Jaxley in v0.2.0 (fixed in v0.2.1).
    """
    ## Create cell
    cell = SimpleCell(3, 1)

    # CA channels.
    cell.branch([0, 1]).insert(CaNernstReversal())
    cell.branch([0, 1]).insert(CaPump())
    cell.branch([0, 1]).insert(KCA11())

    cell.init_states()

    current = jx.step_current(
        i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0
    )
    cell.branch(2).comp(0).stimulate(current)
    cell.branch(2).comp(0).record()
    voltages = jx.integrate(cell)
    assert np.invert(np.any(np.isnan(voltages))), "NaN voltage found"


def test_multiple_channel_currents(SimpleCell):
    """Test whether all channels can"""

    class User(Channel):
        """The channel which uses currents of Dummy1 and Dummy2 to update its states."""

        def __init__(self, name: Optional[str] = None):
            self.current_is_in_mA_per_cm2 = True
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
            self.current_is_in_mA_per_cm2 = True
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
            self.current_is_in_mA_per_cm2 = True
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {}
            self.current_name = f"i_Dummy"

        def update_states(self, states, dt, v, params):
            return {}

        def compute_current(self, states, v, params):
            return 0.01 * jnp.ones_like(v)

    dt = 0.025  # ms
    t_max = 5.0  # ms
    cell = SimpleCell(1, 1)
    cell.branch(0).loc(0.0).stimulate(
        jx.step_current(i_delay=0.5, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=5.0)
    )

    cell.insert(User())
    cell.insert(Dummy1())
    cell.insert(Dummy2())
    cell.branch(0).loc(0.0).record("cumulative")

    s = jx.integrate(cell, delta_t=dt)

    num_channels = 2
    target = (t_max // dt + 2) * 0.001 * 0.01 * num_channels
    assert np.abs(target - s[0, -1]) < 1e-8


def test_delete_channel(SimpleBranch):
    # test complete removal of a channel from a module
    branch1 = SimpleBranch(ncomp=3)
    branch1.comp(0).insert(K())
    branch1.delete(K())

    branch2 = SimpleBranch(ncomp=3)
    branch2.comp(0).insert(K())
    branch2.comp(0).delete(K())

    branch3 = SimpleBranch(ncomp=3)
    branch3.insert(K())
    branch3.delete(K())

    def channel_present(view, channel, partial=False):
        states_and_params = list(channel.channel_states.keys()) + list(
            channel.channel_params.keys()
        )
        # none of the states or params should be in nodes
        cols = view.nodes.columns.to_list()
        channel_cols = [
            col
            for col in cols
            if col.startswith(channel._name) and col != channel._name
        ]
        diff = set(channel_cols).difference(set(states_and_params))
        has_params_or_states = len(diff) > 0
        has_channel_col = channel._name in view.nodes.columns
        has_channel = channel._name in [c._name for c in view.channels]
        has_mem_current = channel.current_name in view.membrane_current_names
        if partial:
            all_nans = (
                not view.nodes[channel_cols].isna().all().all()
                & ~view.nodes[channel._name].all()
            )
            return has_channel or has_mem_current or all_nans
        return has_params_or_states or has_channel_col or has_channel or has_mem_current

    for branch in [branch1, branch2, branch3]:
        assert len(branch.channels) == 0
        assert not channel_present(branch, K())

    # test correct channels are removed only in the viewed part of the module
    branch4 = SimpleBranch(ncomp=3)
    branch4.insert(HH())
    branch4.comp(0).insert(K())
    branch4.comp([1, 2]).insert(Leak())

    branch4.comp(1).delete(Leak())
    # assert K in comp 0 and Leak still present in branch
    assert channel_present(branch4.comp(0), K())
    assert channel_present(branch4.comp(2), Leak(), partial=True)
    assert not channel_present(branch4.comp(1), Leak(), partial=True)
    assert channel_present(branch4, Leak())

    branch4.comp(2).delete(Leak())
    # assert no more Leak
    assert not channel_present(branch4, Leak())


@pytest.mark.parametrize("solver", ["fwd_euler", "bwd_euler"])
def test_lif(solver):
    cell = jx.Cell()
    cell.insert(Leak())
    cell.insert(Fire())
    cell.record("v")
    cell.record("Fire_spikes")

    dt = 0.1
    t_max = 40.0

    cell.stimulate(jx.step_current(5.0, 20.0, 0.005, dt, t_max))
    v = jx.integrate(cell, delta_t=dt, solver=solver)
    assert np.all(v[0] > -71.0)
    assert np.all(v[0] < -40.0)
    assert np.sum(v[1]) == 6.0


@pytest.mark.parametrize("solver", ["fwd_euler", "bwd_euler"])
def test_izhikevich(solver):
    cell = jx.Cell()
    cell.insert(Izhikevich())
    cell.record("v")

    dt = 0.1
    t_max = 200.0

    cell.stimulate(jx.step_current(10.0, 180.0, 0.012, dt, t_max))
    v = jx.integrate(cell, delta_t=dt, solver=solver)

    v_250307 = jnp.asarray(
        [
            -70.0,
            -65.0123028,
            -69.20009282,
            -71.46372414,
            -70.71902502,
            -53.02998553,
            -63.15451445,
            -67.04028155,
            -69.77691688,
            -71.87443646,
            -82.1561912,
        ]
    )
    max_error = np.max(np.abs(v[0, ::200] - v_250307))
    assert max_error < 1e-8, f"Error {max_error} to large."


def test_rate_channel():
    cell = jx.Cell()
    cell.set("length", 1.0 / (2 * pi * 1e-5))
    cell.set("radius", 1.0)  # 1.0 is also the default.
    cell.insert(Rate())
    cell.record("v")

    cell.stimulate(2.0 * jnp.ones((5,)))
    cell.set("v", 2.0)
    cell.record("v")

    dt = 1.0
    t_max = 10.0
    v = jx.integrate(cell, t_max=t_max, delta_t=dt)

    v_250307 = jnp.asarray(
        [
            2.0,
            2.73575888,
            3.00642945,
            3.10600359,
            3.14263486,
            3.15611076,
            1.16106826,
            0.42713314,
            0.1571335,
            0.05780618,
            0.02126571,
            0.00782322,
        ]
    )
    max_error = np.max(np.abs(v[0] - v_250307))
    assert max_error < 1e-8, f"Error {max_error} to large."


def test_multicompartment_lif(SimpleCell):
    """Test that LIF can be run in multicompartment simulation."""
    cell = SimpleCell(2, 4)
    # Leak everywhere, fire only in "soma".
    cell.insert(Leak())
    cell.branch(0).comp(0).insert(Fire())
    cell.record("v")
    cell.branch(0).comp(0).record("Fire_spikes")

    dt = 0.1
    t_max = 40.0

    cell.branch(0).comp(0).stimulate(jx.step_current(5.0, 20.0, 0.01, dt, t_max))
    recordings = jx.integrate(cell, delta_t=dt)
    assert np.invert(np.any(np.isnan(recordings)))


def test_multicompartment_izhikevich(SimpleCell):
    """Test that Izhikevich can be run in multicompartment simulation."""
    cell = SimpleCell(2, 4)
    # Leak everywhere, fire only in "soma".
    cell.insert(Leak())
    cell.branch(0).comp(0).insert(Fire())
    cell.record("v")
    cell.branch(0).comp(0).record("Fire_spikes")

    dt = 0.1
    t_max = 40.0

    cell.branch(0).comp(0).stimulate(jx.step_current(5.0, 20.0, 0.02, dt, t_max))
    recordings = jx.integrate(cell, delta_t=dt)
    assert np.invert(np.any(np.isnan(recordings)))
