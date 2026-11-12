# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from math import pi
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jaxley_mech.channels.pospischil import CaL, CaT, K, Km, Leak, Na

import jaxley as jx
from jaxley.channels import HH, AdEx, Channel, Fire, Izhikevich, Rate
from tests import builders as build
from tests.mechanisms import KCA11, CaNernstReversal, CaPump


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
def test_init_states():
    """Functional test for `init_states()`.

    Checks whether, if everything is initialized in its steady state, the voltage
    after 10ms is almost exactly the same as after 0ms.
    """
    cell = build.cell(nbranches=2, ncomp=4)
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


def test_init_states_with_groups_in_net():
    """PR #711: we had issues with the interaction of groups, nets, and init_states."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell1 = jx.Cell(branch, [-1, 0, 1])
    cell2 = jx.Cell(branch, [-1, 0, 1, 2])
    cell3 = jx.Cell(branch, [-1, 0])
    net = jx.Network([cell1, cell2, cell3])

    net.cell(0).branch(1).add_to_group("apical")
    net.cell(1).branch(3).add_to_group("apical")
    net.cell(2).branch(1).add_to_group("apical")

    net.apical.insert(HH())
    net.cell(0).branch(1).set("v", -80.0)
    net.cell(1).branch(3).set("v", -90.0)
    net.cell(2).branch(1).set("v", -100.0)

    net.init_states()
    init_state = net.nodes["HH_m"].to_numpy()[[2, 3, 12, 13, 16, 17]]
    true_init_state = np.asarray(
        [0.00804324, 0.00804324, 0.00210994, 0.00210994, 0.00053298, 0.00053298]
    )
    error = np.max(np.abs(true_init_state - init_state))
    assert error < 1e-4, "init_states error is too big for networks."


def test_init_states_split_hh_matches_hh():
    """`init_states()` with jaxley's `Na`, `K` and `Leak` matches the `HH` channel.

    `Na.init_state` and `K.init_state` used to call gate methods that do not exist,
    so `init_states()` raised an `AttributeError` on any module with these channels.
    """
    voltages = [-80.0, -65.0, -50.0]
    split = jx.Branch(jx.Compartment(), ncomp=3)
    split.insert(jx.channels.Na())
    split.insert(jx.channels.K())
    split.insert(jx.channels.Leak())
    combined = jx.Branch(jx.Compartment(), ncomp=3)
    combined.insert(HH())
    for module in [split, combined]:
        for i, v in enumerate(voltages):
            module.comp(i).set("v", v)
        module.init_states()

    for split_state, hh_state in [("Na_m", "HH_m"), ("Na_h", "HH_h"), ("K_n", "HH_n")]:
        np.testing.assert_allclose(
            split.nodes[split_state].to_numpy(), combined.nodes[hh_state].to_numpy()
        )
    # The steady state depends on the voltage, so the states differ per compartment.
    assert len(np.unique(split.nodes["Na_m"].to_numpy())) == len(voltages)


def test_init_states_complex_channel():
    """Test for `init_states()` with a more complicated channel model.

    The channel model used for this test uses the `states` in `init_state` and it also
    uses `q10`. The model inserts the channel only is some branches. This test follows
    an issue I had with Jaxley in v0.2.0 (fixed in v0.2.1).
    """
    ## Create cell
    cell = build.cell(nbranches=3, ncomp=1)

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


def test_multiple_channel_currents():
    """Test whether all channels can"""

    class User(Channel):
        """The channel which uses currents of Dummy1 and Dummy2 to update its states."""

        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {"cumulative": 0.0}
            self.current_name = f"i_User"

        def update_states(
            self,
            states: dict[str, Array],
            params: dict[str, Array],
            voltage: Array,
            delta_t: float,
        ):
            state = states["cumulative"]
            state += states["i_Dummy"] * 0.001
            return {"cumulative": state}

        def compute_current(
            self,
            states: dict[str, Array],
            params: dict[str, Array],
            voltage: Array,
            delta_t: float,
        ):
            return 0.01 * jnp.ones_like(voltage)

    class Dummy1(Channel):
        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {}
            self.current_name = f"i_Dummy"

        def update_states(
            self,
            states: dict[str, Array],
            params: dict[str, Array],
            voltage: Array,
            delta_t: float,
        ):
            return {}

        def compute_current(
            self,
            states: dict[str, Array],
            params: dict[str, Array],
            voltage: Array,
            delta_t: float,
        ):
            return 0.01 * jnp.ones_like(voltage)

    class Dummy2(Channel):
        def __init__(self, name: Optional[str] = None):
            super().__init__(name)
            self.channel_params = {}
            self.channel_states = {}
            self.current_name = f"i_Dummy"

        def update_states(
            self,
            states: dict[str, Array],
            params: dict[str, Array],
            voltage: Array,
            delta_t: float,
        ):
            return {}

        def compute_current(
            self,
            states: dict[str, Array],
            params: dict[str, Array],
            voltage: Array,
            delta_t: float,
        ):
            return 0.01 * jnp.ones_like(voltage)

    dt = 0.025  # ms
    t_max = 5.0  # ms
    cell = build.cell(nbranches=1, ncomp=1)
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


def test_delete_channel():
    # test complete removal of a channel from a module
    branch1 = build.branch(ncomp=3)
    branch1.comp(0).insert(K())
    branch1.delete(K())

    branch2 = build.branch(ncomp=3)
    branch2.comp(0).insert(K())
    branch2.comp(0).delete(K())

    branch3 = build.branch(ncomp=3)
    branch3.insert(K())
    branch3.delete(K())

    branch4 = build.branch(ncomp=3)
    branch4.insert(CaL())
    branch4.insert(CaT())
    branch4.delete(CaL())
    branch4.delete(CaT())

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
    branch4 = build.branch(ncomp=3)
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


def test_no_removal_of_shared_params():
    """If a param is present in a different channel, then it should not be removed."""
    na1 = Na("Na1")
    na1.channel_params["shared_gNa"] = na1.channel_params.pop("Na1_gNa")
    na2 = Na("Na2")
    na2.channel_params["shared_gNa"] = na2.channel_params.pop("Na2_gNa")

    cell = jx.Cell()
    cell.insert(na1)
    cell.insert(na2)
    cell.delete(na1)

    # This fails if `shared_gNa` is no longer a column of `.nodes`.
    _ = cell.nodes["shared_gNa"]


@pytest.mark.parametrize("solver", ["fwd_euler", "bwd_euler", "exp_euler"])
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


@pytest.mark.parametrize("solver", ["fwd_euler", "bwd_euler", "exp_euler"])
def test_izhikevich(solver, golden):
    cell = jx.Cell()
    cell.insert(Izhikevich())
    cell.record("v")

    dt = 0.1
    t_max = 200.0

    cell.stimulate(jx.step_current(10.0, 180.0, 0.012, dt, t_max))
    v = jx.integrate(cell, delta_t=dt, solver=solver)

    golden.check({"v": np.asarray(v[0, ::200])}, atol=1e-8)


def test_rate_channel(golden):
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

    golden.check({"v": np.asarray(v[0])}, atol=1e-8)


def test_multicompartment_lif():
    """Test that LIF can be run in multicompartment simulation."""
    cell = build.cell(nbranches=2, ncomp=4)
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


def test_multicompartment_izhikevich():
    """Test that Izhikevich can be run in multicompartment simulation."""
    cell = build.cell(nbranches=2, ncomp=4)
    # Leak everywhere, Izhikevich only in "soma".
    cell.insert(Leak())
    cell.branch(0).comp(0).insert(Izhikevich())
    cell.record("v")
    cell.branch(0).comp(0).record("Izhikevich_u")

    dt = 0.1
    t_max = 40.0

    cell.branch(0).comp(0).stimulate(jx.step_current(5.0, 20.0, 0.02, dt, t_max))
    recordings = jx.integrate(cell, delta_t=dt)
    assert np.invert(np.any(np.isnan(recordings)))
    # The recovery variable `u` only changes if the Izhikevich dynamics run.
    assert np.ptp(recordings[-1]) > 0.0


def test_adex():
    """Test that AdEx produces expected spiking behavior."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore surrogate gradient warning
        cell = jx.Cell()
        cell.insert(AdEx())

    # Set parameters for reliable spiking (from test_adex.py)
    cell.set("capacitance", 200.0)
    cell.set("radius", 33.0)
    cell.set("length", 100.0)
    cell.set("v", -58.0)
    cell.record("v")
    cell.record("AdEx_spikes")

    dt = 0.1
    t_max = 100.0

    cell.stimulate(jx.step_current(5.0, 80.0, 100.0, dt, t_max))
    recordings = jx.integrate(cell, delta_t=dt)

    # Check that simulation runs without NaN
    assert np.invert(np.any(np.isnan(recordings)))

    # Check that spikes occurred (voltage should have been reset at some point)
    v = np.asarray(recordings[0]).flatten()
    spikes = np.asarray(recordings[1]).flatten()
    assert np.any(spikes > 0), "AdEx should produce spikes with this stimulation"


def test_multicompartment_adex():
    """Test that AdEx can be run in multicompartment simulation."""
    import warnings

    cell = build.cell(nbranches=2, ncomp=4)
    # Insert AdEx only in "soma" compartment, Leak elsewhere
    cell.insert(Leak())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore surrogate gradient warning
        cell.branch(0).comp(0).insert(AdEx())

    cell.set("capacitance", 200.0)
    cell.record("v")
    cell.branch(0).comp(0).record("AdEx_spikes")

    dt = 0.1
    t_max = 40.0

    cell.branch(0).comp(0).stimulate(jx.step_current(5.0, 20.0, 100.0, dt, t_max))
    recordings = jx.integrate(cell, delta_t=dt)
    assert np.invert(np.any(np.isnan(recordings)))


def test_init_params():
    cell = jx.Cell()
    cell.insert(HH())
    cell.set("temperature", 27.0)  # Reduce temperature by 10 degrees.
    cell.init_params()
    assert np.all(np.abs(cell.nodes["HH_tadj"] - 1 / 2.3) < 1e-8)
