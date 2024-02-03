import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH, CaL, CaT, K, Km, Leak, Na


def test_channel_set_name():
    # default name is the class name
    assert Na().name == "Na"

    # channel name can be set in the constructor
    na = Na(name="NaPospischil")
    assert na.name == "NaPospischil"
    assert "NaPospischil_gNa" in na.channel_params.keys()
    assert "NaPospischil_eNa" in na.channel_params.keys()
    assert "NaPospischil_h" in na.channel_states.keys()
    assert "NaPospischil_m" in na.channel_states.keys()
    assert "NaPospischil_vt" not in na.channel_params.keys()
    assert "vt" in na.channel_params.keys()

    # channel name can not be changed directly
    k = K()
    with pytest.raises(AttributeError):
        k.name = "KPospischil"
    assert "KPospischil_gNa" not in k.channel_params.keys()
    assert "KPospischil_eNa" not in k.channel_params.keys()
    assert "KPospischil_h" not in k.channel_states.keys()
    assert "KPospischil_m" not in k.channel_states.keys()


def test_channel_change_name():
    na = Na()
    # channel name can be changed with change_name method
    # (and only this way after initialization)
    na.change_name("NaPospischil")
    assert na.name == "NaPospischil"
    assert "NaPospischil_gNa" in na.channel_params.keys()
    assert "NaPospischil_eNa" in na.channel_params.keys()
    assert "NaPospischil_h" in na.channel_states.keys()
    assert "NaPospischil_m" in na.channel_states.keys()
    assert "NaPospischil_vt" not in na.channel_params.keys()
    assert "vt" in na.channel_params.keys()


def test_integration_with_renamed_channels():
    neuron_hh = HH()
    neuron_hh.change_name("NeuronHH")
    standard_hh = HH()

    comp = jx.Compartment()
    branch = jx.Branch(comp, nseg=4)

    branch.comp(0.0).insert(standard_hh)
    branch.insert(neuron_hh)

    branch.comp(1.0).record()
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
    cell.branch(0).comp(0.0).record()

    cell.branch(0).insert(Na())
    cell.branch(1).insert(K())
    cell.branch(1).comp(0.0).insert(Km())
    cell.branch(0).comp(1.0).insert(CaT())
    cell.insert(CaL())
    cell.insert(Leak())

    cell.insert(HH())

    cell.set("voltages", -62.0)  # At -70.0 there is a rebound spike.
    cell.init_states()
    v = jx.integrate(cell, t_max=20.0)

    last_voltage = v[0, -1]
    cell.set("voltages", last_voltage)
    cell.init_states()

    v = jx.integrate(cell, t_max=10.0)
    assert np.abs(v[0, 0] - v[0, -1]) < 0.02
