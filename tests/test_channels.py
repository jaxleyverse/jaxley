import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import pytest

import jaxley as jx
from jaxley.channels import K, Na


def test_channel_set_name():
    # default name is the class name
    assert Na().channel_name == "Na"

    # channel name can be set in the constructor
    na = Na(channel_name="NaPospischil")
    assert na.channel_name == "NaPospischil"
    assert "NaPospischil_gNa" in na.channel_params.keys()
    assert "NaPospischil_eNa" in na.channel_params.keys()
    assert "NaPospischil_h" in na.channel_states.keys()
    assert "NaPospischil_m" in na.channel_states.keys()
    assert "NaPospischil_vt" not in na.channel_params.keys()
    assert "vt" in na.channel_params.keys()

    # channel name can not be changed directly
    k = K()
    with pytest.raises(AttributeError):
        k.channel_name = "KPospischil"
    assert "KPospischil_gNa" not in k.channel_params.keys()
    assert "KPospischil_eNa" not in k.channel_params.keys()
    assert "KPospischil_h" not in k.channel_states.keys()
    assert "KPospischil_m" not in k.channel_states.keys()


def test_channel_change_name():
    na = Na()
    # channel name can be changed with change_name method
    # (and only this way after initialization)
    na.change_name("NaPospischil")
    assert na.channel_name == "NaPospischil"
    assert "NaPospischil_gNa" in na.channel_params.keys()
    assert "NaPospischil_eNa" in na.channel_params.keys()
    assert "NaPospischil_h" in na.channel_states.keys()
    assert "NaPospischil_m" in na.channel_states.keys()
    assert "NaPospischil_vt" not in na.channel_params.keys()
    assert "vt" in na.channel_params.keys()
