import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jaxley as jx
from jaxley.channels import Channel, K, Na


def test_channel_get_name():
    # default name is the class name
    assert Na().get_name() == "Na"
    # default name can be changed
    assert K(channel_name="KPospischil").get_name() == "KPospischil"


def test_channel_change_name():
    channel_na = Na()
    channel_na.change_name("NaPospischil")
    assert channel_na.get_name() == "NaPospischil"
