import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import jaxley as jx
from jaxley.channels import HHChannel
from jaxley.synapses import GlutamateSynapse


def test_make_trainable():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)
    num_branches = len(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = jx.Cell([branch for _ in range(num_branches)], parents=parents).initialize()
    cell.insert(HHChannel())

    cell.branch(0).comp(0.0).set_params("length", 12.0)
    cell.branch(1).comp(1.0).set_params("gNa", 0.2)
    assert cell.num_trainable_params == 2

    cell.branch([0, 1]).make_trainable("radius", 1.0)
    assert cell.num_trainable_params == 4
    cell.branch([0, 1]).make_trainable("length")
    cell.branch([0, 1]).make_trainable("axial_resistivity", [600.0, 700.0])
    cell.branch([0, 1]).make_trainable("gNa")

    cell.get_parameters()


def test_make_trainable_network():
    """Test make_trainable."""
    nseg_per_branch = 8

    depth = 5
    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
    parents = jnp.asarray(parents)
    num_branches = len(parents)

    comp = jx.Compartment().initialize()
    branch = jx.Branch([comp for _ in range(nseg_per_branch)]).initialize()
    cell = jx.Cell([branch for _ in range(num_branches)], parents=parents).initialize()
    cell.insert(HHChannel())

    conns = [
        jx.Connectivity(
            GlutamateSynapse(),
            [
                jx.Connection(0, 0, 0.0, 1, 0, 0.0),
            ],
        )
    ]
    net = jx.Network([cell, cell], conns).initialize()

    cell.branch(0).comp(0.0).set_params("length", 12.0)
    cell.branch(1).comp(1.0).set_params("gNa", 0.2)

    cell.branch([0, 1]).make_trainable("radius", 1.0)
    cell.branch([0, 1]).make_trainable("length")
    cell.branch([0, 1]).make_trainable("axial_resistivity", [600.0, 700.0])
    cell.branch([0, 1]).make_trainable("gNa")

    cell.get_parameters()
    net.GlutamateSynapse.set_params("gS", 0.1)
    assert cell.num_trainable_params == 8  # `set_params()` is ignored.
