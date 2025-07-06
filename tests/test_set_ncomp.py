# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import HH


@pytest.mark.parametrize(
    "property", ["radius", "capacitance", "length", "axial_resistivity"]
)
def test_raise_for_heterogenous_modules(SimpleBranch, property):
    branch0 = SimpleBranch(4)
    branch1 = SimpleBranch(4)
    branch1.comp(1).set(property, 1.5)
    cell = jx.Cell([branch0, branch1], parents=[-1, 0])
    with pytest.raises(ValueError):
        cell.branch(1).set_ncomp(2)


def test_raise_for_heterogenous_channel_existence(SimpleBranch):
    branch0 = SimpleBranch(4)
    branch1 = SimpleBranch(4)
    branch1.comp(2).insert(HH())
    cell = jx.Cell([branch0, branch1], parents=[-1, 0])
    with pytest.raises(ValueError):
        cell.branch(1).set_ncomp(2)


def test_raise_for_heterogenous_channel_properties(SimpleBranch):
    branch0 = SimpleBranch(4)
    branch1 = SimpleBranch(4)
    branch1.insert(HH())
    branch1.comp(3).set("HH_gNa", 0.5)
    cell = jx.Cell([branch0, branch1], parents=[-1, 0])
    with pytest.raises(ValueError):
        cell.branch(1).set_ncomp(2)


def test_raise_for_entire_cells(SimpleCell):
    cell = SimpleCell(3, 4)
    with pytest.raises(AssertionError):
        cell.set_ncomp(2)


def test_raise_for_networks(SimpleCell):
    cell1 = SimpleCell(3, 4)
    cell2 = SimpleCell(3, 4)
    net = jx.Network([cell1, cell2])
    with pytest.raises(AssertionError):
        net.cell(0).branch(1).set_ncomp(2)


def test_raise_for_recording(SimpleCell):
    cell = SimpleCell(3, 2)
    cell.branch(0).comp(0).record()
    with pytest.raises(AssertionError):
        cell.branch(1).set_ncomp(2)


def test_raise_for_stimulus(SimpleCell):
    cell = SimpleCell(3, 2)
    cell.branch(0).comp(0).stimulate(0.4 * jnp.ones(100))
    with pytest.raises(AssertionError):
        cell.branch(1).set_ncomp(2)


@pytest.mark.parametrize("new_ncomp", [1, 2, 4, 5, 8])
def test_simulation_accuracy_api_equivalence_init_vs_setncomp_branch(
    SimpleBranch, new_ncomp
):
    """Test whether a module built from scratch matches module built with `set_ncomp()`.

    This makes one branch, whose `ncomp` is not modified, heterogenous.
    """
    branch1 = SimpleBranch(new_ncomp)

    # The second branch is originally instantiated to have 4 ncomp, but is later
    # modified to have `new_ncomp` compartments.
    branch2 = SimpleBranch(4)
    branch2.comp("all").set("length", 10.0)
    total_branch_len = 4 * 10.0

    # Make the total branch length 40 um.
    branch1.comp("all").set("length", total_branch_len / new_ncomp)

    # Adapt ncomp.
    branch2.set_ncomp(new_ncomp)

    for branch in [branch1, branch2]:
        branch.comp(0).stimulate(0.4 * jnp.ones(100))
        branch.comp(new_ncomp - 1).record()

    v1 = jx.integrate(branch1)
    v2 = jx.integrate(branch2)
    max_error = np.max(np.abs(v1 - v2))
    assert max_error < 1e-8, f"Too large voltage deviation, {max_error} > 1e-8"


@pytest.mark.parametrize("new_ncomp", [1, 2, 4, 5, 8])
def test_simulation_accuracy_api_equivalence_init_vs_setncomp_cell(
    SimpleBranch, new_ncomp
):
    """Test whether a module built from scratch matches module built with `set_ncomp()`."""
    branch1 = SimpleBranch(new_ncomp)

    # The second branch is originally instantiated to have 4 ncomp, but is later
    # modified to have `new_ncomp` compartments.
    branch2 = SimpleBranch(4)
    branch2.comp("all").set("length", 10.0)
    total_branch_len = 4 * 10.0

    # Make the total branch length 20 um.
    branch1.comp("all").set("length", total_branch_len / new_ncomp)
    cell1 = jx.Cell(branch1, parents=[-1, 0])
    cell2 = jx.Cell(branch2, parents=[-1, 0])

    # Adapt ncomp.
    for b in range(2):
        cell2.branch(b).set_ncomp(new_ncomp)

    for cell in [cell1, cell2]:
        cell.branch(0).comp(0).stimulate(0.4 * jnp.ones(100))
        cell.branch(1).comp(new_ncomp - 1).record()

    v1 = jx.integrate(cell1)
    v2 = jx.integrate(cell2)
    max_error = np.max(np.abs(v1 - v2))
    assert max_error < 1e-8, f"Too large voltage deviation, {max_error} > 1e-8"


@pytest.mark.parametrize("new_ncomp", [1, 4, 5])
@pytest.mark.parametrize(
    "file", ["morph_ca1_n120_250.swc", "morph_3_types_single_point_soma.swc"]
)
def test_api_equivalence_swc_lengths_and_radiuses(SimpleMorphCell, new_ncomp, file):
    """Test if the radiuses and lengths of an SWC morph are reconstructed correctly."""
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    cell1 = SimpleMorphCell(fname, ncomp=new_ncomp)
    cell2 = SimpleMorphCell(fname, ncomp=1)

    for b in range(cell2.total_nbranches):
        cell2.branch(b).set_ncomp(new_ncomp)

    for property_name in ["radius", "length"]:
        cell1_vals = cell1.nodes[property_name].to_numpy()
        cell2_vals = cell2.nodes[property_name].to_numpy()
        assert np.allclose(
            cell1_vals, cell2_vals
        ), f"Too large difference in {property_name}"


@pytest.mark.parametrize("new_ncomp", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("file", ["morph_ca1_n120_250.swc"])
def test_simulation_accuracy_swc_init_vs_set_ncomp(SimpleMorphCell, new_ncomp, file):
    """Test whether an SWC initially built with 4 ncomp works after `set_ncomp()`."""
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    cell1 = SimpleMorphCell(fname, ncomp=new_ncomp)
    cell2 = SimpleMorphCell(fname, ncomp=1)

    for b in range(cell2.total_nbranches):
        cell2.branch(b).set_ncomp(new_ncomp)

    for cell in [cell1, cell2]:
        cell.branch(0).comp(0).stimulate(0.4 * jnp.ones(100))
        cell.branch(0).comp(new_ncomp - 1).record()
        cell.branch(3).comp(0).record()
        cell.branch(5).comp(new_ncomp - 1).record()

    v1 = jx.integrate(cell1, voltage_solver="jax.sparse")
    v2 = jx.integrate(cell2, voltage_solver="jax.sparse")
    max_error = np.max(np.abs(v1 - v2))
    assert max_error < 1e-8, f"Too large voltage deviation, {max_error} > 1e-8"


def test_set_ncomp_raises_for_inhomogeneous(SimpleCell):
    """Test whether groups get updated appropriately after `set_ncomp`."""
    cell = SimpleCell(3, 4)
    cell.branch(0).comp(0).add_to_group("exc")
    with pytest.raises(AssertionError):
        cell.branch(0).set_ncomp(2)


def test_dlambda_rule(SimpleMorphCell):
    """Run `set_ncomp` on every branch with the d-lambda rule.

    If you have to modify this test, make sure to also modify the how-to guide."""
    file = "morph_ca1_n120_250.swc"
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "swc_files", file)

    cell = SimpleMorphCell(fname, ncomp=1)

    # Reasonable default values for most models.
    frequency = 100.0
    d_lambda = 0.1  # Larger -> more coarse-grained.

    for branch in cell.branches:
        diameter = 2 * branch.nodes["radius"].to_numpy()[0]
        c_m = branch.nodes["capacitance"].to_numpy()[0]
        r_a = branch.nodes["axial_resistivity"].to_numpy()[0]
        l = branch.nodes["length"].to_numpy()[0]

        lambda_f = 1e5 * np.sqrt(diameter / (4 * np.pi * frequency * c_m * r_a))
        ncomp = int((l / (d_lambda * lambda_f) + 0.9) / 2) * 2 + 1
        branch.set_ncomp(ncomp, initialize=False)

    # After the loop, you have to run `cell.initialize()` because we passed
    # `set_ncomp(..., initialize=False)` for speeding up the loop over branches.
    cell.initialize()
    cell[0, 0].record()
    v = jx.integrate(cell, t_max=1.0)
    assert np.invert(np.any(np.isnan(v))), "Found NaN after d_lambda rule."
