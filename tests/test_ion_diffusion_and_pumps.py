# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import numpy as np
import pytest
from jaxley_mech.channels.l5pc import CaHVA
from jaxley_mech.channels.l5pc import CaPump as CaPumpAsChannel
from jaxley_mech.channels.pospischil import K, Leak, Na

import jaxley as jx
from jaxley.channels import HH
from jaxley.pumps import CaFaradayConcentrationChange, CaNernstReversal, CaPump
from tests import builders as build
from tests.mechanisms import NaNernstReversal, NaPump


def _build_active_cell():
    """Helper that builds an active cell with Na, K, Leak separately."""
    current = jx.step_current(0.5, 3.0, 0.1, 0.025, 10.0)
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell = jx.Cell(branch, parents=[-1, 0])
    cell.branch(0).comp(0).stimulate(current)

    # Insert channels.
    cell.insert(Na())
    cell.insert(K())
    cell.insert(Leak())
    cell.insert(CaHVA())
    cell.insert(NaNernstReversal())
    cell.insert(CaNernstReversal())

    # Recordings and simulation.
    cell.branch(1).comp(1).record("v")
    cell.branch(0).comp(1).record("CaCon_i")
    cell.branch(0).comp(1).record("NaCon_i")
    return cell


def test_that_order_of_insert_does_not_matter():
    """Insert pumps in a different orders and check if all give the same simulation."""
    ############################ Option 1:
    cell = _build_active_cell()
    cell.insert(CaPump())
    # Also test `delete` for a pump.
    cell.insert(CaFaradayConcentrationChange())
    cell.delete(CaFaradayConcentrationChange())
    # Return to the channel that should really be inserted.
    cell.insert(NaPump())
    cell.insert(CaPump(name="CaPump2"))
    cell.diffuse("CaCon_i")
    cell.diffuse("NaCon_i")
    cell.set("CaHVA_gCaHVA", 0.0001)
    cell.set("axial_diffusion_CaCon_i", 200.0)
    cell.set("axial_diffusion_NaCon_i", 400.0)
    cell.init_states()
    v1 = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")

    ############################ Option 2:
    cell = _build_active_cell()
    cell.insert(CaPump())
    cell.diffuse("CaCon_i")
    cell.insert(NaPump())
    cell.insert(CaPump(name="CaPump2"))
    cell.diffuse("NaCon_i")
    # Delete and diffuse again to test `delete_diffusion` method.
    cell.delete_diffusion("NaCon_i")
    cell.diffuse("NaCon_i")
    cell.set("CaHVA_gCaHVA", 0.0001)
    cell.set("axial_diffusion_CaCon_i", 200.0)
    cell.set("axial_diffusion_NaCon_i", 400.0)
    cell.init_states()
    v2 = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")

    ############################ Option 3:
    cell = _build_active_cell()
    cell.insert(CaPump())
    cell.insert(NaPump())
    cell.diffuse("NaCon_i")
    cell.insert(CaPump(name="CaPump2"))
    cell.diffuse("CaCon_i")
    cell.set("CaHVA_gCaHVA", 0.0001)
    cell.set("axial_diffusion_CaCon_i", 200.0)
    cell.set("axial_diffusion_NaCon_i", 400.0)
    cell.init_states()
    v3 = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")

    ############################ Option 4:
    cell = _build_active_cell()
    cell.insert(CaPump(name="CaPump2"))
    cell.insert(CaPump())
    cell.insert(NaPump())
    cell.diffuse("CaCon_i")
    cell.diffuse("NaCon_i")
    cell.set("CaHVA_gCaHVA", 0.0001)
    cell.set("axial_diffusion_CaCon_i", 200.0)
    cell.set("axial_diffusion_NaCon_i", 400.0)
    cell.init_states()
    v4 = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")

    for i, v_compare in enumerate([v2, v3, v4]):
        max_error = np.max(np.abs(v1 - v_compare))
        assert max_error < 1e-8, f"Gap between v1 and v{i+1} is {max_error}."


def _build_calcium_cell():
    """Helper that builds a cell with relevant channels."""
    current = jx.step_current(0.5, 3.0, 0.1, 0.025, 5.0)
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell = jx.Cell(branch, parents=[-1, 0])
    cell.branch(0).comp(0).stimulate(current)

    # Insert channels.
    cell.insert(HH())
    cell.insert(CaHVA())
    cell.insert(CaNernstReversal())

    # Recordings and simulation.
    cell.branch(0).comp(1).record("CaCon_i")
    return cell


def test_that_high_axial_diffusion_approaches_no_diffusion():
    """Test if very high axial ion resisitvity approaches no diffusion at all.

    By the side, this test also implicitly tests the `cranck_nicolson` solver in
    combination with `jax.sparse`.
    """
    cell1 = _build_calcium_cell()
    cell1.insert(CaPump())
    cell1.diffuse("CaCon_i")
    # Very high axial resistivity, almost no diffusion.
    cell1.set("axial_diffusion_CaCon_i", 1e-8)
    v1 = jx.integrate(cell1, solver="crank_nicolson", voltage_solver="jaxley.dhs.cpu")

    cell2 = _build_calcium_cell()
    cell2.insert(CaPump())
    v2 = jx.integrate(cell2, solver="crank_nicolson", voltage_solver="jaxley.dhs.cpu")

    max_error = np.max(np.abs(v1 - v2))
    assert max_error < 1e-7, f"Gap is {max_error}."


def test_pump_exists_only_in_parts_of_the_cell():
    """Test that there is no NaN when a pump exists only in parts of a cell.

    Diffusion can still introduce the new molecule to the rest of the cell.

    By the side, this test also implicitly tests the `cranck_nicolson` solver in
    combination with `jaxley.stone` (default).
    """
    current = jx.step_current(0.5, 3.0, 0.1, 0.025, 5.0)
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=4)
    cell = jx.Cell(branch, parents=[-1, 0])
    cell.branch(0).comp(0).stimulate(current)

    # Insert channels.
    cell.insert(HH())

    # Calcium only in branch(1)
    cell.branch(1).insert(CaHVA())
    cell.branch(1).insert(CaNernstReversal())
    cell.branch(1).insert(CaPump())
    cell.diffuse("CaCon_i")

    # Very high axial resistivity, almost no diffusion.
    cell.record("CaCon_i")
    recordings = jx.integrate(cell, solver="crank_nicolson")
    assert np.invert(np.any(np.isnan(recordings))), "Found NaN value!"


def test_diffusion_in_parts_of_cell_raises():
    """Assert the raise if one runs diffusion on parts of a cell."""
    cell = _build_calcium_cell()

    with pytest.raises(AssertionError):
        cell.branch(0).diffuse("CaCon_i")


def test_raise_ion_diffusion_in_one_cell_of_net1():
    """Assert the raise if a network is built of cells that are diffused."""
    cell1 = _build_calcium_cell()
    cell1.diffuse("CaCon_i")
    cell2 = _build_calcium_cell()
    with pytest.raises(AssertionError):
        net = jx.Network([cell1, cell2])


def test_raise_ion_diffusion_in_one_cell_of_net2():
    """Assert the raise if one runs diffusion on parts of a network."""
    cell1 = _build_calcium_cell()
    cell2 = _build_calcium_cell()
    net = jx.Network([cell1, cell2])
    with pytest.raises(AssertionError):
        net.cell(0).diffuse("CaCon_i")


def test_ion_is_diffused_but_not_pumped():
    """Test if, when an ion is diffused but not pumped, comps converge to one value."""
    cell = _build_calcium_cell()
    cell.diffuse("CaCon_i")
    # Initialize the calcium concentration to be different at the beginning.
    cell.branch(0).set("CaCon_i", 0.2)
    cell.branch(1).set("CaCon_i", 0.1)
    cell.set("axial_diffusion_CaCon_i", 1000.0)  # Relatively high diffusion.
    cell.record("CaCon_i")  # Record everywhere.

    # Run simulation and assert that calcium becomes the same everywhere.
    cacon_i = jx.integrate(cell, t_max=5.0)
    last_timepoint_cacon_i = cacon_i[:, -1]
    avg_cacon_i = np.mean(last_timepoint_cacon_i)
    max_diff_to_mean = np.max(np.abs(avg_cacon_i - last_timepoint_cacon_i))
    assert max_diff_to_mean < 1e-4, f"Error {max_diff_to_mean} > 1e-4"


def test_ion_diffusion_conserves_ion_amount():
    """Without currents or pumps, diffusion only redistributes the ion.

    All compartments have the same volume, so the summed concentration is constant.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(CaNernstReversal())  # Defines `CaCon_i`, but no calcium current.
    cell.diffuse("CaCon_i")
    cell.set("axial_diffusion_CaCon_i", 1.0)
    cell.branch(0).set("CaCon_i", 0.2)
    cell.branch(1).set("CaCon_i", 0.1)
    cell.record("CaCon_i")

    cacon_i = jx.integrate(cell, t_max=5.0)
    total = np.sum(cacon_i, axis=0)
    np.testing.assert_allclose(total, total[0], rtol=1e-6)
    # And the ion really did diffuse.
    assert not np.allclose(cacon_i[:, -1], cacon_i[:, 0])


def test_ion_is_pumped_but_not_diffused():
    """Test whether there is no NaN when an ion is pumped but not diffused."""
    cell = _build_calcium_cell()
    cell.insert(CaPump())
    cacon_i = jx.integrate(cell, t_max=5.0, voltage_solver="jaxley.dhs.cpu")
    assert np.invert(np.any(np.isnan(cacon_i))), "Found NaN when ion is not diffused!"

    # The pump must change the calcium concentration compared to no pump.
    unpumped_cell = _build_calcium_cell()
    unpumped = jx.integrate(unpumped_cell, t_max=5.0, voltage_solver="jaxley.dhs.cpu")
    assert not np.allclose(cacon_i, unpumped), "The pump has no effect."


@pytest.mark.parametrize(
    "module_type, kwargs",
    [
        ("comp", dict(t_max=10.0, voltage_solver="jaxley.dhs.cpu")),
        ("branch", dict(t_max=10.0, solver="fwd_euler")),  # Also tests `fwd_euler`.
        ("cell", dict(t_max=5.0, voltage_solver="jaxley.dhs.cpu")),
        ("net", dict(t_max=5.0, voltage_solver="jaxley.dhs.cpu")),
    ],
)
def test_ion_diffusion_runs(module_type, kwargs):
    """`diffuse()` on any kind of module does not lead to NaN."""
    if module_type == "comp":
        module = build.comp()
    elif module_type == "branch":
        module = build.branch(ncomp=2)
    elif module_type == "cell":
        module = build.cell(nbranches=3, ncomp=2)
    else:
        module = build.net(ncells=2, nbranches=2, ncomp=2)
    module.insert(CaHVA())
    module.insert(CaNernstReversal())
    module.diffuse("CaCon_i")
    module.set("axial_diffusion_CaCon_i", 1.0)
    module.record("v")
    module.record("CaCon_i")
    recs = jx.integrate(module, **kwargs)
    assert np.invert(np.any(np.isnan(recs)))


def test_diffuse_a_channel_state():
    """Test whether diffusing a channel state gives almost same results as pump.

    For full numerical accuracy, channel states should not be diffused (but only things)
    that are pumped). However, we still allow to diffuse channel states. This test
    evaluates whether the error of doing this is small enough.
    """
    cell1 = _build_calcium_cell()
    cell2 = _build_calcium_cell()
    cell1.insert(CaPump())
    cell2.insert(CaPumpAsChannel())

    for cell in [cell1, cell2]:
        cell.set("CaCon_i", 0.0001)
        cell.diffuse("CaCon_i")
        cell.set("axial_diffusion_CaCon_i", 1.0)
        cell.record("v")
        cell.record("CaCon_i")
    v1 = jx.integrate(cell1, t_max=10.0, voltage_solver="jaxley.dhs.cpu")
    v2 = jx.integrate(cell2, t_max=10.0, voltage_solver="jaxley.dhs.cpu")
    max_error = np.max(np.abs(v1 - v2))
    assert max_error < 1e-2, f"Error {max_error} > 1e-2"
