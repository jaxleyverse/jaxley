# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Dict, Optional

import jax
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from jaxley_mech.channels.l5pc import CaHVA, CaLVA
from jaxley_mech.channels.l5pc import CaPump as CaPumpAsChannel

import jaxley as jx
from jaxley.channels import HH, K, Leak, Na
from jaxley.channels.channel import Channel
from jaxley.pumps import CaFaradayConcentrationChange, CaNernstReversal, CaPump, Pump


class NaPump(Pump):
    """A dummy pump which is only used for internal `Jaxley` tests.

    Modeled after the calcium channel in Destexhe et al. 1994.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered).
            f"{self._name}_decay": 80,  # Buffering time constant in ms.
            f"{self._name}_depth": 0.1,  # Depth of shell in um.
            f"{self._name}_minNai": 1e-4,  # Minimum intracell. ca concentration in mM.
        }
        self.channel_states = {"i_Na": 1e-8, "NaCon_i": 5e-05}
        self.ion_name = "NaCon_i"
        self.current_name = "i_NaPump"
        self.META = {
            "reference": "Made up by Jaxley developers (@michaeldeistler)",
            "mechanism": "Sodium dynamics",
        }

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update states if necessary (but this pump has no states to update)."""
        return {"NaCon_i": states["NaCon_i"], "i_Na": states["i_Na"]}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        modified_state,
        params: Dict[str, jnp.ndarray],
    ):
        """Return change of calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = states["i_Na"] / 1_000.0
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minNai"]

        FARADAY = 96485  # Coulombs per mole.

        # Calculate the contribution of calcium currents to cai change.
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)
        state_decay = (modified_state - minCai) / decay
        diff = drive_channel - state_decay
        return -diff

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}


class NaNernstReversal(Channel):
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
        self.channel_states = {"eNa": 0.0, "NaCon_i": 5e-05, "NaCon_e": 3.0}
        self.current_name = f"i_Na"
        self.META = {"ion": "Na"}

    def update_states(self, u, dt, voltages, params):
        """Update internal calcium concentration based on calcium current and decay."""
        R, T, F = (
            self.channel_constants["R"],
            self.channel_constants["T"],
            self.channel_constants["F"],
        )
        Nai = u["NaCon_i"]
        Nao = u["NaCon_e"]
        C = R * T / (2 * F) * 1000  # mV
        vNa = C * jnp.log(Nao / Nai)
        return {"eNa": vNa, "NaCon_i": Nai, "NaCon_e": Nao}

    def compute_current(self, u, voltages, params):
        """This dynamics model does not directly contribute to the membrane current."""
        return 0

    def init_state(self, states, voltages, params, delta_t):
        """Initialize the state at fixed point of gate dynamics."""
        return {}


class CaPump2(Pump):
    """A copy of the Calcium dynamics tracking inside calcium concentration.

    Modeled after Destexhe et al. 1994.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered).
            f"{self._name}_decay": 80,  # Buffering time constant in ms.
            f"{self._name}_depth": 0.1,  # Depth of shell in um.
            f"{self._name}_minCai": 1e-4,  # Minimum intracell. ca concentration in mM.
        }
        self.channel_states = {"i_Ca": 1e-8, "CaCon_i": 5e-05}
        self.ion_name = "CaCon_i"
        self.current_name = "i_CaPump"
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update states if necessary (but this pump has no states to update)."""
        return {"CaCon_i": states["CaCon_i"], "i_Ca": states["i_Ca"]}

    def compute_current(
        self,
        states: Dict[str, jnp.ndarray],
        modified_state,
        params: Dict[str, jnp.ndarray],
    ):
        """Return change of calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = states["i_Ca"] / 1_000.0
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole.

        # Calculate the contribution of calcium currents to cai change.
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)
        state_decay = (modified_state - minCai) / decay
        diff = drive_channel - state_decay
        return -diff

    def init_state(
        self,
        states: Dict[str, jnp.ndarray],
        v: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        delta_t: float,
    ):
        """Initialize states of channel."""
        return {}


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
    cell.insert(CaPump2())
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
    cell.insert(CaPump2())
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
    cell.insert(CaPump2())
    cell.diffuse("CaCon_i")
    cell.set("CaHVA_gCaHVA", 0.0001)
    cell.set("axial_diffusion_CaCon_i", 200.0)
    cell.set("axial_diffusion_NaCon_i", 400.0)
    cell.init_states()
    v3 = jx.integrate(cell, voltage_solver="jaxley.dhs.cpu")

    ############################ Option 4:
    cell = _build_active_cell()
    cell.insert(CaPump2())
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


def test_ion_is_pumped_but_not_diffused():
    """Test whether there is no NaN when an ion is pumped but not diffused."""
    cell = _build_calcium_cell()
    cacon_i = jx.integrate(cell, t_max=5.0, voltage_solver="jaxley.dhs.cpu")
    assert np.invert(np.any(np.isnan(cacon_i))), "Found NaN when ion is not diffused!"


def test_ion_diffusion_compartment():
    """Test whether `diffuse()` on a compartment does not lead to NaN."""
    comp = jx.Compartment()
    comp.insert(CaHVA())
    comp.insert(CaNernstReversal())
    comp.diffuse("CaCon_i")
    comp.set("axial_diffusion_CaCon_i", 1.0)
    comp.record("v")
    comp.record("CaCon_i")
    recs = jx.integrate(comp, t_max=10.0, voltage_solver="jaxley.dhs.cpu")
    assert np.invert(np.any(np.isnan(recs)))


def test_ion_diffusion_branch():
    """Test whether `diffuse()` on a branch does not lead to NaN.

    By the side, this tests the `fwd_euler` solver together with ion diffusion.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    branch.insert(CaHVA())
    branch.insert(CaNernstReversal())
    branch.diffuse("CaCon_i")
    branch.set("axial_diffusion_CaCon_i", 1.0)
    branch.record("v")
    branch.record("CaCon_i")
    recs = jx.integrate(branch, t_max=10.0, solver="fwd_euler")
    assert np.invert(np.any(np.isnan(recs)))


def test_ion_diffusion_cell():
    """Test whether `diffuse()` on a cell does not lead to NaN."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell = jx.Cell(branch, parents=[-1, 0, 0])
    cell.insert(CaHVA())
    cell.insert(CaNernstReversal())
    cell.diffuse("CaCon_i")
    cell.set("axial_diffusion_CaCon_i", 1.0)
    cell.record("v")
    cell.record("CaCon_i")
    recs = jx.integrate(cell, t_max=5.0, voltage_solver="jaxley.dhs.cpu")
    assert np.invert(np.any(np.isnan(recs)))


def test_ion_diffusion_net():
    """Test whether `diffuse()` on a network does not lead to NaN."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=2)
    cell = jx.Cell(branch, parents=[-1, 0])
    net = jx.Network([cell, cell])
    net.insert(CaHVA())
    net.insert(CaNernstReversal())
    net.diffuse("CaCon_i")
    net.set("axial_diffusion_CaCon_i", 1.0)
    net.record("v")
    net.record("CaCon_i")
    recs = jx.integrate(net, t_max=5.0, voltage_solver="jaxley.dhs.cpu")
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
