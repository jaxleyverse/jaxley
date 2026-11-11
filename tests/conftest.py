# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
import sys
import warnings
from copy import deepcopy
from typing import Optional

import pytest

import jaxley as jx
from jaxley.channels import AdEx, Fire, Leak
from jaxley.synapses import IonotropicSynapse, SpikeSynapse


@pytest.fixture(scope="session")
def SimpleComp():
    """Fixture for creating or retrieving an already created compartment."""
    comps = {}

    def get_or_build_comp(
        copy: bool = True, force_init: bool = False
    ) -> jx.Compartment:
        """Create or retrieve a compartment.

        Args:
            copy: Whether to return a copy of the compartment. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Compartment()."""
        if "comp" not in comps or force_init:
            comps["comp"] = jx.Compartment()
        return deepcopy(comps["comp"]) if copy and not force_init else comps["comp"]

    yield get_or_build_comp
    comps = {}


@pytest.fixture(scope="session")
def SimpleBranch(SimpleComp):
    """Fixture for creating or retrieving an already created branch."""
    branches = {}

    def get_or_build_branch(
        ncomp: int, copy: bool = True, force_init: bool = False
    ) -> jx.Branch:
        """Create or retrieve a branch.

        If a branch with the same number of compartments already exists, it is returned.

        Args:
            ncomp: Number of compartments in the branch.
            copy: Whether to return a copy of the branch. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Branch()."""
        if ncomp not in branches or force_init:
            comp = SimpleComp(force_init=force_init)
            branches[ncomp] = jx.Branch([comp] * ncomp)
        return deepcopy(branches[ncomp]) if copy and not force_init else branches[ncomp]

    yield get_or_build_branch
    branches = {}


@pytest.fixture(scope="session")
def SimpleCell(SimpleBranch):
    """Fixture for creating or retrieving an already created cell."""
    cells = {}

    def get_or_build_cell(
        nbranches: int, ncomp: int, copy: bool = True, force_init: bool = False
    ) -> jx.Cell:
        """Create or retrieve a cell.

        If a cell with the same number of branches and compartments already exists, it
        is returned. The branch structure is assumed as [-1, 0, 0, 1, 1, 2, 2, ...].

        Args:
            nbranches: Number of branches in the cell.
            ncomp: Number of compartments in each branch.
            copy: Whether to return a copy of the cell. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Cell()."""
        if key := (nbranches, ncomp) not in cells or force_init:
            parents = [-1]
            depth = 0
            while nbranches > len(parents):
                parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
                depth += 1
            parents = parents[:nbranches]
            branch = SimpleBranch(ncomp=ncomp, force_init=force_init)
            cells[key] = jx.Cell([branch] * nbranches, parents)
        return deepcopy(cells[key]) if copy and not force_init else cells[key]

    yield get_or_build_cell
    cells = {}


@pytest.fixture(scope="session")
def SimpleNet(SimpleCell):
    """Fixture for creating or retrieving an already created network."""
    nets = {}

    def get_or_build_net(
        ncells: int,
        nbranches: int,
        ncomp: int,
        connect: bool = False,
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Network:
        """Create or retrieve a network.

        If a network with the same number of cells, branches, compartments, and
        connections already exists, it is returned.

        Args:
            ncells: Number of cells in the network.
            nbranches: Number of branches in each cell.
            ncomp: Number of compartments in each branch.
            connect: Whether to connect the first two cells in the network.
            copy: Whether to return a copy of the network. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Network()."""
        if key := (ncells, nbranches, ncomp, connect) not in nets or force_init:
            net = jx.Network(
                [SimpleCell(nbranches=nbranches, ncomp=ncomp, force_init=force_init)]
                * ncells
            )
            if connect:
                jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
            nets[key] = net
        return deepcopy(nets[key]) if copy and not force_init else nets[key]

    yield get_or_build_net
    nets = {}


@pytest.fixture(scope="session")
def AdExNet(SimpleComp):
    """Fixture for creating a network with AdEx spiking neurons."""
    nets = {}

    def get_or_build_net(
        ncells: int,
        nbranches: int,
        ncomp: int,
        connect: bool = False,
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Network:
        """Create or retrieve a spiking network with AdEx neurons.

        Args:
            ncells: Number of cells in the network.
            nbranches: Number of branches in each cell.
            ncomp: Number of compartments in each branch.
            connect: Whether to connect cells with synapses.
            copy: Whether to return a copy of the network. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Network() with AdEx neurons."""
        if key := (ncells, nbranches, ncomp, connect) not in nets or force_init:
            # Build cells with AdEx
            cells = []
            for _ in range(ncells):
                # Simple branch structure
                parents = [-1]
                depth = 0
                while nbranches > len(parents):
                    parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
                    depth += 1
                parents = parents[:nbranches]

                comp = SimpleComp(force_init=force_init)
                branch = jx.Branch([comp] * ncomp)
                cell = jx.Cell([branch] * nbranches, parents)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Ignore surrogate warning
                    cell.insert(AdEx())
                cell.set("capacitance", 200.0)
                cells.append(cell)

            net = jx.Network(cells)
            if connect and ncells >= 2:
                jx.connect(net[0, 0, 0], net[1, 0, 0], IonotropicSynapse())
            nets[key] = net
        return deepcopy(nets[key]) if copy and not force_init else nets[key]

    yield get_or_build_net
    nets = {}


@pytest.fixture(scope="session")
def SimpleMorphCell():
    """Fixture for creating or retrieving an already created morpholgy."""

    cells = {}

    def get_or_build_cell(
        fname: Optional[str] = None,
        ncomp: int = 1,
        max_branch_len: float = 2_000.0,
        ignore_swc_tracing_interruptions: bool = True,
        swc_backend: str = "graph",
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Cell:
        """Create or retrieve a cell from an SWC file.

        If a cell with the same SWC file, number of compartments, and maximum branch
        length already exists, it is returned.

        Args:
            fname: Path to the SWC file.
            ncomp: Number of compartments in each branch.
            max_branch_len: Maximum length of a branch.
            copy: Whether to return a copy of the cell. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Cell()."""
        dirname = os.path.dirname(__file__)
        default_fname = os.path.join(dirname, "swc_files", "morph_ca1_n120.swc")
        fname = default_fname if fname is None else fname
        if key := (fname, ncomp, max_branch_len) not in cells or force_init:
            cells[key] = jx.read_swc(
                fname,
                ncomp=ncomp,
                max_branch_len=max_branch_len,
                assign_groups=True,
                backend=swc_backend,
                ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
            )
        return deepcopy(cells[key]) if copy and not force_init else cells[key]

    yield get_or_build_cell
    cells = {}


@pytest.fixture(scope="session")
def SpikeNet(SimpleCell):
    """Fixture for creating or retrieving a Network with SpikeSynapses."""
    spike_nets = {}

    def get_or_build_net(
        ncells: int,
        nbranches: int,
        ncomp: int,
        connect: bool = False,
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Network:
        """Create or retrieve a network.

        If a network with the same number of cells, branches, compartments, and
        connections already exists, it is returned.

        Args:
            ncells: Number of cells in the network.
            nbranches: Number of branches in each cell.
            ncomp: Number of compartments in each branch.
            connect: Whether to connect the first two cells in the network.
            copy: Whether to return a copy of the network. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Network()."""
        if key := (ncells, nbranches, ncomp, connect) not in spike_nets or force_init:
            net = jx.Network(
                [SimpleCell(nbranches=nbranches, ncomp=ncomp, force_init=force_init)]
                * ncells
            )
            net.select(nodes="all").insert(Fire())
            net.select(nodes="all").insert(Leak())
            if connect:
                half = ncells // 2
                jx.fully_connect(
                    net.cell(range(half)), net.cell(range(half, ncells)), SpikeSynapse()
                )
            spike_nets[key] = net
        return deepcopy(spike_nets[key]) if copy and not force_init else spike_nets[key]

    yield get_or_build_net
    spike_nets = {}


def pytest_terminal_summary(terminalreporter):
    """Print the timings of `pytest -m perf` (see `tests/test_perf.py`)."""
    perf = sys.modules.get("tests.test_perf")
    if perf is not None and perf.RESULTS:
        backend = perf.jax.default_backend()
        table = perf.format_table(perf.RESULTS, f"perf [s] ({backend})")
        terminalreporter.write_line("\n" + table)
