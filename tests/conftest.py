# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os
from copy import deepcopy

import pytest

import jaxley as jx
from jaxley.synapses import IonotropicSynapse


@pytest.fixture(scope="session")
def step_current():
    def get_step_current(
        i_delay: float = 0.5,
        i_dur: float = 1.0,
        i_amp: float = 0.1,
        dt: float = 0.025,
        t_max: float = 3.0,
        i_offset: float = 0.0,
    ):
        """Create a step current stimulus."""
        return jx.step_current(i_delay, i_dur, i_amp, dt, t_max, i_offset)

    yield get_step_current


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
        nseg: int, copy: bool = True, force_init: bool = False
    ) -> jx.Branch:
        """Create or retrieve a branch.

        If a branch with the same number of compartments already exists, it is returned.

        Args:
            nseg: Number of compartments in the branch.
            copy: Whether to return a copy of the branch. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Branch()."""
        if nseg not in branches or force_init:
            comp = SimpleComp(force_init=force_init)
            branches[nseg] = jx.Branch([comp] * nseg)
        return deepcopy(branches[nseg]) if copy and not force_init else branches[nseg]

    yield get_or_build_branch
    branches = {}


@pytest.fixture(scope="session")
def SimpleCell(SimpleBranch):
    """Fixture for creating or retrieving an already created cell."""
    cells = {}

    def get_or_build_cell(
        nbranches: int, nseg: int, copy: bool = True, force_init: bool = False
    ) -> jx.Cell:
        """Create or retrieve a cell.

        If a cell with the same number of branches and compartments already exists, it
        is returned. The branch strcuture is assumed as [-1, 0, 0, 1, 1, 2, 2, ...].

        Args:
            nbranches: Number of branches in the cell.
            nseg: Number of compartments in each branch.
            copy: Whether to return a copy of the cell. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Cell()."""
        if key := (nbranches, nseg) not in cells or force_init:
            parents = [-1]
            depth = 0
            while nbranches > len(parents):
                parents = [-1] + [b // 2 for b in range(0, 2**depth - 2)]
                depth += 1
            parents = parents[:nbranches]
            branch = SimpleBranch(nseg=nseg, force_init=force_init)
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
        nseg: int,
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
            nseg: Number of compartments in each branch.
            connect: Whether to connect the first two cells in the network.
            copy: Whether to return a copy of the network. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Network()."""
        if key := (ncells, nbranches, nseg, connect) not in nets or force_init:
            net = jx.Network(
                [SimpleCell(nbranches=nbranches, nseg=nseg, force_init=force_init)]
                * ncells
            )
            if connect:
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
        fname: str = None,
        nseg: int = 1,
        max_branch_len: float = 2_000.0,
        copy: bool = True,
        force_init: bool = False,
    ) -> jx.Cell:
        """Create or retrieve a cell from an SWC file.

        If a cell with the same SWC file, number of compartments, and maximum branch
        length already exists, it is returned.

        Args:
            fname: Path to the SWC file.
            nseg: Number of compartments in each branch.
            max_branch_len: Maximum length of a branch.
            copy: Whether to return a copy of the cell. Default is True.
            force_init: Force the init from scratch. Default is False.

        Returns:
            jx.Cell()."""
        dirname = os.path.dirname(__file__)
        default_fname = os.path.join(dirname, "swc_files", "morph.swc")
        fname = default_fname if fname is None else fname
        if key := (fname, nseg, max_branch_len) not in cells or force_init:
            cells[key] = jx.read_swc(fname, nseg, max_branch_len, assign_groups=True)
        return deepcopy(cells[key]) if copy and not force_init else cells[key]

    yield get_or_build_cell
    cells = {}


@pytest.fixture(scope="session")
def swc2jaxley():
    """Fixture for creating or retrieving an already computed params of a morphology."""

    params = {}

    def get_or_compute_swc2jaxley_params(
        fname: str = None,
        max_branch_len: float = 2_000.0,
        sort: bool = True,
        force_init: bool = False,
    ):
        dirname = os.path.dirname(__file__)
        default_fname = os.path.join(dirname, "swc_files", "morph.swc")
        fname = default_fname if fname is None else fname
        if key := (fname, max_branch_len, sort) not in params or force_init:
            params[key] = jx.utils.swc.swc_to_jaxley(fname, max_branch_len, sort)
        return params[key]

    yield get_or_compute_swc2jaxley_params
    params = {}
