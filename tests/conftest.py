# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import json
import os
from copy import deepcopy
from typing import Optional

import pytest

import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from tests.test_regression import generate_regression_report, load_json


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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_collection_modifyitems(config, items):
    NEW_BASELINE = (
        int(os.environ["NEW_BASELINE"]) if "NEW_BASELINE" in os.environ else 0
    )

    dirname = os.path.dirname(__file__)
    baseline_fname = os.path.join(dirname, "regression_test_baselines.json")

    def should_skip_regression():
        return not NEW_BASELINE and not os.path.exists(baseline_fname)

    if should_skip_regression():
        for item in items:
            if "regression" in item.keywords:
                skip_regression = pytest.mark.skip(
                    reason="need NEW_BASELINE env to run"
                )
                item.add_marker(skip_regression)

    # Avoid that functions marked as `additional_neuron_tests` arent tested by default.
    #
    # If we run `pytest tests` then the `additional_neuron_tests` tests are not run.
    # This is useful because some tests based on NEURON (e.g.
    # `jaxley_vs_neuron/test_cell::test_similarity_ion_diffusion`) cause a segmentation
    # error downstream (in later tests). In addition, I have not set up a github action
    # to automatically compile NEURON channel models (which would also cause the tests
    # to fail. For these two reasons, we skip some NEURON tests by default and only run
    # them at release locally.
    #
    # Written by ChatGPT.
    if config.getoption("-m") == "additional_neuron_tests":
        return  # Run normally if explicitly requested

    skip_additional = pytest.mark.skip(
        reason="Skipped unless explicitly enabled with -m 'additional_neuron_tests'"
    )
    for item in items:
        if "additional_neuron_tests" in item.keywords:
            item.add_marker(skip_additional)


@pytest.fixture(scope="session", autouse=True)
def print_session_report(request, pytestconfig):
    """Cleanup a testing directory once we are finished."""
    NEW_BASELINE = (
        int(os.environ["NEW_BASELINE"]) if "NEW_BASELINE" in os.environ else 0
    )

    dirname = os.path.dirname(__file__)
    baseline_fname = os.path.join(dirname, "regression_test_baselines.json")
    results_fname = os.path.join(dirname, "regression_test_results.json")

    collected_regression_tests = [
        item for item in request.session.items if item.get_closest_marker("regression")
    ]

    def update_baseline():
        results = load_json(results_fname)
        with open(baseline_fname, "w") as f:
            json.dump(results, f, indent=2)
        os.remove(results_fname)

    def print_regression_report():
        baselines = load_json(baseline_fname)
        results = load_json(results_fname)

        report = generate_regression_report(baselines, results)
        # "No baselines found. Run `git checkout main;UPDATE_BASELINE=1 pytest -m regression; git checkout -`"
        with open(dirname + "/regression_test_report.txt", "w") as f:
            f.write(report)

        # the following allows to print the report to the console despite pytest
        # capturing the output and without specifying the "-s" flag
        capmanager = request.config.pluginmanager.getplugin("capturemanager")
        with capmanager.global_and_fixture_disabled():
            print("\n\n\nRegression Test Report\n----------------------\n")
            print(report)

    if len(collected_regression_tests) > 0:
        if NEW_BASELINE:
            request.addfinalizer(update_baseline)
        request.addfinalizer(print_regression_report)
