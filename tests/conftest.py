# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Settings shared by all tests. Modules for tests are built with `tests/builders.py`."""

import re
import sys

import jax

# Before any test module is imported, so that every test (also when run alone) uses
# float64 on the CPU.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import pytest


def _is_perf_run(config) -> bool:
    return config.option.markexpr == "perf"


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """Run `pytest -m perf` serially, despite `-n 4` in `addopts`: parallel workers would
    disturb the timings, and the table is printed by this process."""
    if _is_perf_run(config):
        config.option.numprocesses = 0


def pytest_configure(config):
    """Keep compiled XLA programs on disk, so later runs skip compiling.

    Most of the test time is compiling, mostly many small programs, so all of them are
    cached. The cache is in `.pytest_cache/d/jax`, so `pytest --cache-clear` resets it.
    Not for `pytest -m perf`, which measures compile times.
    """
    if hasattr(config, "cache") and not _is_perf_run(config):
        jax.config.update("jax_compilation_cache_dir", str(config.cache.mkdir("jax")))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def pytest_addoption(parser):
    parser.addoption(
        "--force-regen",
        action="store_true",
        help="Overwrite the golden values (`.npz` files) with the current results.",
    )


class Golden:
    """Compares arrays with golden values stored next to the test.

    The file is `<test module>/<test name>.npz`, next to the test module. If it is
    missing, or with `pytest --force-regen`, the current values are written and the
    test fails, so that a regeneration is always visible.
    """

    def __init__(self, request):
        name = re.sub(r"\W", "_", request.node.name)
        self.path = request.path.with_suffix("") / f"{name}.npz"
        self.force_regen = request.config.getoption("force_regen")

    def check(self, arrays: dict, atol=0.0, rtol=0.0, tolerances=None):
        """Compare `arrays` (`{name: array}`) with the golden values.

        `atol` and `rtol` apply to all arrays. `tolerances` replaces them for single
        arrays, e.g. `tolerances={"cacon": dict(atol=1e-10)}` (unset values are 0).
        """
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        if self.force_regen or not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path, **arrays)
            pytest.fail(f"Golden values written to {self.path}, check and commit it.")

        with np.load(self.path) as stored:
            expected = dict(stored)
        assert set(arrays) == set(expected), f"Keys differ from {self.path}"
        for key, value in arrays.items():
            tol = dict(atol=atol, rtol=rtol)
            if tolerances and key in tolerances:
                tol = dict(atol=0.0, rtol=0.0) | tolerances[key]
            assert value.shape == expected[key].shape, f"Shape of {key} changed"
            np.testing.assert_allclose(
                value, expected[key], **tol, err_msg=f"{key} differs from {self.path}"
            )


@pytest.fixture
def golden(request):
    """Golden values of the test, see `Golden`."""
    return Golden(request)


@pytest.fixture(autouse=True)
def seed_numpy():
    """Make tests that use `np.random` (directly or through jaxley) reproducible."""
    np.random.seed(0)


def pytest_terminal_summary(terminalreporter):
    """Print the timings of `pytest -m perf` (see `tests/test_perf.py`)."""
    perf = sys.modules.get("tests.test_perf")
    if perf is not None and perf.RESULTS:
        backend = perf.jax.default_backend()
        table = perf.format_table(perf.RESULTS, f"perf [s] ({backend})")
        terminalreporter.write_line("\n" + table)
