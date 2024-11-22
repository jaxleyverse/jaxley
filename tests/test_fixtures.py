# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import time
import warnings

import pytest

import jaxley as jx

pytest.skip(allow_module_level=True)


def test_module_retrieval(SimpleNet):
    t0 = time.time()
    comp = jx.Compartment()
    branch = jx.Branch([comp] * 4)
    cell = jx.Cell([branch] * 4, [-1, 0, 0, 1])
    net = jx.Network([cell] * 2)
    t1 = time.time()

    net = SimpleNet(2, 4, 4, force_init=False)
    t2 = time.time()

    assert ((t2 - t1) - (t1 - t0)) / (
        t1 - t0
    ) < 0.1, f"Fixture is slower than manual init."

    net = SimpleNet(2, 4, 4, force_init=False)
    t3 = time.time()
    assert (
        t1 - t0 > t2 - t1 > t3 - t2
    ), f"T_get: from pre-existing fixture {t3 - t2}, from fixture: {(t2 - t1)}, manual: {(t1 - t0)}"


def test_direct_submodule_retrieval(SimpleBranch):
    t1 = time.time()
    branch = SimpleBranch(2, 3, force_init=False)
    t2 = time.time()
    branch = SimpleBranch(4, 3, force_init=False)
    t3 = time.time()
    assert (
        t2 - t1 > t3 - t2
    ), f"T_get: from pre-existing fixture {t3 - t2}, from fixture: {(t2 - t1)}"


def test_recursive_submodule_retrieval(SimpleNet):
    t1 = time.time()
    net = SimpleNet(3, 4, 3, force_init=False)
    t2 = time.time()
    net = SimpleNet(3, 4, 3, force_init=False)
    t3 = time.time()
    assert (
        t2 - t1 > t3 - t2
    ), f"T_get: from pre-existing fixture {t3 - t2}, from fixture: {(t2 - t1)}"


def test_module_reinit(SimpleComp):
    t0 = time.time()
    comp = jx.Compartment()
    t1 = time.time()

    comp = SimpleComp(force_init=False)

    t2 = time.time()
    comp = SimpleComp(force_init=False)
    t3 = time.time()
    net = SimpleComp(force_init=True)
    t4 = time.time()

    msg = f"T_get: reinit {t4 - t3}, from fixture: {(t3 - t2)}, manual: {(t1 - t0)}"
    assert t1 - t0 > t4 - t3 or abs(((t1 - t0) - (t4 - t3)) / (t1 - t0)) < 0.3, msg
    assert t4 - t3 > t3 - t2, msg
