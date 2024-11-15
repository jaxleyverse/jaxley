import time
import warnings


def test_module_retrieval(SimpleNet):
    t1 = time.time()
    net = SimpleNet(2, 4, 4)
    t2 = time.time()
    net = SimpleNet(2, 4, 4)
    t3 = time.time()
    assert t2 - t1 > t3 - t2


def test_direct_submodule_retrieval(SimpleBranch):
    t1 = time.time()
    branch = SimpleBranch(2, 4)
    t2 = time.time()
    branch = SimpleBranch(4, 4)
    t3 = time.time()
    assert t2 - t1 > t3 - t2


def test_recursive_submodule_retrieval(SimpleNet):
    t1 = time.time()
    net = SimpleNet(3, 4, 4)
    t2 = time.time()
    net = SimpleNet(3, 4, 4)
    t3 = time.time()
    assert t2 - t1 > t3 - t2
