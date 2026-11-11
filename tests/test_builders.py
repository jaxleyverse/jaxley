# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from tests import builders as build


def test_builders_cache_and_copy():
    """Builders build once and return independent copies."""
    calls_before = build.net.cache_info()
    net1 = build.net(ncells=2, nbranches=3, ncomp=2)
    net1.set("radius", 5.0)
    net2 = build.net(ncells=2, nbranches=3, ncomp=2)

    assert net2 is not net1
    assert (net2.nodes["radius"] == 1.0).all()  # The change to net1 did not leak.
    assert build.net.cache_info().hits >= calls_before.hits + 1


def test_binary_tree_parents():
    assert build.binary_tree_parents(1) == [-1]
    assert build.binary_tree_parents(7) == [-1, 0, 0, 1, 1, 2, 2]
