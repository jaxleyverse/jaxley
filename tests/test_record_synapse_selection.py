# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
"""Recording must intersect the requested synapse type with the selected view."""

import jax
import numpy as np
import pytest

import jaxley as jx
from jaxley.channels import Leak
from jaxley.synapses import IonotropicSynapse, Synapse

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def make_network(target_count=2, other_first=False):
    cell = jx.Cell()
    cell.insert(Leak())
    net = jx.Network([cell] * 5)
    links = [
        (list(range(target_count)), [3] * target_count, "target"),
        ([0, 1, 2], [4, 4, 4], "other"),
    ]
    if other_first:
        links.reverse()
    for pre, post, name in links:
        jx.connect(
            net.select(nodes=pre), net.select(nodes=post), IonotropicSynapse(name)
        )
    net.target.edge(0).set("target_s", 0.1)
    if target_count > 1:
        net.target.edge(1).set("target_s", 0.8)
    return net


@pytest.mark.parametrize("state", ["target_s", "i_target"])
def test_whole_network_records_only_target_type(state):
    """The minimal failing test: one target and three unrelated synapses."""
    net = make_network(target_count=1)
    net.record(state, verbose=False)
    assert net.recordings.rec_index.tolist() == [0]
    assert net.recordings.state.tolist() == [state]


@pytest.mark.parametrize("state", ["target_s", "i_target"])
@pytest.mark.parametrize("other_first", [False, True])
def test_outputs_match_explicit_type_view(state, other_first):
    net = make_network(other_first=other_first)
    net.record(state, verbose=False)
    whole = jx.integrate(net, t_max=0.1)
    net.delete_recordings()
    net.target.record(state, verbose=False)
    explicit = jx.integrate(net, t_max=0.1)
    assert whole.shape == explicit.shape
    np.testing.assert_allclose(whole, explicit, atol=1e-12)
    assert not np.allclose(explicit[0], explicit[1])


@pytest.mark.parametrize("state", ["target_s", "i_target"])
def test_mixed_view_keeps_only_selected_target_edges_and_order(state):
    net = make_network()
    net.select(edges=np.array([4, 1])).record(state, verbose=False)
    assert net.recordings.rec_index.tolist() == [1]
    net.delete_recordings()
    view = net.select(edges=np.array([4, 1, 0]))
    expected = view.edges.loc[view.edges.type == "target", "index_within_type"].tolist()
    view.record(state, verbose=False)
    assert net.recordings.rec_index.tolist() == expected


def test_recording_order_duplicate_calls_and_voltage_are_unchanged():
    net = make_network(other_first=True)
    net.cell(2).record("v", verbose=False)
    net.record("i_target", verbose=False)
    net.record("i_target", verbose=False)
    net.cell(0).record("v", verbose=False)
    assert list(zip(net.recordings.state, net.recordings.rec_index)) == [
        ("v", 2),
        ("i_target", 0),
        ("i_target", 1),
        ("v", 0),
    ]
    with pytest.raises(KeyError):
        net.record("unknown_state", verbose=False)


def test_view_without_requested_current_adds_no_records():
    net = make_network()
    net.cell(0).record(verbose=False)
    before = net.recordings.copy()
    # Current names are globally registered even when this view has no such edge.
    net.other.record("i_target", verbose=False)
    assert net.recordings.equals(before)


def test_state_values_do_not_determine_type_membership():
    net = make_network()
    net.target.edge(1).set("target_s", np.nan)
    net.record("target_s", verbose=False)
    # Preserve the request; do not hide a bad model state by dropping its recording.
    assert net.recordings.rec_index.tolist() == [0, 1]


class NonPrefixedStateSynapse(Synapse):
    def __init__(self):
        super().__init__("custom_name_with_underscores")
        self.synapse_params = {"custom_g": 1e-4}
        self.synapse_states = {"occupancy": 0.2}

    def update_states(self, states, delta_t, pre_voltage, post_voltage, params):
        return {"occupancy": states["occupancy"]}

    def compute_current(self, states, pre_voltage, post_voltage, params):
        return params["custom_g"] * states["occupancy"] * post_voltage


def test_custom_state_name_is_resolved_from_mechanism_metadata():
    net = make_network()
    jx.connect(net.cell(2), net.cell(3), NonPrefixedStateSynapse())
    net.record("occupancy", verbose=False)
    assert net.recordings.rec_index.tolist() == [0]
    assert jx.integrate(net, t_max=0.1).shape[0] == 1
