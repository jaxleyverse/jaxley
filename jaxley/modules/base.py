import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from math import pi
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax.lax import ScatterDimensionNumbers, scatter_add

from jaxley.channels import Channel
from jaxley.solver_voltage import step_voltage_explicit, step_voltage_implicit
from jaxley.synapses import Synapse
from jaxley.utils.cell_utils import (
    _compute_index_of_child,
    _compute_num_children,
    compute_levels,
)
from jaxley.utils.plot_utils import plot_morph


class Module(ABC):
    def __init__(self):
        self.nseg: int = None
        self.total_nbranches: int = 0
        self.nbranches_per_cell: List[int] = None

        self.conns: List[Synapse] = None
        self.group_views = {}

        self.nodes: Optional[pd.DataFrame] = None

        self.syn_edges = pd.DataFrame(
            columns=[
                "pre_locs",
                "post_locs",
                "pre_branch_index",
                "post_branch_index",
                "pre_cell_index",
                "post_cell_index",
                "type",
                "type_ind",
                "global_pre_comp_index",
                "global_post_comp_index",
                "global_pre_branch_index",
                "global_post_branch_index",
            ]
        )
        self.branch_edges: Optional[pd.DataFrame] = None

        self.cumsum_nbranches: Optional[jnp.ndarray] = None

        self.comb_parents: jnp.ndarray = jnp.asarray([-1])
        self.comb_branches_in_each_level: List[jnp.ndarray] = [jnp.asarray([0])]

        self.initialized_morph: bool = False
        self.initialized_syns: bool = False

        self.params: Dict[str, jnp.ndarray] = {}
        self.states: Dict[str, jnp.ndarray] = {}

        self.syn_params: Dict[str, jnp.ndarray] = {}
        self.syn_states: Dict[str, jnp.ndarray] = {}
        self.syn_classes: List = []

        # List of all `jx.Channel`s.
        self.channels: List[Channel] = []

        # For trainable parameters.
        self.indices_set_by_trainables: List[jnp.ndarray] = []
        self.trainable_params: List[Dict[str, jnp.ndarray]] = []
        self.allow_make_trainable: bool = True
        self.num_trainable_params: int = 0

        # For recordings.
        self.recordings: pd.DataFrame = pd.DataFrame().from_dict({})

        # For stimuli.
        self.currents: Optional[jnp.ndarray] = None
        self.current_inds: pd.DataFrame = pd.DataFrame().from_dict({})

        # x, y, z coordinates and radius.
        self.xyzr: List[np.ndarray] = []

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channels)} different channels. Use `.show()` for details."

    def __str__(self):
        return f"jx.{type(self).__name__}"

    def show(
        self,
        channel_name: Optional[str] = None,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        """Print detailed information about the Module."""
        if channel_name is None:
            return self._show_base(self.nodes, indices, params, states)
        else:
            return self._show_channel(
                self.nodes,
                channel_name,
                indices,
                params,
                states,
            )

    def _show_base(
        self,
        view,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        inds = view.index.values
        printable_nodes = deepcopy(view)

        if not indices:
            for key in printable_nodes:
                printable_nodes = printable_nodes.drop(key, axis=1)

        if params:
            for key, val in self.params.items():
                printable_nodes[key] = val[inds]

        if states:
            for key, val in self.states.items():
                printable_nodes[key] = val[inds]

        return printable_nodes

    def _show_channel(self, view, channel_name, indices, params, states):
        ind_of_params = self.channel_inds(view.index.values, channel_name)
        nodes = deepcopy(self.channel_nodes[channel_name].loc[ind_of_params])
        # TODO remove the following lines?
        nodes["comp_index"] -= nodes["comp_index"].iloc[0]
        nodes["branch_index"] -= nodes["branch_index"].iloc[0]
        nodes["cell_index"] -= nodes["cell_index"].iloc[0]

        if not indices:
            for key in nodes:
                nodes = nodes.drop(key, axis=1)

        if params:
            for key, val in self.channel_params.items():
                nodes[key] = val[ind_of_params]

        if states:
            for key, val in self.channel_states.items():
                nodes[key] = val[ind_of_params]

        return nodes

    @abstractmethod
    def init_conds(self, params):
        """Initialize coupling conductances.

        Args:
            params: Conductances and morphology parameters, not yet including
                coupling conductances.
        """
        raise NotImplementedError

    def _append_to_channel_nodes(self, view, channel: "jx.Channel"):
        """Adds channel nodes from constituents to `self.channel_nodes`."""
        name = type(channel).__name__

        # Channel does not yet exist in the `jx.Module` at all.
        if channel not in self.channels:
            self.channels.append(channel)
            self.nodes[name] = False

        # Add a binary column that indicates if a channel is present.
        self.nodes.loc[view.index.values, name] = True

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_params:
            self.nodes.loc[view.index.values, key] = channel.channel_params[key]

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_states:
            self.nodes.loc[view.index.values, key] = channel.channel_states[key]

    def set_params(self, key, val):
        """Set parameter."""
        # Alternatively, we could do `assert key not in self.syn_params`.
        nodes = self.syn_edges if key in self.syn_params else self.nodes
        self._set_params(key, val, nodes)

    def _set_params(self, key, val, view):
        if key in self.params:
            self.params[key] = self.params[key].at[view.index.values].set(val)
        elif key in self.channel_params:
            # TODO
            channel_name = self.identify_channel_based_on_param_name(key)
            ind_of_params = self.channel_inds(view.index.values, channel_name)
            self.channel_params[key] = (
                self.channel_params[key].at[ind_of_params].set(val)
            )
        elif key in self.syn_params:
            self.syn_params[key] = self.syn_params[key].at[view.index.values].set(val)
        else:
            raise KeyError("Key not recognized.")

    def set_states(self, key, val):
        """Set parameters."""
        # Alternatively, we could do `assert key not in self.syn_states`.
        nodes = self.syn_edges if key in self.syn_states else self.nodes
        self._set_states(key, val, nodes)

    def _set_states(self, key: str, val: float, view):
        if key in self.states:
            self.states[key] = self.states[key].at[view.index.values].set(val)
        elif key in self.channel_states:
            # TODO
            channel_name = self.identify_channel_based_on_state_name(key)
            ind_of_params = self.channel_inds(view.index.values, channel_name)
            self.channel_states[key] = (
                self.channel_states[key].at[ind_of_params].set(val)
            )
        elif key in self.syn_states:
            self.syn_states[key] = self.syn_states[key].at[view.index.values].set(val)
        else:
            raise KeyError("Key not recognized.")

    def get_params(self, key: str):
        """Return parameters."""
        # Alternatively, we could do `assert key not in self.syn_params`.
        nodes = self.syn_edges if key in self.syn_params else self.nodes
        return self._get_params(key, nodes)

    def _get_params(self, key: str, view):
        if key in self.params:
            return self.params[key][view.index.values]
        elif key in self.channel_params:
            channel_name = self.identify_channel_based_on_param_name(key)
            ind_of_params = self.channel_inds(view.index.values, channel_name)
            return self.channel_params[key][ind_of_params]
        elif key in self.syn_params:
            return self.syn_params[key][view.index.values]
        else:
            raise KeyError("Key not recognized.")

    def get_states(self, key: str):
        """Return states."""
        # Alternatively, we could do `assert key not in self.syn_states`.
        nodes = self.syn_edges if key in self.syn_states else self.nodes
        return self._get_states(key, nodes)

    def _get_states(self, key: str, view):
        if key in self.states:
            return self.states[key][view.index.values]
        elif key in self.channel_states:
            channel_name = self.identify_channel_based_on_state_name(key)
            ind_of_states = self.channel_inds(view.index.values, channel_name)
            return self.channel_states[key][ind_of_states]
        elif key in self.syn_states:
            return self.syn_states[key][view.index.values]
        else:
            raise KeyError("Key not recognized.")

    def make_trainable(
        self,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        """Make a parameter trainable.

        Args:
            key: Name of the parameter to make trainable.
            init_val: Initial value of the parameter. If `float`, the same value is
                used for every created parameter. If `list`, the length of the list has
                to match the number of created parameters. If `None`, the current
                parameter value is used and if parameter sharing is performed that the
                current parameter value is averaged over all shared parameters.
            verbose: Whether to print the number of parameters that are added and the
                total number of parameters.
        """
        view = deepcopy(self.nodes.assign(controlled_by_param=0))
        self._make_trainable(view, key, init_val, verbose=verbose)

    def _make_trainable(
        self,
        view,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        assert (
            self.allow_make_trainable
        ), "network.cell('all').make_trainable() is not supported. Use a for-loop over cells."

        grouped_view = view.groupby("controlled_by_param")
        inds_of_comps = list(grouped_view.apply(lambda x: x.index.values))

        if key in self.params:
            indices_per_param = jnp.stack(inds_of_comps)
            param_vals = self.params[key][indices_per_param]
        elif key in self.channel_params:
            name = self.identify_channel_based_on_param_name(key)
            indices_per_param = jnp.stack(
                [self.channel_inds(ind, name) for ind in inds_of_comps]
            )
            param_vals = self.channel_params[key][indices_per_param]
        elif key in self.syn_params:
            indices_per_param = jnp.stack(inds_of_comps)
            param_vals = self.syn_params[key][indices_per_param]
        else:
            raise KeyError(f"Parameter {key} not recognized.")

        self.indices_set_by_trainables.append(indices_per_param)

        num_created_parameters = len(indices_per_param)
        if init_val is not None:
            if isinstance(init_val, float):
                new_params = jnp.asarray([[init_val]] * num_created_parameters)
            elif isinstance(init_val, list):
                assert (
                    len(init_val) == num_created_parameters
                ), f"len(init_val)={len(init_val)}, but trying to create {num_created_parameters} parameters."
                new_params = jnp.asarray(init_val)[:, None]
            else:
                raise ValueError(
                    f"init_val must a float, list, or None, but it is a {type(init_val).__name__}."
                )
        else:
            new_params = jnp.mean(param_vals, axis=1, keepdims=True)

        self.trainable_params.append({key: new_params})
        self.num_trainable_params += num_created_parameters
        if verbose:
            print(
                f"Number of newly added trainable parameters: {num_created_parameters}. Total number of trainable parameters: {self.num_trainable_params}"
            )

    def add_to_group(self, group_name):
        raise ValueError("`add_to_group()` makes no sense for an entire module.")

    def _add_to_group(self, group_name, view):
        if group_name in self.group_views:
            view = pd.concat([self.group_views[group_name].view, view])
        self.group_views[group_name] = GroupView(self, view)

    def get_parameters(self):
        """Get all trainable parameters."""
        return self.trainable_params

    def get_all_parameters(self, trainable_params):
        """Return all parameters (and coupling conductances) needed to simulate.

        This is done by first obtaining the current value of every parameter (not only
        the trainable ones) and then replacing the trainable ones with the value
        in `trainable_params()`.
        """
        params = {}
        basic_param_names = ["length", "radius", "axial_resistivity"]
        for name in basic_param_names:
            params[name] = jnp.asarray(self.nodes[name].to_numpy())

        for key, val in self.syn_params.items():
            params[key] = val

        for channel in self.channels:
            channel_name = type(channel).__name__
            params[channel_name] = {}
            inds_of_channel = self.nodes.loc[self.nodes[channel_name]][
                "comp_index"
            ].to_numpy()
            for key in channel.channel_params:
                param_vals_with_nans = self.nodes[key].to_numpy()
                param_vals = param_vals_with_nans[inds_of_channel]
                params[channel_name][key] = param_vals

        for inds, set_param in zip(self.indices_set_by_trainables, trainable_params):
            for key in set_param.keys():
                params[key] = params[key].at[inds].set(set_param[key])

        # Compute conductance params and append them.
        cond_params = self.init_conds(params)
        for key in cond_params:
            params[key] = cond_params[key]

        return params

    @property
    def initialized(self):
        """Whether the `Module` is ready to be solved or not."""
        return self.initialized_morph and self.initialized_syns

    def initialize(self):
        """Initialize the module."""
        self.init_morph()
        self.init_syns()
        return self

    def record(self):
        """Insert a recording into the compartment."""
        self._record(self.nodes)

    def _record(self, view):
        assert (
            len(view) == 1
        ), "Can only record from compartments, not branches, cells, or networks."
        self.recordings = pd.concat([self.recordings, view])

    def delete_recordings(self):
        """Removes all recordings from the module."""
        self.recordings = pd.DataFrame().from_dict({})

    def stimulate(self, current):
        """Insert a stimulus into the compartment."""
        self._stimulate(current, self.nodes)

    def _stimulate(self, current, view):
        assert (
            len(view) == 1
        ), "Can only stimulate compartments, not branches, cells, or networks."
        if self.currents is not None:
            self.currents = jnp.concatenate(
                [self.currents, jnp.expand_dims(current, axis=0)]
            )
        else:
            self.currents = jnp.expand_dims(current, axis=0)
        self.current_inds = pd.concat([self.current_inds, view])

    def delete_stimuli(self):
        """Removes all stimuli from the module."""
        self.currents = None
        self.current_inds = pd.DataFrame().from_dict({})

    def insert(self, channel):
        """Insert a channel."""
        self._insert(channel, self.nodes)

    def _insert(self, channel, view):
        self._append_to_channel_nodes(view, channel)

    def init_syns(self):
        self.initialized_syns = True

    def init_morph(self):
        self.initialized_morph = True

    def step(
        self,
        u,
        delta_t,
        i_inds,
        i_current,
        params: Dict[str, jnp.ndarray],
        solver: str = "bwd_euler",
        tridiag_solver: str = "stone",
    ):
        """One step of integration."""
        voltages = u["voltages"]

        # Parameters have to go in here.
        u, (v_terms, const_terms) = self._step_channels(
            u, delta_t, self.channels, self.nodes, params
        )

        # External input.
        i_ext = self.get_external_input(
            voltages, i_inds, i_current, params["radius"], params["length"]
        )

        # Step of the synapse.
        u, syn_voltage_terms, syn_constant_terms = self._step_synapse(
            u,
            self.syn_classes,
            params,
            delta_t,
            self.syn_edges,
        )

        # Voltage steps.
        if solver == "bwd_euler":
            new_voltages = step_voltage_implicit(
                voltages=voltages,
                voltage_terms=v_terms + syn_voltage_terms,
                constant_terms=const_terms + i_ext + syn_constant_terms,
                coupling_conds_bwd=params["coupling_conds_bwd"],
                coupling_conds_fwd=params["coupling_conds_fwd"],
                summed_coupling_conds=params["summed_coupling_conds"],
                branch_cond_fwd=params["branch_conds_fwd"],
                branch_cond_bwd=params["branch_conds_bwd"],
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                branches_in_each_level=self.comb_branches_in_each_level,
                tridiag_solver=tridiag_solver,
                delta_t=delta_t,
            )
        else:
            new_voltages = step_voltage_explicit(
                voltages,
                v_terms + syn_voltage_terms,
                const_terms + i_ext + syn_constant_terms,
                coupling_conds_bwd=params["coupling_conds_bwd"],
                coupling_conds_fwd=params["coupling_conds_fwd"],
                branch_cond_fwd=params["branch_conds_fwd"],
                branch_cond_bwd=params["branch_conds_bwd"],
                nbranches=self.total_nbranches,
                parents=self.comb_parents,
                delta_t=delta_t,
            )

        u["voltages"] = new_voltages.flatten(order="C")

        return u

    @staticmethod
    def _step_channels(
        states,
        delta_t,
        channels: List[Channel],
        channel_nodes: List[pd.DataFrame],
        params: Dict[str, jnp.ndarray],
    ):
        """One step of integration of the channels."""
        voltages = states["voltages"]

        # Update states of the channels.
        new_channel_states = {}
        for channel in channels:
            name = type(channel).__name__
            indices = channel_nodes.loc[channel_nodes[name]]["comp_index"].to_numpy()
            states_updated = channel.update_states(
                states[name], delta_t, voltages[indices], params[name]
            )
            new_channel_states[name] = states_updated

        # Rebuild state.
        for key, val in new_channel_states.items():
            states[key] = val

        # Compute current through channels.
        voltage_terms = jnp.zeros_like(voltages)
        constant_terms = jnp.zeros_like(voltages)
        # Run with two different voltages that are `diff` apart to infer the slope and
        # offset.
        diff = 1e-3
        for channel in channels:
            name = type(channel).__name__
            indices = channel_nodes.loc[channel_nodes[name]]["comp_index"].to_numpy()
            v_and_perturbed = jnp.stack([voltages[indices], voltages[indices] + diff])
            membrane_currents = channel.vmapped_compute_current(
                states[name], v_and_perturbed, params[name]
            )
            voltage_term = (membrane_currents[1] - membrane_currents[0]) / diff
            constant_term = membrane_currents[0] - voltage_term * voltages[indices]
            voltage_terms = voltage_terms.at[indices].add(voltage_term)
            constant_terms = constant_terms.at[indices].add(-constant_term)

        return states, (voltage_terms, constant_terms)

    @staticmethod
    def _step_synapse(
        u,
        syn_channels,
        params,
        delta_t,
        edges,
    ):
        """One step of integration of the channels.

        `Network` overrides this method (because it actually has synapses), whereas
        `Compartment`, `Branch`, and `Cell` do not override this.
        """
        voltages = u["voltages"]
        return u, jnp.zeros_like(voltages), jnp.zeros_like(voltages)

    @staticmethod
    def get_external_input(
        voltages: jnp.ndarray,
        i_inds: jnp.ndarray,
        i_stim: jnp.ndarray,
        radius: float,
        length_single_compartment: float,
    ):
        """
        Return external input to each compartment in uA / cm^2.
        """
        zero_vec = jnp.zeros_like(voltages)
        # `radius`: um
        # `length_single_compartment`: um
        # `i_stim`: nA
        current = (
            i_stim / 2 / pi / radius[i_inds] / length_single_compartment[i_inds]
        )  # nA / um^2
        current *= 100_000  # Convert (nA / um^2) to (uA / cm^2)

        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        stim_at_timestep = scatter_add(zero_vec, i_inds[:, None], current, dnums)
        return stim_at_timestep

    def vis(
        self,
        ax=None,
        col: str = "k",
        dims: Tuple[int] = (0, 1),
        morph_plot_kwargs: Dict = {},
    ) -> None:
        """Visualize the module.

        Args:
            ax: An axis into which to plot.
            col: The color for all branches.
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
        """
        return self._vis(
            dims=dims,
            col=col,
            ax=ax,
            view=self.nodes,
            morph_plot_kwargs=morph_plot_kwargs,
        )

    def _vis(self, ax, col, dims, view, morph_plot_kwargs):
        branches_inds = view["branch_index"].to_numpy()
        coords = []
        for branch_ind in branches_inds:
            assert not np.any(
                np.isnan(self.xyzr[branch_ind][:, dims])
            ), "No coordinates available. Use `vis(detail='point')` or run `.compute_xyz()` before running `.vis()`."
            coords.append(self.xyzr[branch_ind])

        ax = plot_morph(
            coords,
            dims=dims,
            col=col,
            ax=ax,
            morph_plot_kwargs=morph_plot_kwargs,
        )

        return ax

    def compute_xyz(self):
        """Return xyz coordinates of every branch, based on the branch length."""
        max_y_multiplier = 5.0
        min_y_multiplier = 0.5

        parents = self.comb_parents
        num_children = _compute_num_children(parents)
        index_of_child = _compute_index_of_child(parents)
        levels = compute_levels(parents)

        # Extract branch.
        inds_branch = self.nodes.groupby("branch_index")["comp_index"].apply(list)
        branch_lens = [
            np.sum(self.params["length"][np.asarray(i)]) for i in inds_branch
        ]
        endpoints = []

        # Different levels will get a different "angle" at which the children emerge from
        # the parents. This angle is defined by the `y_offset_multiplier`. This value
        # defines the range between y-location of the first and of the last child of a
        # parent.
        y_offset_multiplier = np.linspace(
            max_y_multiplier, min_y_multiplier, np.max(levels) + 1
        )

        for b in range(self.total_nbranches):
            # For networks with mixed SWC and from-scatch neurons, only update those
            # branches that do not have coordingates yet.
            if np.any(np.isnan(self.xyzr[b])):
                if parents[b] > -1:
                    start_point = endpoints[parents[b]]
                    num_children_of_parent = num_children[parents[b]]
                    y_offset = (
                        ((index_of_child[b] / (num_children_of_parent - 1))) - 0.5
                    ) * y_offset_multiplier[levels[b]]
                else:
                    start_point = [0, 0]
                    y_offset = 0.0

                len_of_path = np.sqrt(y_offset**2 + 1.0)

                end_point = [
                    start_point[0] + branch_lens[b] / len_of_path * 1.0,
                    start_point[1] + branch_lens[b] / len_of_path * y_offset,
                ]
                endpoints.append(end_point)

                self.xyzr[b][:, :2] = np.asarray([start_point, end_point])
            else:
                # Dummy to keey the index `endpoints[parent[b]]` above working.
                endpoints.append(np.zeros((2,)))

    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Move cells or networks in the (x, y, z) plane."""
        self._move(x, y, z, self.nodes)

    def _move(self, x: float, y: float, z: float, view):
        # Need to cast to set because this will return one columnn per compartment,
        # not one column per branch.
        indizes = set(view["branch_index"].to_numpy().tolist())
        for i in indizes:
            self.xyzr[i][:, 0] += x
            self.xyzr[i][:, 1] += y
            self.xyzr[i][:, 2] += z


class View:
    """View of a `Module`."""

    def __init__(self, pointer: Module, view: pd.DataFrame):
        self.pointer = pointer
        self.view = view
        self.allow_make_trainable = True

    def __repr__(self):
        return f"{type(self).__name__}. Use `.show()` for details."

    def __str__(self):
        return f"{type(self).__name__}"

    def show(
        self,
        channel_name: Optional[str] = None,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
    ):
        if channel_name is None:
            myview = self.view.drop("global_comp_index", axis=1)
            myview = myview.drop("global_branch_index", axis=1)
            myview = myview.drop("global_cell_index", axis=1)
            return self.pointer._show_base(myview, indices, params, states)
        else:
            return self.pointer._show_channel(
                self.view, channel_name, indices, params, states
            )

    def set_global_index_and_index(self, nodes):
        """Use the global compartment, branch, and cell index as the index."""
        nodes = nodes.drop("controlled_by_param", axis=1)
        nodes = nodes.drop("comp_index", axis=1)
        nodes = nodes.drop("branch_index", axis=1)
        nodes = nodes.drop("cell_index", axis=1)
        nodes = nodes.rename(
            columns={
                "global_comp_index": "comp_index",
                "global_branch_index": "branch_index",
                "global_cell_index": "cell_index",
            }
        )
        return nodes

    def insert(self, channel):
        """Insert a channel."""
        assert not inspect.isclass(
            channel
        ), """
            Channel is a class, but it was not initialized. Use `.insert(Channel())` 
            instead of `.insert(Channel)`.
            """
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._insert(channel, nodes)

    def record(self):
        """Insert a channel."""
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._record(nodes)

    def stimulate(self, current):
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._stimulate(current, nodes)

    def set_params(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.pointer._set_params(key, val, self.view)

    def set_states(self, key: str, val: float):
        """Set parameters of the pointer."""
        self.pointer._set_states(key, val, self.view)

    def get_params(self, key: str):
        """Return parameters."""
        return self.pointer._get_params(key, self.view)

    def get_states(self, key: str):
        """Return states."""
        return self.pointer._get_states(key, self.view)

    def make_trainable(self, key: str, init_val: Optional[Union[float, list]] = None):
        """Make a parameter trainable."""
        self.pointer._make_trainable(self.view, key, init_val)

    def add_to_group(self, group_name: str):
        self.pointer._add_to_group(group_name, self.view)

    def vis(
        self,
        ax=None,
        col="k",
        dims=(0, 1),
        morph_plot_kwargs: Dict = {},
    ):
        nodes = self.set_global_index_and_index(self.view)
        return self.pointer._vis(
            ax=ax,
            col=col,
            dims=dims,
            view=nodes,
            morph_plot_kwargs=morph_plot_kwargs,
        )

    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        nodes = self.set_global_index_and_index(self.view)
        self.pointer._move(x, y, z, nodes)

    def adjust_view(self, key: str, index: float):
        """Update view."""
        if isinstance(index, int) or isinstance(index, np.int64):
            self.view = self.view[self.view[key] == index]
        elif isinstance(index, list):
            self.view = self.view[self.view[key].isin(index)]
        else:
            assert index == "all"
        self.view["controlled_by_param"] -= self.view["controlled_by_param"].iloc[0]
        return self


class GroupView(View):
    """GroupView (aka sectionlist).

    The only difference to a standard `View` is that it sets `controlled_by_param` to
    0 for all compartments. This means that a group will always be controlled by a
    single parameter.
    """

    def __init__(self, pointer, view):
        view["controlled_by_param"] = 0
        super().__init__(pointer, view)
