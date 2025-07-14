# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from jax import jit, vmap
from jax.lax import ScatterDimensionNumbers, scatter_add
from matplotlib.axes import Axes

from jaxley.channels import Channel
from jaxley.pumps import Pump
from jaxley.solver_voltage import (
    step_voltage_explicit,
    step_voltage_implicit_with_dhs_solve,
    step_voltage_implicit_with_jax_spsolve,
    step_voltage_implicit_with_stone,
)
from jaxley.utils.cell_utils import (
    _compute_index_of_child,
    _compute_num_children,
    _get_comp_edges_in_view,
    compute_levels,
    convert_point_process_to_distributed,
    interpolate_xyzr,
    params_to_pstate,
    query_channel_states_and_params,
    v_interp,
)
from jaxley.utils.debug_solver import compute_morphology_indices
from jaxley.utils.jax_utils import infer_device
from jaxley.utils.misc_utils import cumsum_leading_zero, deprecated, is_str_all
from jaxley.utils.morph_attributes import (
    compute_axial_conductances,
    cylinder_area,
    cylinder_resistive_load,
    cylinder_volume,
    morph_attrs_from_xyzr,
    split_xyzr_into_equal_length_segments,
)
from jaxley.utils.plot_utils import plot_comps, plot_graph, plot_morph
from jaxley.utils.solver_utils import (
    comp_edges_to_indices,
    convert_to_csc,
    dhs_group_comps_into_levels,
    dhs_permutation_indices,
    dhs_solve_index,
)


def only_allow_module(func):
    """Decorator to only allow the function to be called on Module instances.

    Decorates methods of Module that cannot be called on Views of Modules instances.
    and have to be called on the Module itself."""

    def wrapper(self, *args, **kwargs):
        module_name = self.base.__class__.__name__
        method_name = func.__name__
        assert not isinstance(self, View), (
            f"{method_name} is currently not supported for Views. Call on "
            f"the {module_name} base Module."
        )
        return func(self, *args, **kwargs)

    return wrapper


class Module(ABC):
    """Module base class which implements features shared by all modules.

    Modules are everything that can be passed to `jx.integrate`, i.e. compartments,
    branches, cells, and networks.

    This base class defines the scaffold for all jaxley modules (compartments,
    branches, cells, networks).

    Modules can be traversed and modified using the `at`, `cell`, `branch`, `comp`,
    `edge`, and `loc` methods. The `scope` method can be used to toggle between
    global and local indices. Traversal of Modules will return a `View` of itself,
    that has a modified set of attributes, which only consider the part of the Module
    that is in view.

    For developers: The above has consequences for how to operate on `Module` and which
    changes take affect where. The following guidelines should be followed (copied from
    `View`):

    1. We consider a Module to have everything in view.
    2. Views can display and keep track of how a module is traversed. But(!),
       do not support making changes or setting variables. This still has to be
       done in the base Module, i.e. `self.base`. In order to enssure that these
       changes only affects whatever is currently in view `self._nodes_in_view`,
       or `self._edges_in_view` among others have to be used. Operating on nodes
       currently in view can for example be done with
       `self.base.node.loc[self._nodes_in_view]`.
    3. Every attribute of Module that changes based on what's in view, i.e. `xyzr`,
       needs to modified when View is instantiated. I.e. `xyzr` of `cell.branch(0)`,
       should be `[self.base.xyzr[0]]` This could be achieved via:
       `[self.base.xyzr[b] for b in self._branches_in_view]`.

    For developers: If you want to add a new method to `Module`, here is an example of
    how to make methods of Module compatible with View:

    .. code-block:: python

        # Use data in view to return something.
        def count_small_branches(self):
            # no need to use self.base.attr + viewed indices,
            # since no change is made to the attr in question (nodes)
            comp_lens = self.nodes["length"]
            branch_lens = comp_lens.groupby("global_branch_index").sum()
            return np.sum(branch_lens < 10)

        # Change data in view.
        def change_attr_in_view(self):
            # changes to attrs have to be made via self.base.attr + viewed indices
            a = func1(self.base.attr1[self._cells_in_view])
            b = func2(self.base.attr2[self._edges_in_view])
            self.base.attr3[self._branches_in_view] = a + b
    """

    def __init__(self):
        self._solver_device = infer_device()
        self.ncomp: int = None
        self.total_nbranches: int = 0
        self.nbranches_per_cell: List[int] = None

        self.group_names: List[str] = []

        self.nodes: Optional[pd.DataFrame] = None
        self._scope = "local"  # defaults to local scope
        self._nodes_in_view: np.ndarray = None
        self._edges_in_view: np.ndarray = None

        self._branchpoints: pd.DataFrame = pd.DataFrame(columns=["x", "y", "z"])
        self._comp_edges: pd.DataFrame = pd.DataFrame()

        self.edges = pd.DataFrame(
            columns=[
                "global_edge_index",
                "index_within_type",
                "pre_index",
                "post_index",
                "pre_locs",
                "post_locs",
                "type",
                "type_ind",
            ]
        )

        self._cumsum_nbranches: Optional[np.ndarray] = None

        self.comb_parents: jnp.ndarray = jnp.asarray([-1])

        self.initialized_solver: bool = False
        self.initialized_syns: bool = False

        # List of all types of `jx.Synapse`s.
        self.synapses: List = []
        self.synapse_param_names = []
        self.synapse_state_names = []
        self.synapse_names = []
        self.synapse_current_names: List[str] = []

        # List of types of all `jx.Channel`s.
        self.channels: List[Channel] = []
        self.membrane_current_names: List[str] = []

        # List of all pumps.
        self.pumped_ions: List[str] = []
        self.pumps: List[Pump] = []

        # List of all states (excluding voltage) that are being diffused.
        self.diffusion_states: List[str] = []

        # For trainable parameters.
        self.indices_set_by_trainables: List[jnp.ndarray] = []
        self.trainable_params: List[Dict[str, jnp.ndarray]] = []
        self.allow_make_trainable: bool = True
        self.num_trainable_params: int = 0

        # For recordings.
        self.recordings: pd.DataFrame = pd.DataFrame().from_dict({})

        # For stimuli or clamps.
        # E.g. `self.externals = {"v": zeros(1000,2), "i": ones(1000, 2)}`
        # for 1000 timesteps and two compartments.
        self.externals: Dict[str, jnp.ndarray] = {}
        # E.g. `self.external)inds = {"v": jnp.asarray([0,1]), "i": jnp.asarray([2,3])}`
        self.external_inds: Dict[str, jnp.ndarray] = {}

        # x, y, z coordinates and radius.
        self.xyzr: List[np.ndarray] = []
        self._radius_generating_fns = None  # Defined by `.read_swc()`.

        # For debugging the solver. Will be empty by default and only filled if
        # `self._init_morph_for_debugging` is run.
        self.debug_states = {}

        # needs to be set at the end
        self.base: Module = self

    def __repr__(self):
        return f"{type(self).__name__} with {len(self.channels)} different channels. Use `.nodes` for details."

    def __str__(self):
        return f"jx.{type(self).__name__}"

    def __dir__(self):
        base_dir = object.__dir__(self)
        return sorted(base_dir + self.synapse_names + list(self.group_nodes.keys()))

    def __getattr__(self, key):
        # Ensure that hidden methods such as `__deepcopy__` still work.
        if key.startswith("__"):
            return super().__getattribute__(key)

        # intercepts calls to groups
        if key in self.base.group_names:
            view = self.select(self.nodes[key])
            view._set_controlled_by_param(key)
            return view

        # intercepts calls to channels
        if key in [c._name for c in self.base.channels]:
            channel_names = [c._name for c in self.channels]
            inds = self.nodes.index[self.nodes[key]].to_numpy()
            view = self.select(inds) if key in channel_names else self.select(None)
            view._set_controlled_by_param(key)
            return view

        # intercepts calls to synapse types
        if key in self.base.synapse_names:
            syn_inds = self.edges[self.edges["type"] == key][
                "global_edge_index"
            ].to_numpy()
            orig_scope = self._scope
            view = (
                self.scope("global").edge(syn_inds).scope(orig_scope)
                if key in self.synapse_names
                else self.select(None)
            )
            view._set_controlled_by_param(key)  # overwrites param set by edge
            # Ensure synapse param sharing works with `edge`
            # `edge` will be removed as part of #463
            view.edges["local_edge_index"] = np.arange(len(view.edges))
            return view

    def _childviews(self) -> List[str]:
        """Returns levels that module can be viewed at.

        I.e. for net -> [cell, branch, comp]. For branch -> [comp]"""
        levels = ["network", "cell", "branch", "comp"]
        if self._current_view in levels:
            children = levels[levels.index(self._current_view) + 1 :]
            return children
        return []

    def _has_childview(self, key: str) -> bool:
        child_views = self._childviews()
        return key in child_views

    def __getitem__(self, index):
        """Lazy indexing of the module."""
        supported_parents = ["network", "cell", "branch"]  # cannot index into comp

        not_group_view = self._current_view not in self.group_names
        assert (
            self._current_view in supported_parents or not_group_view
        ), "Lazy indexing is only supported for `Network`, `Cell`, `Branch` and Views thereof."
        index = index if isinstance(index, tuple) else (index,)

        child_views = self._childviews()
        assert len(index) <= len(child_views), "Too many indices."
        view = self
        for i, child in zip(index, child_views):
            view = view._at_nodes(child, i)
        return view

    def _update_local_indices(self) -> pd.DataFrame:
        """Compute local indices from the global indices that are in view.

        This is recomputed everytime a View is created."""
        rerank = lambda df: df.rank(method="dense").astype(int) - 1

        def reorder_cols(
            df: pd.DataFrame, cols: List[str], first: bool = True
        ) -> pd.DataFrame:
            """Move cols to front/back.

            Args:
                df: DataFrame to reorder.
                cols: List of columns to place before/after remaining columns.
                first: If True, cols are placed in front, otherwise at the end.

            Returns:
                DataFrame with reordered columns."""
            new_cols = [col for col in df.columns if first == (col in cols)]
            new_cols += [col for col in df.columns if first != (col in cols)]
            return df[new_cols]

        def reindex_a_by_b(
            df: pd.DataFrame, a: str, b: Optional[Union[str, List[str]]] = None
        ) -> pd.DataFrame:
            """Reindex based on a different col or several columns
            for b=[0,0,1,1,2,2,2] -> a=[0,1,0,1,0,1,2]"""
            grouped_df = df.groupby(b) if b is not None else df
            df.loc[:, a] = rerank(grouped_df[a])
            return df

        index_names = ["cell_index", "branch_index", "comp_index"]  # order is important
        global_idx_cols = [f"global_{name}" for name in index_names]
        local_idx_cols = [f"local_{name}" for name in index_names]
        idcs = self.nodes[global_idx_cols]

        # update local indices of nodes
        idcs = reindex_a_by_b(idcs, global_idx_cols[0])
        idcs = reindex_a_by_b(idcs, global_idx_cols[1], global_idx_cols[0])
        idcs = reindex_a_by_b(idcs, global_idx_cols[2], global_idx_cols[:2])
        idcs.columns = [col.replace("global", "local") for col in global_idx_cols]
        self.nodes[local_idx_cols] = idcs[local_idx_cols].astype(int)

        # move indices to the front of the dataframe; move controlled_by_param to the end
        # move indices of current scope to the front and the others to the back
        not_scope = "global" if self._scope == "local" else "local"
        self.nodes = reorder_cols(
            self.nodes, [f"{self._scope}_{name}" for name in index_names], first=True
        )
        self.nodes = reorder_cols(
            self.nodes, [f"{not_scope}_{name}" for name in index_names], first=False
        )

        self.edges = reorder_cols(self.edges, ["global_edge_index"])
        self.nodes = reorder_cols(self.nodes, ["controlled_by_param"], first=False)
        self.edges = reorder_cols(self.edges, ["controlled_by_param"], first=False)

    def _init_view(self):
        """Init attributes critical for View.

        Needs to be called at init of a Module."""
        modules = ["compartment", "branch", "cell", "network"]
        module_inheritance = [c.__name__.lower() for c in self.__class__.__mro__]
        module_type = next((t for t in modules if t in module_inheritance), None)
        self._current_view = "comp" if module_type == "compartment" else module_type
        self._nodes_in_view = self.nodes.index.to_numpy()
        self._edges_in_view = self.edges.index.to_numpy()

        # To enable updating `self._comp_edges` and `self._branchpoints` during `View`.
        self._comp_edges_in_view = self._comp_edges.index.to_numpy()
        self._branchpoints_in_view = self._branchpoints.index.to_numpy()

        self.nodes["controlled_by_param"] = 0

    def _compute_coords_of_comp_centers(self) -> np.ndarray:
        """Compute xyz coordinates of compartment centers.

        Centers are the midpoint between the compartment endpoints on the morphology
        as defined by xyzr.

        Note: For sake of performance, interpolation is not done for each branch
        individually, but only once along a concatenated (and padded) array of all branches.
        This means for ncomps = [2,4] and normalized cum_branch_lens of [[0,1],[0,1]] we would
        interpolate xyz at the locations comp_ends = [[0,0.5,1], [0,0.25,0.5,0.75,1]],
        where 0 is the start of the branch and 1 is the end point at the full branch_len.
        To avoid do this in one go we set comp_ends = [0,0.5,1,2,2.25,2.5,2.75,3], and
        norm_cum_branch_len = [0,1,2,3] incrememting and also padding them by 1 to
        avoid overlapping branch_lens i.e. norm_cum_branch_len = [0,1,1,2] for only
        incrementing.
        """
        nodes_by_branches = self.nodes.groupby("global_branch_index")
        ncomps = nodes_by_branches["global_comp_index"].nunique().to_numpy()

        comp_ends = [
            np.linspace(0, 1, ncomp + 1) + 2 * i for i, ncomp in enumerate(ncomps)
        ]
        comp_ends = np.hstack(comp_ends)

        comp_ends = comp_ends.reshape(-1)
        cum_branch_lens = []
        for i, xyzr in enumerate(self.xyzr):
            branch_len = np.sqrt(np.sum(np.diff(xyzr[:, :3], axis=0) ** 2, axis=1))
            cum_branch_len = np.cumsum(np.concatenate([np.array([0]), branch_len]))
            max_len = cum_branch_len.max()
            # add padding like above
            cum_branch_len = cum_branch_len / (max_len if max_len > 0 else 1) + 2 * i
            cum_branch_len[np.isnan(cum_branch_len)] = 0
            cum_branch_lens.append(cum_branch_len)
        cum_branch_lens = np.hstack(cum_branch_lens)
        xyz = np.vstack(self.xyzr)[:, :3]
        xyz = v_interp(comp_ends, cum_branch_lens, xyz).T
        centers = (xyz[:-1] + xyz[1:]) / 2  # unaware of inter vs intra comp centers
        cum_ncomps = np.cumsum(ncomps)
        # this means centers between comps have to be removed here
        between_comp_inds = (cum_ncomps + np.arange(len(cum_ncomps)))[:-1]
        centers = np.delete(centers, between_comp_inds, axis=0)
        return centers

    def compute_compartment_centers(self):
        """Add compartment centers to nodes dataframe"""
        centers = self._compute_coords_of_comp_centers()
        self.base.nodes.loc[self._nodes_in_view, ["x", "y", "z"]] = centers

        # Estimate the branchpoint xyz as the mean of the xyz of all neighboring
        # compartments.
        for branchpoint in self.base._branchpoints.index:
            edges = self.base._comp_edges.copy()
            neighbors = edges[edges["sink"] == branchpoint]["source"]
            neighbor_xyz = self.base.nodes.loc[neighbors, ["x", "y", "z"]].mean()
            self.base._branchpoints.loc[branchpoint, ["x", "y", "z"]] = neighbor_xyz

    def _reformat_index(self, idx: Any, dtype: type = int) -> np.ndarray:
        """Transforms different types of indices into an array.

        Takes slice, list, array, ints, range and None and transforms
        it into array of indices. If index == "all" it returns "all"
        to be handled downstream.

        Args:
            idx: index that specifies at which locations to view the module.
            dtype: defaults to int, but can also reformat float for use in `loc`

        Returns:
            array of indices of shape (N,)"""
        if is_str_all(idx):  # also asserts that the only allowed str == "all"
            return idx

        if isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.number):
            np_dtype = idx.dtype.type
        else:
            np_dtype = np.dtype(int).type if dtype is int else np.dtype(float).type
        idx = np.array([], dtype=dtype) if idx is None else idx
        idx = np.array([idx]) if isinstance(idx, (dtype, np_dtype)) else idx
        idx = np.array(idx) if isinstance(idx, (list, range, pd.Index)) else idx

        idx = np.arange(len(self.base.nodes))[idx] if isinstance(idx, slice) else idx
        if idx.dtype == bool:
            shape = (*self.shape, len(self.edges))
            which_idx = len(idx) == np.array(shape)
            assert np.any(which_idx), "Index not matching num of cells/branches/comps."
            dim = shape[np.where(which_idx)[0][0]]
            idx = np.arange(dim)[idx]

            # Typically, `select` is run on `Module`, not on `View`. In these cases,
            # `nodes` will exactly the index of the `index` of the `self.nodes`
            # dataframe, and the line below is not needed. But if one wants to call
            # select multiple times in a chained way (e.g. when having multiple groups
            # and wanting to get their intersection, e.g., `net.exc.fast_spiking` or
            # `net.exc.soma`), the global index traced in `self.nodes.index` does no
            # longer match `nodes`. The line below translates the local index of
            # `nodes` to the global `self.nodes.index`.
            idx = self.nodes.index[idx].to_numpy()
        assert isinstance(idx, np.ndarray), "Invalid type"
        assert idx.dtype in [
            np_dtype,
            bool,
        ], f"Invalid dtype, found {str(idx.dtype)} instead of {str([np_dtype, bool])}"

        return idx.reshape(-1)

    def _set_controlled_by_param(self, key: str):
        """Determines which parameters are shared in `make_trainable`.

        Adds column to nodes/edges dataframes to read of shared params from.

        Args:
            key: key specifying group / view that is in control of the params."""
        if key in ["comp", "branch", "cell"]:
            self.nodes["controlled_by_param"] = self.nodes[f"global_{key}_index"]
            self.edges["controlled_by_param"] = 0
        elif key == "edge":
            self.edges["controlled_by_param"] = np.arange(len(self.edges))
        elif key == "filter":
            self.nodes["controlled_by_param"] = np.arange(len(self.nodes))
            self.edges["controlled_by_param"] = np.arange(len(self.edges))
        else:
            self.nodes["controlled_by_param"] = 0
            self.edges["controlled_by_param"] = 0
        self._current_view = key

    def select(
        self, nodes: np.ndarray = None, edges: np.ndarray = None, sorted: bool = False
    ) -> View:
        """Return View of the module filtered by specific node or edges indices.

        The selection is made based on the `index` of the `self.nodes` or `self.edges`,
        i.e., not on a local compartment index or a local row number (`loc`, not
        `iloc`).

        Args:
            nodes: indices of nodes to view. If None, all nodes are viewed.
            edges: indices of edges to view. If None, all edges are viewed.
            sorted: if True, nodes and edges are sorted.

        Returns:
            View for subset of selected nodes and/or edges."""
        nodes = self._reformat_index(nodes) if nodes is not None else None
        nodes = self._nodes_in_view if is_str_all(nodes) else nodes
        nodes = np.sort(nodes) if sorted else nodes

        edges = self._reformat_index(edges) if edges is not None else None
        edges = self._edges_in_view if is_str_all(edges) else edges
        edges = np.sort(edges) if sorted else edges

        view = View(self, nodes, edges)
        view._set_controlled_by_param("filter")
        return view

    def set_scope(self, scope: str):
        """Toggle between "global" or "local" scope.

        Determines if global or local indices are used for viewing the module.

        Args:
            scope: either "global" or "local"."""
        assert scope in ["global", "local"], "Invalid scope."
        self._scope = scope

    def scope(self, scope: str) -> View:
        """Return a View of the module with the specified scope.

        For example `cell.scope("global").branch(2).scope("local").comp(1)`
        will return the 1st compartment of branch 2.

        Args:
            scope: either "global" or "local".

        Returns:
            View with the specified scope."""
        view = self.view
        view.set_scope(scope)
        return view

    def _at_nodes(
        self, key: str, idx: Any, comp_edge_condition: str = "source_or_sink"
    ) -> View:
        """Return a View of the module filtering `nodes` by specified key and index.

        Args:
            key: Must be in {`cell`, `branch`, `comp`}. Determines which index is
                used to filter.
            idx: The indices to filter for.
            comp_edge_condition: Either of
                {`source_and_sink`, `source_or_sink`, `endpoint`, `startpoint`}. Sets
                how the `comp_edges` are built. If `source_and_sink`, an edge between
                compartments is kept only if source and sink compartments are within
                the view. If `source_or_sink`, an edge is kept if either the source
                or the sink are within the view. If `endpoint`, then the edge is kept
                if the compartment is in source or sink and if it is an edge between
                parent compartment and branchpoint. If `startpoint`, then the edge is
                kept if the compartment is in source or sink and if it is an edge
                between child compartment and branchpoint. This is used because we
                want different treatment of the `comp_edges` depending on whether we
                index with `.branch()` (`source_or_sink`), `.comp()`
                (`source_and_sink`), `.loc(0.0)` (`startpoint`), or `.loc(1.0)`
                (`endpoint`).
        """
        base_name = self.base.__class__.__name__
        assert self.base._has_childview(key), f"{base_name} does not support {key}."
        idx = self._reformat_index(idx)
        idx = self.nodes[self._scope + f"_{key}_index"] if is_str_all(idx) else idx
        where = self.nodes[self._scope + f"_{key}_index"].isin(idx)
        inds = self.nodes.index[where].to_numpy()

        view = View(self, nodes=inds, comp_edge_condition=comp_edge_condition)
        view._set_controlled_by_param(key)
        return view

    def _at_edges(self, key: str, idx: Any) -> View:
        """Return a View of the module filtering `edges` by specified key and index.

        Keys can be `pre`, `post`, `edge` and determine which index is used to filter.
        """
        idx = self._reformat_index(idx)
        idx = self.edges[self._scope + f"_{key}_index"] if is_str_all(idx) else idx
        where = self.edges[self._scope + f"_{key}_index"].isin(idx)
        inds = self.edges.index[where].to_numpy()

        view = View(self, edges=inds)
        view._set_controlled_by_param(key)
        return view

    def cell(self, idx: Any) -> View:
        """Return a View of the module at the selected cell(s).

        Args:
            idx: index of the cell to view.

        Returns:
            View of the module at the specified cell index."""
        return self._at_nodes("cell", idx)

    def branch(self, idx: Any) -> View:
        """Return a View of the module at the selected branches(s).

        Args:
            idx: index of the branch to view.

        Returns:
            View of the module at the specified branch index."""
        return self._at_nodes("branch", idx, comp_edge_condition="source_or_sink")

    def comp(self, idx: Any) -> View:
        """Return a View of the module at the selected compartments(s).

        Args:
            idx: index of the comp to view.

        Returns:
            View of the module at the specified compartment index."""
        return self._at_nodes("comp", idx, comp_edge_condition="source_and_sink")

    def edge(self, idx: Any) -> View:
        """Return a View of the module at the selected synapse edges(s).

        Args:
            idx: index of the edge to view.

        Returns:
            View of the module at the specified edge index."""
        return self._at_edges("edge", idx)

    def loc(self, at: Any) -> View:
        """Return a View of the module at the selected branch location(s).

        Args:
            at: location along the branch.

        Returns:
            View of the module at the specified branch location."""
        global_comp_idxs = []
        for i in self._branches_in_view:
            ncomp = self.base.ncomp_per_branch[i]
            comp_locs = np.linspace(0, 1, ncomp)
            at = comp_locs if is_str_all(at) else self._reformat_index(at, dtype=float)
            comp_edges = np.linspace(0, 1 + 1e-10, ncomp + 1)
            idx = np.digitize(at, comp_edges) - 1 + self.base.cumsum_ncomp[i]
            global_comp_idxs.append(idx)
        global_comp_idxs = np.concatenate(global_comp_idxs)
        orig_scope = self._scope
        # global scope needed to select correct comps, for i.e. branches w. ncomp=[1,2]
        # loc(0.9)  will correspond to different local branches (0 vs 1).

        if len(at) > 1:
            # If multiple locations are requested, then we interpret it just like
            # `.comp()`.
            comp_edge_condition = "source_and_sink"
        elif np.isclose(at, 0.0):
            comp_edge_condition = "startpoint"
        elif np.isclose(at, 1.0):
            comp_edge_condition = "endpoint"
        else:
            # For any `0 < at < 1`, we do not add any edges to branchpoints.
            comp_edge_condition = "source_and_sink"
        # This could also use `.comp(global_comp_idxs)` instead of
        # `._at_nodes("comp", global_comp_idxs)`, but this would force us to add
        # `comp_edge_condition` as an attribute to `.comp()`, which @michaeldeistler
        # found ugly (because it is user-facing).
        view = (
            self.scope("global")
            ._at_nodes(
                "comp", global_comp_idxs, comp_edge_condition=comp_edge_condition
            )
            .scope(orig_scope)
        )
        view._current_view = "loc"
        return view

    @property
    def _comps_in_view(self):
        """Lists the global compartment indices which are currently part of the view."""
        # method also exists in View. this copy forgoes need to instantiate a View
        return self.nodes["global_comp_index"].unique()

    @property
    def _branches_in_view(self):
        """Lists the global branch indices which are currently part of the view."""
        # method also exists in View. this copy forgoes need to instantiate a View
        return self.nodes["global_branch_index"].unique()

    @property
    def _cells_in_view(self):
        """Lists the global cell indices which are currently part of the view."""
        # method also exists in View. this copy forgoes need to instantiate a View
        return self.nodes["global_cell_index"].unique()

    def _iter_submodules(self, name: str):
        """Iterate over submoduleslevel.

        Used for `cells`, `branches`, `comps`."""
        col = self._scope + f"_{name}_index"
        idxs = self.nodes[col].unique()
        for idx in idxs:
            yield self._at_nodes(name, idx)

    @property
    def cells(self):
        """Iterate over all cells in the module.

        Returns a generator that yields a View of each cell."""
        yield from self._iter_submodules("cell")

    @property
    def branches(self):
        """Iterate over all branches in the module.

        Returns a generator that yields a View of each branch."""
        yield from self._iter_submodules("branch")

    @property
    def comps(self):
        """Iterate over all compartments in the module.
        Can be called on any module, i.e. `net.comps`, `cell.comps` or
        `branch.comps`. `__iter__` does not allow for this.

        Returns a generator that yields a View of each compartment."""
        yield from self._iter_submodules("comp")

    def __iter__(self):
        """Iterate over parts of the module.

        Internally calls `cells`, `branches`, `comps` at the appropriate level.

        Example usage
        ^^^^^^^^^^^^^

        .. code-block:: python

            for cell in network:
                for branch in cell:
                    for comp in branch:
                        print(comp.nodes.shape)
        """
        next_level = self._childviews()[0]
        yield from self._iter_submodules(next_level)

    @property
    def shape(self) -> Tuple[int]:
        """Returns the number of submodules contained in a module.

        .. code-block:: python

            network.shape = (num_cells, num_branches, num_compartments)
            cell.shape = (num_branches, num_compartments)
            branch.shape = (num_compartments,)
        """
        cols = ["global_cell_index", "global_branch_index", "global_comp_index"]
        raw_shape = self.nodes[cols].nunique().to_list()

        # ensure (net.shape -> dim=3, cell.shape -> dim=2, branch.shape -> dim=1, comp.shape -> dim=0)
        levels = ["network", "cell", "branch", "comp"]
        module = self.base.__class__.__name__.lower()
        module = "comp" if module == "compartment" else module
        shape = tuple(raw_shape[levels.index(module) :])
        return shape

    def copy(
        self, reset_index: bool = False, as_module: bool = False
    ) -> Union[Module, View]:
        """Extract part of a module and return a copy of its View or a new module.

        This can be used to call `jx.integrate` on part of a Module.

        Args:
            reset_index: if True, the indices of the new module are reset to start from 0.
            as_module: if True, a new module is returned instead of a View.

        Returns:
            A part of the module or a copied view of it."""
        view = deepcopy(self)
        warnings.warn("This method is experimental, use at your own risk.")
        # TODO FROM #447: add reset_index, i.e. for parents, nodes, edges etc. such that they
        # start from 0/-1 and are contiguous
        if as_module:
            raise NotImplementedError("Not yet implemented.")
            # initialize a new module with the same attributes
        return view

    @property
    def view(self):
        """Return view of the module."""
        return View(self, self._nodes_in_view, self._edges_in_view)

    @property
    def _module_type(self):
        """Return type of the module (compartment, branch, cell, network) as string.

        This is used to perform asserts for some modules (e.g. network cannot use
        `set_ncomp`) without having to import the module in `base.py`."""
        return self.__class__.__name__.lower()

    def _append_params_and_states(self, param_dict: Dict, state_dict: Dict):
        """Insert the default params of the module (e.g. radius, length).

        This is run at `__init__()`. It does not deal with channels.
        """
        for param_name, param_value in param_dict.items():
            self.base.nodes[param_name] = param_value
        for state_name, state_value in state_dict.items():
            self.base.nodes[state_name] = state_value

    def _gather_channels_from_constituents(self, constituents: List):
        """Modify `self.channels` and `self.nodes` with channel info from constituents.

        This is run at `__init__()`. It takes all branches of constituents (e.g.
        of all branches when the are assembled into a cell) and adds columns to
        `.nodes` for the relevant channels.
        """
        for module in constituents:
            assert len(module.diffusion_states) == 0, (
                "Cannot have diffusion in subparts of a module. As a workaround, set "
                "the diffusion constant for all parts that should not have ion "
                "diffusion to a very small value (e.g. 1e-8)."
            )
            for channel in module.channels:
                if channel._name not in [c._name for c in self.channels]:
                    self.base.channels.append(channel)
                if channel.current_name not in self.membrane_current_names:
                    self.base.membrane_current_names.append(channel.current_name)
            for pump in module.pumps:
                if pump._name not in [c._name for c in self.pumps]:
                    self.base.pumps.append(pump)
                if pump.current_name not in self.membrane_current_names:
                    self.base.membrane_current_names.append(pump.current_name)
            for group in module.group_names:
                if group not in self.base.group_names:
                    self.base.group_names.append(group)

        # Setting columns of channel and pump names to `False` instead of `NaN`.
        for channel in self.base.channels + self.base.pumps:
            name = channel._name
            self.base.nodes.loc[self.nodes[name].isna(), name] = False

    @only_allow_module
    def to_jax(self):
        # TODO FROM #447: Make this work for View?
        """Move `.nodes` to `.jaxnodes`.

        Before the actual simulation is run (via `jx.integrate`), all parameters of
        the `jx.Module` are stored in `.nodes` (a `pd.DataFrame`). However, for
        simulation, these parameters have to be moved to be `jnp.ndarrays` such that
        they can be processed on GPU/TPU and such that the simulation can be
        differentiated. `.to_jax()` copies the `.nodes` to `.jaxnodes`.
        """
        self.base.jaxnodes = {}
        for key, value in self.base.nodes.to_dict(orient="list").items():
            # inds = jnp.arange(len(value))
            values = -1 * jnp.ones((self._n_nodes))
            values = values.at[self.base.nodes.index.to_numpy()].set(value)
            self.base.jaxnodes[key] = values

        # `jaxedges` contains only parameters (no indices).
        # `jaxedges` contains only non-Nan elements. This is unlike the channels where
        # we allow parameter sharing.
        self.base.jaxedges = {}
        edges = self.base.edges.to_dict(orient="list")
        for i, synapse in enumerate(self.base.synapses):
            condition = np.asarray(edges["type_ind"]) == i
            for key in synapse.synapse_params:
                self.base.jaxedges[key] = jnp.asarray(np.asarray(edges[key])[condition])
            for key in synapse.synapse_states:
                self.base.jaxedges[key] = jnp.asarray(np.asarray(edges[key])[condition])

    def show(
        self,
        param_names: Optional[Union[str, List[str]]] = None,
        *,
        indices: bool = True,
        params: bool = True,
        states: bool = True,
        channel_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Print detailed information about the Module or a view of it.

        Args:
            param_names: The names of the parameters to show. If `None`, all parameters
                are shown.
            indices: Whether to show the indices of the compartments.
            params: Whether to show the parameters of the compartments.
            states: Whether to show the states of the compartments.
            channel_names: The names of the channels to show. If `None`, all channels are
                shown.

        Returns:
            A `pd.DataFrame` with the requested information.
        """
        nodes = self.nodes.copy()  # prevents this from being edited

        cols = []
        inds = ["comp_index", "branch_index", "cell_index"]
        scopes = ["local", "global"]
        inds = [f"{s}_{i}" for i in inds for s in scopes] if indices else []
        cols += inds
        cols += [ch._name for ch in self.channels] if channel_names else []
        cols += (
            sum([list(ch.channel_params) for ch in self.channels], []) if params else []
        )
        cols += (
            sum([list(ch.channel_states) for ch in self.channels], []) if states else []
        )

        if not param_names is None:
            cols = (
                inds + [c for c in cols if c in param_names]
                if params
                else list(param_names)
            )

        return nodes[cols]

    @only_allow_module
    def _init_solvers(self, allowed_nodes_per_level: Optional[int] = None):
        """Initialize the morphology such that it can be processed by the solvers.

        Args:
            allowed_nodes_per_level: Only relevant to the `jaxley.dhs` solver. It sets
                how many nodes are visited before the level is increased, even if the
                number of hops did not change. This sets the amount of parallelism
                of the simulation.
        """
        self._init_solver_jax_spsolve()
        self._init_solver_jaxley_dhs_solve(
            allowed_nodes_per_level=allowed_nodes_per_level
        )
        self.initialized_solver = True

    def _init_solver_jax_spsolve(self):
        """Initialize morphology for the jax sparse voltage solver.

        Explanation of `self._comp_eges['type']`:
        `type == 0`: compartment <--> compartment (within branch)
        `type == 1`: branchpoint --> parent-compartment
        `type == 2`: branchpoint --> child-compartment
        `type == 3`: parent-compartment --> branchpoint
        `type == 4`: child-compartment --> branchpoint
        """
        data_inds, indices, indptr = comp_edges_to_indices(self._comp_edges)
        self._data_inds = data_inds
        self._indices_jax_spsolve = indices
        self._indptr_jax_spsolve = indptr

    def _init_solver_jaxley_dhs_solve(
        self, allowed_nodes_per_level: Optional[int] = None, root: int = 0
    ) -> None:
        """Create module attributes for indexing with the `jaxley.dhs` voltage volver.

        This function first generates the networkX `comp_graph`, then traverses it
        to identify the solve order, and then pre-computes the relevant attributes used
        for re-ordering compartments during the voltage solve with `jaxley.dhs`.

        This base-method is used by `jx.Compartment`, `jx.Branch`, and `jx.Cell`.
        The `jx.Network` implements its own method.

        Args:
            allowed_nodes_per_level: How many nodes are visited before the level is
                increased, even if the number of hops did not change. This sets the
                amount of parallelism of the simulation. Setting this value to 1
                automatically sets `self._solver_device` to `cpu`, and setting it to
                values larger than 1 automatically sets `self._solver_device` to `gpu`.
            root: The root node from which to start tracing.
        """
        # Infer the amount of parallelism of the solver. Note that the `jaxley.dhs.cpu`
        # requires `allowed_nodes_per_level = 1`, or you have to run the following
        # after having initialized the module (in order to fill up all
        # `node_order_grouped` to be of the same shape):
        #
        # ```
        # nodes_and_parents = self._dhs_solve_indexer["node_order_grouped"]
        # padded_stack = np.full((len(nodes_and_parents), allowed_nodes_per_level, 2), -1)
        # for idx, arr in enumerate(nodes_and_parents):
        #     padded_stack[idx, : arr.shape[0], :] = arr
        # self._dhs_solve_indexer["node_order_grouped"] = padded_stack
        # ```
        #
        if allowed_nodes_per_level is None:
            if self._solver_device == "cpu":
                allowed_nodes_per_level = 1
            else:
                allowed_nodes_per_level = 32
        else:
            self._solver_device = "cpu" if allowed_nodes_per_level == 1 else "gpu"

        if np.any(np.isnan(self.xyzr[0][:, :3])):
            self.compute_xyz()
            self.compute_compartment_centers()
        comp_graph = to_graph(self)

        # Export to graph and traverse it to identify the solve order.
        node_order, node_to_solve_index_mapping = dhs_solve_index(
            comp_graph, allowed_nodes_per_level=allowed_nodes_per_level, root=root
        )

        # Set the order in which compartments are processed during Dendritic Hierarchical
        # Scheduling (DHS). The `_dhs_node_order` contains edges between compartments,
        # the values correspond to compartment indices.
        dhs_node_order = np.asarray(node_order[1:])

        # We have to change the order of compartments at every time step of the solve.
        # Because of this, we make it as efficient as possible to perform this ordering
        # with the arrays below. Example:
        # ```
        # voltages = voltages[mapping_array]  # Permute `voltages` to solve order.
        # voltages = voltages[inv_mapping_array]  # Permute back to compartment order.
        # ```
        map_dict = node_to_solve_index_mapping  # Abbreviation.
        inv_mapping_array = np.array([map_dict[i] for i in sorted(map_dict)])
        mapping_array = np.argsort(inv_mapping_array)
        #
        self._dhs_solve_indexer = {}
        self._dhs_solve_indexer["map_dict"] = map_dict
        self._dhs_solve_indexer["inv_map_to_solve_order"] = inv_mapping_array
        self._dhs_solve_indexer["map_to_solve_order"] = mapping_array

        # Define the matrix permutation for DHS.
        lower_and_upper_inds = np.arange((self._n_nodes - 1) * 2)
        lower_and_upper_inds, new_node_order = dhs_permutation_indices(
            lower_and_upper_inds,
            self._off_diagonal_inds,
            dhs_node_order,
            self._dhs_solve_indexer["map_dict"],
        )

        # Concatenate a `0` such that the `lower` and `upper` will have the same
        # shape as the `diag` and `solve`. The 0-eth element will never actually be
        # accessed, but it makes indexing easier in the voltage solver.
        #
        # Here, we assume that `comp_edges` has lowers first and uppers only after that
        # (by using `[:self._n_nodes-1]`). TODO we should make this more robust in the
        # future as we move towards simulating _any_ graph.
        self._dhs_solve_indexer["map_to_solve_order_lower"] = jnp.concatenate(
            [
                jnp.asarray([0]).astype(int),
                lower_and_upper_inds.astype(int)[: self._n_nodes - 1],
            ]
        )
        self._dhs_solve_indexer["map_to_solve_order_upper"] = jnp.concatenate(
            [
                jnp.asarray([0]).astype(int),
                lower_and_upper_inds.astype(int)[self._n_nodes - 1 :],
            ]
        )
        self._dhs_solve_indexer["node_order"] = new_node_order
        self._dhs_solve_indexer["node_order_grouped"] = dhs_group_comps_into_levels(
            new_node_order
        )

        # Define a simple lookup table that allows to retrieve the parent of a node.
        # E.g.:
        # ```parent_node = parents[node]``` or:
        # ```two_step_parent = parents[parents[node]]```.
        parents = -1 * np.ones(self._n_nodes + 1)
        for nodes in self._dhs_solve_indexer["node_order_grouped"]:
            parents[nodes[:, 0]] = nodes[:, 1]
        self._dhs_solve_indexer["parent_lookup"] = parents.astype(int)

    def set(self, key: str, val: Union[float, jnp.ndarray]):
        """Set parameter of module (or its view) to a new value.

        Note that this function can not be called within `jax.jit` or `jax.grad`.
        Instead, it should be used set the parameters of the module **before** the
        simulation. Use `.data_set()` to set parameters during `jax.jit` or
        `jax.grad`.

        Args:
            key: The name of the parameter to set.
            val: The value to set the parameter to. If it is `jnp.ndarray` then it
                must be of shape `(len(num_compartments))`.
        """
        if key in [f"axial_diffusion_{ion_name}" for ion_name in self.diffusion_states]:
            assert val > 0, (
                f"You are trying to set `{key}` to `{val}`. "
                f"We only allow strictly positive values for the "
                f"diffusion. Zero is not allowed either, but you can use very small "
                f"values (e.g. 1e-8)."
            )

        if key in self.nodes.columns:
            not_nan = ~self.nodes[key].isna().to_numpy()
            rows = self._nodes_in_view[not_nan]
            self.base.nodes.loc[rows, key] = val

            # When the key is `radius` or `length`, we also have to update the
            # membrane surface area. In principle, we could also do this on the fly
            # when a simulation is started, but computing the membrane area for
            # SWC-traced neurons can be computationally expensive.
            if key in ["radius", "length"]:
                # Add an additional warning if the neuron was read from SWC.
                xyzr = np.concatenate(self.xyzr)
                xyzr_is_available = np.invert(np.any(np.isnan(xyzr[:, 3])))
                if xyzr_is_available:
                    warn(
                        f"You are modifying the {key} of a neuron that was read "
                        f"from an SWC file. By doing this, Jaxley recomputes the "
                        f"membrane surface area as `A = 2 * pi * r * l`. "
                        f"This formula differs from the formula used by the SWC "
                        f"reader, which takes the exact positions and radiuses of "
                        f"SWC-traced points into account. Because of this, even "
                        f"statements such as `cell.set('{key}', cell.nodes.{key})` "
                        f"will likely change the electrophysiology of the cell."
                    )
                # If radius and length are updated by the pstate, then we have to also
                # update 1) area, 2) volume, and 3) resistive_loads.
                l = self.base.nodes["length"]
                r = self.base.nodes["radius"]
                # l/2 because we want the input load (left half of the cylinder) and
                # the output load (right half of the cylinder).
                resistive_load = cylinder_resistive_load(l / 2, r)
                self.base.nodes.loc[rows, "area"] = cylinder_area(l, r)
                self.base.nodes.loc[rows, "volume"] = cylinder_volume(l, r)
                self.base.nodes.loc[rows, "resistive_load_out"] = resistive_load
                self.base.nodes.loc[rows, "resistive_load_in"] = resistive_load
        elif key in self.edges.columns:
            not_nan = ~self.edges[key].isna().to_numpy()
            self.base.edges.loc[self._edges_in_view[not_nan], key] = val
        else:
            raise KeyError(f"Key '{key}' not found in nodes or edges")

    def data_set(
        self,
        key: str,
        val: Union[float, jnp.ndarray],
        param_state: Optional[List[Dict]],
    ):
        """Set parameter of module (or its view) to a new value within `jit`.

        Args:
            key: The name of the parameter to set.
            val: The value to set the parameter to. If it is `jnp.ndarray` then it
                must be of shape `(len(num_compartments))`.
            param_state: State of the set parameters, internally used such that this
                function does not modify global state.
        """
        # Note: `data_set` does not support arrays for `val`.
        is_node_param = key in self.nodes.columns
        data = self.nodes if is_node_param else self.edges
        viewed_inds = self._nodes_in_view if is_node_param else self._edges_in_view
        if key in data.columns:
            not_nan = ~data[key].isna()
            indices = jnp.asarray(viewed_inds[not_nan]).reshape(
                -1, 1
            )  # shape (n_comp, 1)
            val = jnp.broadcast_to(
                jnp.asarray(val), (indices.shape[0],)
            )  # shape (n_comp,)
            added_param_state = [
                {
                    "indices": indices,
                    "key": key,
                    "val": val,
                }
            ]
            if param_state is not None:
                param_state += added_param_state
            else:
                param_state = added_param_state
        else:
            raise KeyError("Key not recognized.")
        return param_state

    def set_ncomp(
        self, ncomp: int, min_radius: Optional[float] = None, initialize: bool = True
    ):
        """Set the number of compartments with which the branch is discretized.

        Args:
            ncomp: The number of compartments that the branch should be discretized
                into.
            min_radius: Only used if the morphology was read from an SWC file. If passed
                the radius is capped to be at least this value.
            initialize: If `False`, it skips the initialization stage and the user
                has to run it manually afterwards. This is useful when `set_ncomp`
                is run in a loop (e.g. for the d_lambda rule), where one can
                initialize only once after the entire loop to largely speed up
                computation time. If `False`, then the user has to run
                `cell.initialize()` manually afterwards.

        Raises:
            - When there are stimuli in any compartment in the module.
            - When there are recordings in any compartment in the module.
            - When the channels of the compartments are not the same within the branch
            that is modified.
            - When the lengths of the compartments are not the same within the branch
            that is modified.
            - When the branch that is modified has compartments belonging to different
            groups.
            - Unless the morphology was read from an SWC file, when the radiuses of the
            compartments are not the same within the branch that is modified.
        """
        assert len(self.base.externals) == 0, "No stimuli allowed!"
        assert len(self.base.recordings) == 0, "No recordings allowed!"
        assert len(self.base.trainable_params) == 0, "No trainables allowed!"

        assert self.base._module_type != "network", "This is not allowed for networks."
        assert not (
            self.base._module_type == "cell"
            and (
                len(self._branches_in_view) == len(self.base._branches_in_view)
                and len(self.base._branches_in_view) > 1
            )
        ), "This is not allowed for a `cell`, use `cell.branch(i)` instead."

        # Update all attributes that are affected by compartment structure.
        view = self.nodes.copy()
        all_nodes = self.base.nodes
        start_idx = self.nodes["global_comp_index"].to_numpy()[0]
        ncomp_per_branch = self.base.ncomp_per_branch
        channel_names = [c._name for c in self.base.channels]
        channel_param_names = list(
            chain(*[c.channel_params for c in self.base.channels])
        )
        channel_state_names = list(
            chain(*[c.channel_states for c in self.base.channels])
        )

        within_branch_radiuses = view["radius"].to_numpy()
        compartment_lengths = view["length"].to_numpy()
        num_previous_ncomp = len(within_branch_radiuses)
        branch_indices = pd.unique(view["global_branch_index"])

        xyzr = self.base.xyzr[branch_indices[0]]
        xyzr_is_available = np.invert(np.any(np.isnan(xyzr[:, 3])))

        assert len(branch_indices) <= 1, "You can only modify ncomp of a single branch."

        error_msg = lambda name: (
            f"You previously modified the {name} of individual compartments, but "
            f"now you are modifying the number of compartments in this branch. "
            f"This is not allowed. First build the morphology with `set_ncomp()` and "
            f"then modify the radiuses and lengths of compartments."
        )

        if (
            ~np.all(within_branch_radiuses == within_branch_radiuses[0])
            and not xyzr_is_available
        ):
            raise ValueError(error_msg("radius"))

        for property_name in ["length", "capacitance", "axial_resistivity"]:
            compartment_properties = view[property_name].to_numpy()
            if ~np.all(compartment_properties == compartment_properties[0]):
                raise ValueError(error_msg(property_name))

        if (
            num_previous_ncomp > 1
            and not (self.nodes[channel_names].var() == 0.0).all()
        ):
            raise ValueError(
                "Some channel exists only in some compartments of the branch which you "
                "are trying to modify. This is not allowed. First specify the number "
                "of compartments with `.set_ncomp()` and then insert the channels "
                "accordingly."
            )

        if (
            num_previous_ncomp > 1
            and not (
                self.nodes[channel_param_names + channel_state_names].var() == 0.0
            ).all()
        ):
            raise ValueError(
                "Some channel has different parameters or states between the "
                "different compartments of the branch which you are trying to modify. "
                "This is not allowed. First specify the number of compartments with "
                "`.set_ncomp()` and then insert the channels accordingly."
            )

        for group_name in self.group_names:
            group_ncomp = view[group_name].sum()
            assert group_ncomp == 0 or group_ncomp == num_previous_ncomp, (
                f"{group_ncomp} compartments within the branch are part of the "
                f"group '{group_name}', but the other "
                f"{num_previous_ncomp - group_ncomp} compartments are not. This "
                f"is not allowed: Every compartment must belong to the same group for "
                f"`.set_ncomp()` to work."
            )

        # Add new rows as the average of all rows. Special case for the length is below.
        start_index = int(self.nodes.index.to_numpy()[0])
        average_row = self.nodes.mean(skipna=False, numeric_only=False)
        average_row = pd.DataFrame([average_row])
        view = pd.concat([average_row] * ncomp, axis="rows", ignore_index=True)

        # Set the correct datatype after having performed an average which cast
        # everything to float.
        integer_cols = ["global_cell_index", "global_branch_index", "global_comp_index"]
        view[integer_cols] = view[integer_cols].astype(int)

        # Whether or not a channel or group exists in a compartment is a boolean.
        boolean_cols = channel_names + self.base.group_names
        view[boolean_cols] = view[boolean_cols].astype(bool)

        # Special treatment for the lengths and radiuses. These are not being set as
        # the average because we:
        # 1) Want to maintain the total length of a branch.
        # 2) Want to use the SWC inferred radius.
        #
        # Compute new compartment lengths.
        comp_lengths = np.sum(compartment_lengths) / ncomp
        view["length"] = comp_lengths

        # Compute new compartment radiuses.
        if xyzr_is_available:
            # If all xyzr-radiuses of the branch are available, then use them to
            # compute the new compartment radiuses.
            comp_xyzrs = split_xyzr_into_equal_length_segments(xyzr, ncomp)
            morph_attrs = np.asarray(
                [morph_attrs_from_xyzr(xyzr, min_radius, ncomp) for xyzr in comp_xyzrs]
            )
            view["radius"] = morph_attrs[:, 0]
            view["area"] = morph_attrs[:, 1]
            view["volume"] = morph_attrs[:, 2]
            view["resistive_load_in"] = morph_attrs[:, 3]
            view["resistive_load_out"] = morph_attrs[:, 4]
        else:
            view["radius"] = within_branch_radiuses[0] * np.ones(ncomp)
            l = comp_lengths
            r = within_branch_radiuses[0]
            # l/2 because we want the input load (left half of the cylinder) and
            # the output load (right half of the cylinder).
            resistive_load = cylinder_resistive_load(l / 2, r)
            view["area"] = cylinder_area(l, r)
            view["volume"] = cylinder_volume(l, r)
            view["resistive_load_out"] = resistive_load
            view["resistive_load_in"] = resistive_load

        # Update `.nodes`.
        # 1) Delete N rows starting from start_idx
        number_deleted = num_previous_ncomp
        all_nodes = all_nodes.drop(index=range(start_idx, start_idx + number_deleted))

        # 2) Insert M new rows at the same location
        df1 = all_nodes.iloc[:start_idx]  # Rows before the insertion point
        df2 = all_nodes.iloc[start_idx:]  # Rows after the insertion point

        # 3) Combine the parts: before, new rows, and after
        view.index = np.arange(len(view)).astype(int) + start_index
        df2.index -= num_previous_ncomp
        df2.index += ncomp
        all_nodes = pd.concat([df1, view, df2])

        # Override `comp_index` to just be a consecutive list.
        all_nodes["global_comp_index"] = np.arange(len(all_nodes))

        # Update compartment structure arguments.
        ncomp_per_branch[branch_indices] = ncomp
        ncomp = int(np.max(ncomp_per_branch))
        cumsum_ncomp = cumsum_leading_zero(ncomp_per_branch)
        internal_node_inds = np.arange(cumsum_ncomp[-1])

        self.base.nodes = all_nodes
        self.base.ncomp_per_branch = ncomp_per_branch
        self.base.ncomp = ncomp
        self.base.cumsum_ncomp = cumsum_ncomp
        self.base._internal_node_inds = internal_node_inds

        # Update the morphology indexing (e.g., `.comp_edges`).
        if initialize:
            self.base.initialize()

    def make_trainable(
        self,
        key: str,
        init_val: Optional[Union[float, list]] = None,
        verbose: bool = True,
    ):
        """Make a parameter trainable.

        If a parameter is made trainable, it will be returned by `get_parameters()`
        and should then be passed to `jx.integrate(..., params=params)`.

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
        if key in ["radius", "length"]:
            # Add an additional warning if the neuron was read from SWC.
            xyzr = np.concatenate(self.xyzr)
            xyzr_is_available = np.invert(np.any(np.isnan(xyzr[:, 3])))
            if xyzr_is_available:
                warn(
                    f"You are making trainable the {key} of a neuron that was read "
                    f"from an SWC file. By doing this, Jaxley recomputes the "
                    f"membrane surface area as `A = 2 * pi * r * l`. "
                    f"This formula differs from the formula used by the SWC "
                    f"reader, which takes the exact positions and radiuses of "
                    f"SWC-traced points into account. Because of this, "
                    f"statements such as `cell.make_trainable('{key}')` "
                    f"will likely change the electrophysiology of the cell, even if "
                    f"the trainable parameters were not modified."
                )

        assert self.allow_make_trainable, (
            "network.cell('all').make_trainable() is not supported. Use a "
            "for-loop over cells."
        )

        data = self.nodes if key in self.nodes.columns else None
        data = self.edges if key in self.edges.columns else data

        assert data is not None, f"Key '{key}' not found in nodes or edges"
        not_nan = ~data[key].isna()
        data = data.loc[not_nan].copy()
        assert (
            len(data) > 0
        ), "No settable parameters found in the selected compartments."

        grouped_view = data.groupby("controlled_by_param")
        # Because of this `x.index.values` we cannot support `make_trainable()` on
        # the module level for synapse parameters (but only for `SynapseView`).
        comp_inds = list(
            grouped_view.apply(lambda x: x.index.values, include_groups=False)
        )

        # check if all shapes in comp_inds are the same. If not the case this means
        # the groups in controlled_by_param have different sizes, i.e. due to different
        # number of comps for two different branches. In this case we pad the smaller
        # groups with -1 to make them the same size.
        lens = np.array([inds.shape[0] for inds in comp_inds])
        max_len = np.max(lens)
        pad = lambda x: np.pad(x, (0, max_len - x.shape[0]), constant_values=-1)
        if not np.all(lens == max_len):
            comp_inds = [
                pad(inds) if inds.shape[0] < max_len else inds for inds in comp_inds
            ]

        # Sorted inds are only used to infer the correct starting values.
        indices_per_param = jnp.stack(comp_inds)

        # Assign dummy param (ignored by nanmean later). This adds a new row to the
        # `data` (which is, e.g., self.nodes). That new row has index `-1`, which does
        # not clash with any other node index (they are in
        # `[0, ..., num_total_comps-1]`).
        data.loc[-1, key] = np.nan
        param_vals = jnp.asarray([data.loc[inds, key].to_numpy() for inds in comp_inds])

        # Set the value which the trainable parameter should take.
        num_created_parameters = len(indices_per_param)
        if init_val is not None:
            if isinstance(init_val, float):
                new_params = jnp.asarray([init_val] * num_created_parameters)
            elif isinstance(init_val, list):
                assert len(init_val) == num_created_parameters, (
                    f"len(init_val)={len(init_val)}, but trying to create "
                    f"{num_created_parameters} parameters."
                )
                new_params = jnp.asarray(init_val)
            else:
                raise ValueError(
                    f"init_val must a float, list, or None, but it is a "
                    f"{type(init_val).__name__}."
                )
        else:
            new_params = jnp.nanmean(param_vals, axis=1)
        self.base.trainable_params.append({key: new_params})
        self.base.indices_set_by_trainables.append(indices_per_param)
        self.base.num_trainable_params += num_created_parameters
        if verbose:
            print(
                f"Number of newly added trainable parameters: "
                f"{num_created_parameters}. Total number of trainable "
                f"parameters: {self.base.num_trainable_params}"
            )

    def write_trainables(self, trainable_params: List[Dict[str, jnp.ndarray]]):
        """Write the trainables into `.nodes` and `.edges`.

        This allows to, e.g., visualize trained networks with `.vis()`.

        Args:
            trainable_params: The trainable parameters returned by `get_parameters()`.
        """
        # We do not support views. Why? `jaxedges` does not have any NaN
        # elements, whereas edges does. Because of this, we already need special
        # treatment to make this function work, and it would be an even bigger hassle
        # if we wanted to support this.
        assert self.__class__.__name__ in [
            "Compartment",
            "Branch",
            "Cell",
            "Network",
        ], "Only supports modules."

        # We could also implement this without casting the module to jax.
        # However, I think it allows us to reuse as much code as possible and it avoids
        # any kind of issues with indexing or parameter sharing (as this is fully
        # taken care of by `get_all_parameters()`).
        self.base.to_jax()
        pstate = params_to_pstate(trainable_params, self.base.indices_set_by_trainables)
        all_params = self.base.get_all_parameters(pstate)

        # The value for `delta_t` does not matter here because it is only used to
        # compute the initial current. However, the initial current cannot be made
        # trainable and so its value never gets used below.
        all_states = self.base.get_all_states(pstate, all_params, delta_t=0.025)

        # Loop only over the keys in `pstate` to avoid unnecessary computation.
        for parameter in pstate:
            key = parameter["key"]
            if key in self.base.nodes.columns:
                vals_to_set = all_params if key in all_params.keys() else all_states
                self.base.set(key, vals_to_set[key][self._internal_node_inds])

        # `jaxedges` contains only non-Nan elements. This is unlike the channels where
        # we allow parameter sharing.
        edges = self.base.edges.to_dict(orient="list")
        for i, synapse in enumerate(self.base.synapses):
            condition = np.asarray(edges["type_ind"]) == i
            for key in list(synapse.synapse_params.keys()):
                self.base.edges.loc[condition, key] = all_params[key]
            for key in list(synapse.synapse_states.keys()):
                self.base.edges.loc[condition, key] = all_states[key]

    @deprecated(
        "0.11.0",
        (
            " Instead, please use, e.g., "
            "`jx.morphology_utils.distance(cell[0, 0], cell[2, 1], kind='direct')`. "
            "Note that, unlike `cell[0, 0].distance(cell[2, 1]), that "
            "function returns a list of distances (to all endpoints)."
        ),
    )
    def distance(self, endpoint: "View") -> float:
        """Return the direct distance between two compartments.

        This function computes the direct distance. To compute the pathwise distance,
        use `distance_pathwise()`.

        Args:
            endpoint: The compartment to which to compute the distance to.
        """
        assert len(self.xyzr) == 1 and len(endpoint.xyzr) == 1
        start_xyz = jnp.mean(self.xyzr[0][:, :3], axis=0)
        end_xyz = jnp.mean(endpoint.xyzr[0][:, :3], axis=0)
        return jnp.sqrt(jnp.sum((start_xyz - end_xyz) ** 2))

    def delete_trainables(self):
        """Removes all trainable parameters from the module."""

        if isinstance(self, View):
            trainables_and_inds = self._filter_trainables(is_viewed=False)
            self.base.indices_set_by_trainables = trainables_and_inds[0]
            self.base.trainable_params = trainables_and_inds[1]
            self.base.num_trainable_params -= self.num_trainable_params
        else:
            self.base.indices_set_by_trainables = []
            self.base.trainable_params = []
            self.base.num_trainable_params = 0
        self._update_view()

    def add_to_group(self, group_name: str):
        """Add a view of the module to a group.

        Groups can then be indexed. For example:

        .. code-block:: python

            net.cell(0).add_to_group("excitatory")
            net.excitatory.set("radius", 0.1)

        Args:
            group_name: The name of the group.
        """
        if group_name not in self.base.group_names:
            channel_names = [channel._name for channel in self.base.channels]
            assert group_name not in channel_names, (
                "Trying to create a group with the same name as one of the channels. "
                "This is not supported. Choose a different name for the group."
            )
            self.base.group_names.append(group_name)
            self.base.nodes[group_name] = False
            self.base.nodes.loc[self._nodes_in_view, group_name] = True
        else:
            self.base.nodes.loc[self._nodes_in_view, group_name] = True

    def _get_state_names(self) -> Tuple[List, List]:
        """Collect all recordable / clampable states in the membrane and synapses.

        Returns states separated by comps and edges."""
        channel_states = [
            name for c in self.channels + self.pumps for name in c.channel_states
        ]
        synapse_states = [
            name for s in self.synapses if s is not None for name in s.synapse_states
        ]
        membrane_states = ["v", "i"] + self.membrane_current_names
        return (
            channel_states + membrane_states,
            synapse_states + self.synapse_current_names,
        )

    def get_parameters(self) -> List[Dict[str, jnp.ndarray]]:
        """Get all trainable parameters.

        The returned parameters should be passed to
        ``jx.integrate(..., params=params)``.

        Returns:
            A list of all trainable parameters in the form of
                [{"gNa": jnp.array([0.1, 0.2, 0.3])}, ...].
        """
        return self.trainable_params

    @only_allow_module
    def get_all_parameters(self, pstate: List[Dict]) -> Dict[str, jnp.ndarray]:
        # TODO FROM #447: MAKE THIS WORK FOR VIEW?
        """Return all parameters (and coupling conductances) needed to simulate.

        Runs `compute_axial_conductances()` and returns every parameter that is needed
        to solve the ODE. This includes conductances, radiuses, lengths,
        axial_resistivities, but also coupling conductances.

        This is done by first obtaining the current value of every parameter (not only
        the trainable ones) and then replacing the trainable ones with the value
        in `trainable_params()`. This function is run within `jx.integrate()`.

        pstate can be obtained by calling `params_to_pstate()`.

        .. code-block:: python

            params = module.get_parameters() # i.e. [0, 1, 2]
            pstate = params_to_pstate(params, module.indices_set_by_trainables)
            module.to_jax() # needed for call to module.jaxnodes

        Args:
            pstate: The state of the trainable parameters. pstate takes the form
                [{
                    "key": "gNa", "indices": jnp.array([0, 1, 2]),
                    "val": jnp.array([0.1, 0.2, 0.3])
                }, ...].
            voltage_solver: The voltage solver that is used. Since `jax.sparse` and
                `jaxley.xyz` require different formats of the axial conductances, this
                function will default to different building methods.

        Returns:
            A dictionary of all module parameters.
        """
        params = {}
        for key in [
            "radius",
            "length",
            "axial_resistivity",
            "capacitance",
            "area",
            "volume",
            "resistive_load_out",
            "resistive_load_in",
        ]:
            params[key] = self.base.jaxnodes[key]

        for key in self.diffusion_states:
            params[f"axial_diffusion_{key}"] = self.jaxnodes[f"axial_diffusion_{key}"]

        for channel in self.base.channels + self.base.pumps:
            for channel_params in channel.channel_params:
                params[channel_params] = self.base.jaxnodes[channel_params]

        for synapse_params in self.base.synapse_param_names:
            params[synapse_params] = self.base.jaxedges[synapse_params]

        # Override with those parameters set by `.make_trainable()`.
        for parameter in pstate:
            key = parameter["key"]
            inds = parameter["indices"]
            set_param = parameter["val"]

            # This is needed since SynapseViews worked differently before.
            # This mimics the old behaviour and tranformes the new indices
            # to the old indices.
            # TODO FROM #447: Longterm this should be gotten rid of.
            # Instead edges should work similar to nodes (would also allow for
            # param sharing).
            synapse_inds = self.base.edges.groupby("type").rank()["global_edge_index"]
            synapse_inds = (synapse_inds.astype(int) - 1).to_numpy()
            if key in self.base.synapse_param_names:
                inds = synapse_inds[inds]

            if key in params:  # Only parameters, not initial states.
                # `inds` is of shape `(num_params, num_comps_per_param)`.
                # `set_param` is of shape `(num_params,)`
                # We need to unsqueeze `set_param` to make it `(num_params, 1)` for the
                # `.set()` to work. This is done with `[:, None]`.
                params[key] = params[key].at[inds].set(set_param[:, None])

            # If radius and length are updated by the pstate, then we have to also
            # update 1) area, 2) source_frustum, and 3) sink_frustum.
            if key in ["radius", "length"]:
                l = params["length"][inds]
                r = params["radius"][inds]
                # l/2 because we want the input load (left half of the cylinder) and
                # the output load (right half of the cylinder).
                resistive_load = cylinder_resistive_load(l / 2, r)
                params["area"] = params["area"].at[inds].set(cylinder_area(l, r))
                params["volume"] = params["volume"].at[inds].set(cylinder_volume(l, r))
                params["resistive_load_out"] = (
                    params["resistive_load_out"].at[inds].set(resistive_load)
                )
                params["resistive_load_in"] = (
                    params["resistive_load_in"].at[inds].set(resistive_load)
                )

        # Compute conductance params and add them to the params dictionary.
        params["axial_conductances"] = compute_axial_conductances(
            self.base._comp_edges, params, self.base.diffusion_states
        )
        return params

    @only_allow_module
    def _get_states_from_nodes_and_edges(self) -> Dict[str, jnp.ndarray]:
        """Return states as they are set in the `.nodes` and `.edges` tables.

        TODO FROM #447: MAKE THIS WORK FOR VIEW?
        """
        # Create `.jaxnodes` from `.nodes` and `.jaxedges` from `.edges`.
        self.base.to_jax()
        states = {"v": self.base.jaxnodes["v"]}
        # Join node and edge states into a single state dictionary.
        for channel in self.base.channels + self.base.pumps:
            for channel_states in channel.channel_states:
                states[channel_states] = self.base.jaxnodes[channel_states]
        for synapse_states in self.base.synapse_state_names:
            states[synapse_states] = self.base.jaxedges[synapse_states]
        return states

    @only_allow_module
    def get_all_states(
        self, pstate: List[Dict], all_params, delta_t: float
    ) -> Dict[str, jnp.ndarray]:
        # TODO FROM #447: MAKE THIS WORK FOR VIEW?
        """Get the full initial state of the module from jaxnodes and trainables.

        Args:
            pstate: The state of the trainable parameters.
            all_params: All parameters of the module.
            delta_t: The time step.

        Returns:
            A dictionary of all states of the module.
        """
        states = self.base._get_states_from_nodes_and_edges()

        # Override with the initial states set by `.make_trainable()`.
        for parameter in pstate:
            key = parameter["key"]
            inds = parameter["indices"]
            set_param = parameter["val"]
            if key in states:  # Only initial states, not parameters.
                # `inds` is of shape `(num_params, num_comps_per_param)`.
                # `set_param` is of shape `(num_params,)`
                # We need to unsqueeze `set_param` to make it `(num_params, 1)` for the
                # `.set()` to work. This is done with `[:, None]`.
                states[key] = states[key].at[inds].set(set_param[:, None])

        # Add to the states the initial current through every channel.
        states, _ = self.base._channel_currents(
            states, delta_t, self.channels + self.pumps, self.nodes, all_params
        )

        # Add to the states the initial current through every synapse.
        states, _ = self.base._synapse_currents(
            states,
            self.synapses,
            all_params,
            delta_t,
            self.edges,
        )
        return states

    @property
    def initialized(self) -> bool:
        """Whether the `Module` is ready to be solved or not."""
        return self.initialized_solver

    def initialize(self):
        """Initialize the module.

        This function does several things:
        1) It computes local indices in the `.nodes` dataframe (from global indices).
        2) It builds the compartment graph (`._comp_edges` and `._branchpoints`).
        3) It initializes the `View`.
        4) It initializes all solvers required for solving the differential equation.

        This function should be run whenever the graph-structure (i.e., the morphology
        or the compartmentalization) of the module have been changed. Inbuilt functions
        such as `morph_attach()`, `morph_delete()`, or `set_ncomp()` run this function
        automatically though, so there is no need for the user to run it manually.
        """
        # Compute the local indices from the global indices.
        self._update_local_indices()

        # Initialize compartment graph structure (`_comp_edges`, `_branchpoints`, ...).
        self._init_comp_graph()

        # Initialize view of nodes, edges, and compartment graph structure.
        self._init_view()

        # Initialize solvers.
        self._init_solvers()

        return self

    @only_allow_module
    def init_states(self, delta_t: float = 0.025):
        # TODO FROM #447: MAKE THIS WORK FOR VIEW?
        """Initialize all mechanisms in their steady state.

        This considers the voltages and parameters of each compartment.

        Args:
            delta_t: Passed on to `channel.init_state()`.
        """
        # Update states of the channels.
        channel_nodes = self.base.nodes
        states = self.base._get_states_from_nodes_and_edges()

        # We do not use any `pstate` for initializing. In principle, we could change
        # that by allowing an input `params` and `pstate` to this function.
        # `voltage_solver` could also be `jax.sparse` here, because both of them
        # build the channel parameters in the same way.
        params = self.base.get_all_parameters([])

        for channel in self.base.channels + self.base.pumps:
            name = channel._name
            channel_indices = channel_nodes.loc[channel_nodes[name]][
                "global_comp_index"
            ].to_numpy()
            voltages = channel_nodes.loc[channel_indices, "v"].to_numpy()

            channel_param_names = list(channel.channel_params.keys())
            channel_state_names = list(channel.channel_states.keys())
            channel_states = query_channel_states_and_params(
                states, channel_state_names, channel_indices
            )
            channel_params = query_channel_states_and_params(
                params, channel_param_names, channel_indices
            )

            init_state = channel.init_state(
                channel_states, voltages, channel_params, delta_t
            )

            # `init_state` might not return all channel states. Only the ones that are
            # returned are updated here.
            for key, val in init_state.items():
                # Note that we are overriding `self.nodes` here, but `self.nodes` is
                # not used above to actually compute the current states (so there are
                # no issues with overriding states).
                self.nodes.loc[channel_indices, key] = val

    def _init_morph_for_debugging(self):
        """Instandiates row and column inds which can be used to solve the voltage eqs.

        This is important only for expert users who try to modify the solver for the
        voltage equations. By default, this function is never run.

        This is useful for debugging the solver because one can use
        `scipy.linalg.sparse.spsolve` after every step of the solve.

        Here is the code snippet that can be used for debugging then (to be inserted in
        `solver_voltage`):
        ```python
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        from jaxley.utils.debug_solver import build_voltage_matrix_elements

        elements, solve, num_entries, start_ind_for_branchpoints = (
            build_voltage_matrix_elements(
                uppers,
                lowers,
                diags,
                solves,
                branchpoint_conds_children[debug_states["child_inds"]],
                branchpoint_conds_parents[debug_states["par_inds"]],
                branchpoint_weights_children[debug_states["child_inds"]],
                branchpoint_weights_parents[debug_states["par_inds"]],
                branchpoint_diags,
                branchpoint_solves,
                debug_states["ncomp"],
                nbranches,
            )
        )
        sparse_matrix = csc_matrix(
            (elements, (debug_states["row_inds"], debug_states["col_inds"])),
            shape=(num_entries, num_entries),
        )
        solution = spsolve(sparse_matrix, solve)
        solution = solution[:start_ind_for_branchpoints]  # Delete branchpoint voltages.
        solves = jnp.reshape(solution, (debug_states["ncomp"], nbranches))
        return solves
        ```
        """
        # For scipy and jax.scipy.
        row_and_col_inds = compute_morphology_indices(
            len(self.base._par_inds),
            self.base._child_belongs_to_branchpoint,
            self.base._par_inds,
            self.base._child_inds,
            self.base.ncomp,
            self.base.total_nbranches,
        )

        num_elements = len(row_and_col_inds["row_inds"])
        data_inds, indices, indptr = convert_to_csc(
            num_elements=num_elements,
            row_ind=row_and_col_inds["row_inds"],
            col_ind=row_and_col_inds["col_inds"],
        )
        self.base.debug_states["row_inds"] = row_and_col_inds["row_inds"]
        self.base.debug_states["col_inds"] = row_and_col_inds["col_inds"]
        self.base.debug_states["data_inds"] = data_inds
        self.base.debug_states["indices"] = indices
        self.base.debug_states["indptr"] = indptr

        self.base.debug_states["ncomp"] = self.base.ncomp
        self.base.debug_states["child_inds"] = self.base._child_inds
        self.base.debug_states["par_inds"] = self.base._par_inds

    def record(self, state: str = "v", verbose=True):
        comp_states, edge_states = self._get_state_names()
        if state in comp_states:
            in_view = self._nodes_in_view
        elif state in edge_states:
            in_view = self.base.edges.iloc[
                self._edges_in_view.tolist()
            ].index_within_type.to_numpy(dtype=int)
        else:
            raise KeyError(f"{state} is not a recognized state in this module.")

        new_recs = pd.DataFrame(in_view, columns=["rec_index"])
        new_recs["state"] = state
        self.base.recordings = pd.concat([self.base.recordings, new_recs])
        has_duplicates = self.base.recordings.duplicated()
        self.base.recordings = self.base.recordings.loc[~has_duplicates]
        if verbose:
            print(
                f"Added {len(in_view)-sum(has_duplicates)} recordings. See `.recordings` for details."
            )

    def _update_view(self):
        """Update the attrs of the view after changes in the base module."""
        if isinstance(self, View):
            scope = self._scope
            current_view = self._current_view
            # copy dict of new View. For some reason doing self = View(self)
            # did not work.
            self.__dict__ = View(
                self.base, self._nodes_in_view, self._edges_in_view
            ).__dict__

            # retain the scope and current_view of the previous view
            self._scope = scope
            self._current_view = current_view

    def delete_recordings(self):
        """Removes all recordings from the module."""
        if isinstance(self, View):
            base_recs = self.base.recordings
            self.base.recordings = base_recs[
                ~base_recs.isin(self.recordings).all(axis=1)
            ]
            self._update_view()
        else:
            self.base.recordings = pd.DataFrame().from_dict({})

    def stimulate(self, current: Optional[jnp.ndarray] = None, verbose: bool = True):
        """Insert a stimulus into the compartment.

        current must be a 1d array or have batch dimension of size `(num_compartments, )`
        or `(1, )`. If 1d, the same stimulus is added to all compartments.

        This function cannot be run during `jax.jit` and `jax.grad`. Because of this,
        it should only be used for static stimuli (i.e., stimuli that do not depend
        on the data and that should not be learned). For stimuli that depend on data
        (or that should be learned), please use `data_stimulate()`.

        Args:
            current: Current in `nA`.
        """
        self._external_input("i", current, verbose=verbose)

    def clamp(self, state_name: str, state_array: jnp.ndarray, verbose: bool = True):
        """Clamp a state to a given value across specified compartments.

        Args:
            state_name: The name of the state to clamp.
            state_array: Array of values to clamp the state to.
            verbose: If True, prints details about the clamping.

        This function sets external states for the compartments.
        """
        self._external_input(state_name, state_array, verbose=verbose)

    def _external_input(
        self,
        key: str,
        values: Optional[jnp.ndarray],
        verbose: bool = True,
    ):
        comp_states, edge_states = self._get_state_names()
        if key not in comp_states + edge_states:
            raise KeyError(f"{key} is not a recognized state in this module.")
        values = values if values.ndim == 2 else jnp.expand_dims(values, axis=0)
        batch_size = values.shape[0]
        num_inserted = (
            len(self._nodes_in_view) if key in comp_states else len(self._edges_in_view)
        )
        is_multiple = num_inserted == batch_size
        values = values if is_multiple else jnp.repeat(values, num_inserted, axis=0)
        assert batch_size in [
            1,
            num_inserted,
        ], "Number of comps and stimuli do not match."

        if key in self.base.externals.keys():
            self.base.externals[key] = jnp.concatenate(
                [self.base.externals[key], values]
            )
            self.base.external_inds[key] = jnp.concatenate(
                [self.base.external_inds[key], self._nodes_in_view]
            )
        else:
            if key in comp_states:
                self.base.externals[key] = values
                self.base.external_inds[key] = self._nodes_in_view
            else:
                self.base.externals[key] = values
                self.base.external_inds[key] = self._edges_in_view
        if verbose:
            print(
                f"Added {num_inserted} external_states. See `.externals` for details."
            )

    def data_stimulate(
        self,
        current: jnp.ndarray,
        data_stimuli: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
        verbose: bool = False,
    ) -> Tuple[jnp.ndarray, pd.DataFrame]:
        """Insert a stimulus into the module within jit (or grad).

        Args:
            current: Current in `nA`.
            verbose: Whether or not to print the number of inserted stimuli. `False`
                by default because this method is meant to be jitted.
        """
        return self._data_external_input(
            "i", current, data_stimuli, self.nodes, verbose=verbose
        )

    def data_clamp(
        self,
        state_name: str,
        state_array: jnp.ndarray,
        data_clamps: Optional[Tuple[jnp.ndarray, pd.DataFrame]] = None,
        verbose: bool = False,
    ):
        """Insert a clamp into the module within jit (or grad).

        Args:
            state_name: Name of the state variable to set.
            state_array: Time series of the state variable in the default Jaxley unit.
                State array should be of shape (num_clamps, simulation_time) or
                (simulation_time, ) for a single clamp.
            verbose: Whether or not to print the number of inserted clamps. `False`
                by default because this method is meant to be jitted.
        """
        comp_states, edge_states = self._get_state_names()
        if state_name not in comp_states + edge_states:
            raise KeyError(f"{state_name} is not a recognized state in this module.")
        data = self.nodes if state_name in comp_states else self.edges
        return self._data_external_input(
            state_name, state_array, data_clamps, data, verbose=verbose
        )

    def _data_external_input(
        self,
        state_name: str,
        state_array: jnp.ndarray,
        data_external_input: Optional[Tuple[jnp.ndarray, pd.DataFrame]],
        view: pd.DataFrame,
        verbose: bool = False,
    ):
        comp_states, edge_states = self._get_state_names()
        state_array = (
            state_array
            if state_array.ndim == 2
            else jnp.expand_dims(state_array, axis=0)
        )
        batch_size = state_array.shape[0]
        num_inserted = (
            len(self._nodes_in_view)
            if state_name in comp_states
            else len(self._edges_in_view)
        )
        is_multiple = num_inserted == batch_size
        state_array = (
            state_array
            if is_multiple
            else jnp.repeat(state_array, num_inserted, axis=0)
        )
        assert batch_size in [
            1,
            num_inserted,
        ], "Number of comps and clamps do not match."

        if data_external_input is not None:
            external_input = data_external_input[1]
            external_input = jnp.concatenate([external_input, state_array])
            inds = data_external_input[2]
        else:
            external_input = state_array
            inds = pd.DataFrame().from_dict({})

        inds = pd.concat([inds, view])

        if verbose:
            if state_name == "i":
                print(f"Added {len(view)} stimuli.")
            else:
                print(f"Added {len(view)} clamps.")

        return (state_name, external_input, inds)

    def delete_stimuli(self):
        """Removes all stimuli from the module."""
        self.delete_clamps("i")

    def delete_clamps(self, state_name: Optional[str] = None):
        """Removes all clamps of the given state from the module."""
        all_externals = list(self.externals.keys())
        if "i" in all_externals:
            all_externals.remove("i")
        state_names = all_externals if state_name is None else [state_name]
        for state_name in state_names:
            if state_name in self.externals:
                keep_inds = ~np.isin(
                    self.base.external_inds[state_name], self._nodes_in_view
                )
                base_exts = self.base.externals
                base_exts_inds = self.base.external_inds
                if np.all(~keep_inds):
                    base_exts.pop(state_name, None)
                    base_exts_inds.pop(state_name, None)
                else:
                    base_exts[state_name] = base_exts[state_name][keep_inds]
                    base_exts_inds[state_name] = base_exts_inds[state_name][keep_inds]
                self._update_view()
            else:
                pass  # does not have to be deleted if not in externals

    def insert(self, channel: Union[Channel, Pump]):
        """Insert a channel or pump into the module.

        Args:
            channel: The channel to insert."""
        name = channel._name

        assert name not in self.group_names, (
            "You are trying to insert a channel whose name is the same as one of the "
            "group names. This is not supported. Either rename the channel or use a "
            "different name for the group."
        )

        # Channel does not yet exist in the `jx.Module` at all.
        if isinstance(channel, Channel) and name not in [
            c._name for c in self.base.channels
        ]:
            self.base.channels.append(channel)
            self.base.nodes[name] = (
                False  # Previous columns do not have the new channel.
            )
        # Pump does not exist yet in the `jx.Module` at all.
        if isinstance(channel, Pump) and name not in [c._name for c in self.base.pumps]:
            self.base.pumps.append(channel)
            self.base.nodes[name] = (
                False  # Previous columns do not have the new channel.
            )
            if channel.ion_name not in self.base.pumped_ions:
                self.base.pumped_ions.append(channel.ion_name)

        if channel.current_name not in self.base.membrane_current_names:
            self.base.membrane_current_names.append(channel.current_name)

        # Add a binary column that indicates if a channel is present.
        self.base.nodes.loc[self._nodes_in_view, name] = True

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_params:
            self.base.nodes.loc[self._nodes_in_view, key] = channel.channel_params[key]

        # Loop over all new parameters, e.g. gNa, eNa.
        for key in channel.channel_states:
            self.base.nodes.loc[self._nodes_in_view, key] = channel.channel_states[key]

    @only_allow_module
    def diffuse(self, state: str) -> None:
        """Diffuse a particular state across compartments with Fickian diffusion.

        Args:
            state: Name of the state that should be diffused.
        """
        self.base.diffusion_states.append(state)
        self.base.nodes.loc[self._nodes_in_view, f"axial_diffusion_{state}"] = 1.0

        # The diffused state might not exist in all compartments that across which
        # we are diffusing (e.g. there are active calcium mechanisms only in the soma,
        # but calcium should still diffuse into the dendrites). Here, we ensure that
        # the state is not `NaN` in every compartment across which we are diffusing.
        state_is_nan = pd.isna(self.base.nodes.loc[self._nodes_in_view, state])
        # 0.0 would lead to division by zero in Nernst reversal, but states that have
        # the NernstReversal should have the state anyways.
        self.base.nodes.loc[state_is_nan, state] = 0.0

    @only_allow_module
    def delete_diffusion(self, state: str) -> None:
        """Deletes ion diffusion in the entire module.

        Args:
            state: Name of the state that should no longer be diffused.
        """
        assert (
            state in self.base.diffusion_states
        ), f"State {state} is not part of `self.diffusion_states`."
        self.base.diffusion_states.remove(state)
        self.base.nodes.drop(columns=[f"axial_diffusion_{state}"], inplace=True)

    def delete(self, channel: Union[Channel, Pump]):
        """Remove a channel or pump from the module.

        Args:
            channel: The channel to remove."""
        name = channel._name
        channel_names = [c._name for c in self.channels + self.pumps]
        all_channel_names = [c._name for c in self.base.channels]
        all_pump_names = [c._name for c in self.base.pumps]
        if name in channel_names:
            channel_cols = list(channel.channel_params.keys())
            channel_cols += list(channel.channel_states.keys())
            self.base.nodes.loc[self._nodes_in_view, channel_cols] = float("nan")
            self.base.nodes.loc[self._nodes_in_view, name] = False

            # only delete cols if no other comps in the module have the same channel
            if np.all(~self.base.nodes[name]):
                if isinstance(channel, Channel):
                    self.base.channels.pop(all_channel_names.index(name))
                elif isinstance(channel, Pump):
                    self.base.pumps.pop(all_pump_names.index(name))
                else:
                    raise ValueError(
                        "The channel/pump to be deleted is neither a channel nor a "
                        "pump. Maybe you ran `cell.delete(HH)` instead of "
                        "`cell.delete(HH())` (ie you forgot to initialize the channel "
                        "via round brackets: `HH()`."
                    )
                self.base.membrane_current_names.remove(channel.current_name)
                self.base.nodes.drop(columns=channel_cols + [name], inplace=True)
        else:
            raise ValueError(f"Channel {name} not found in the module.")

    @only_allow_module
    def step(
        self,
        u: Dict[str, jnp.ndarray],
        delta_t: float,
        external_inds: Dict[str, jnp.ndarray],
        externals: Dict[str, jnp.ndarray],
        params: Dict[str, jnp.ndarray],
        solver: str = "bwd_euler",
        voltage_solver: str = "jaxley.stone",
    ) -> Dict[str, jnp.ndarray]:
        """One step of solving the Ordinary Differential Equation.

        This function is called inside of `integrate` and increments the state of the
        module by one time step. Calls `_step_channels` and `_step_synapse` to update
        the states of the channels and synapses.

        Args:
            u: The state of the module. voltages = u["v"]
            delta_t: The time step.
            external_inds: The indices of the external inputs.
            externals: The external inputs.
            params: The parameters of the module.
            solver: The solver to use for the voltages. Either of ["bwd_euler",
                "fwd_euler", "crank_nicolson"].
            voltage_solver: The tridiagonal solver used to diagonalize the
                coefficient matrix of the ODE system. Either of ["jaxley.thomas",
                "jaxley.stone"].

        Returns:
            The updated state of the module.
        """
        # Extract the external inputs
        if "i" in externals.keys():
            i_current = externals["i"]
            i_inds = external_inds["i"]
            i_ext = self._get_external_input(u["v"], i_inds, i_current, params["area"])
        else:
            i_ext = 0.0

        # Steps of the channel & pump states and computes the current through these
        # channels and pumps.
        u, (linear_terms, const_terms) = self._step_channels(
            u, delta_t, self.channels + self.pumps, self.nodes, params
        )

        # Step of the synapse.
        u, (v_syn_linear_terms, v_syn_const_terms) = self._step_synapse(
            u,
            self.synapses,
            params,
            delta_t,
            self.edges,
        )

        # Voltage steps.
        cm = params["capacitance"]  # Abbreviation.

        # Arguments used by all solvers.
        state_vals = {
            "states": [u["v"]],
            "linear_terms": [(linear_terms["v"] + v_syn_linear_terms) / cm],
            "constant_terms": [(const_terms["v"] + i_ext + v_syn_const_terms) / cm],
            # The axial conductances have already been divided by `cm` in the
            # `cell_utils.py` in the `compute_axial_conductances` method.
            "axial_conductances": [params["axial_conductances"]["v"]],
        }

        for ion_name in self.pumped_ions:
            if ion_name not in self.diffusion_states:
                # If an ion is pumped but _not_ diffused, we update the state of the ion
                # (i.e., its concentration) with implicit Euler. We could also use
                # exponential-euler here, but we use implicit Euler for consistency with
                # the case of the ion being diffused. TODO: In the long run, we should
                # give the user the option to specify the solver.
                #
                # Implicit Euler for diagonal system (i.e. all compartments are
                # independent):
                #
                # v_dot = const + v * linear
                # v_n = v_{n+1} - dt * (const + v_{n+1} * linear)
                # ...
                # v_{n+1} = (v_n + dt * const) / (1 - dt * linear)
                u[ion_name] = (u[ion_name] + delta_t * const_terms[ion_name]) / (
                    1 + delta_t * linear_terms[ion_name]
                )

        for ion_name in self.diffusion_states:
            if ion_name not in self.pumped_ions:
                # Ions that are not pumped have no active component.
                ion_linear_term = jnp.zeros_like(u[ion_name])
                ion_const_term = jnp.zeros_like(u[ion_name])
            else:
                ion_linear_term = linear_terms[ion_name]
                ion_const_term = const_terms[ion_name]
            # Append the states of the pumps if they are diffusing (the user must
            # manually specify ion diffusion with `cell.diffuse(ion_state_name)`). Note
            # that these values are _not_ divided by the capacitance `cm`.
            if ion_name in self.diffusion_states:
                state_vals["states"] += [u[ion_name]]
                state_vals["linear_terms"] += [ion_linear_term]
                state_vals["constant_terms"] += [ion_const_term]
                state_vals["axial_conductances"] += [
                    params[f"axial_conductances"][ion_name]
                ]

        # Stack all states such that they can be handled by `vmap` in the solve.
        for state_name in [
            "states",
            "linear_terms",
            "constant_terms",
            "axial_conductances",
        ]:
            state_vals[state_name] = jnp.stack(state_vals[state_name])

        # Clamp for channels and synapses.
        for key in externals.keys():
            if key not in ["i", "v"]:
                u[key] = u[key].at[external_inds[key]].set(externals[key])

        # Add solver specific arguments.
        if solver == "fwd_euler":
            solver_kwargs = {
                "sinks": np.asarray(self._comp_edges["sink"].to_list()),
                "sources": np.asarray(self._comp_edges["source"].to_list()),
                "types": np.asarray(self._comp_edges["type"].to_list()),
            }
        elif voltage_solver == "jax.sparse":
            solver_kwargs = {
                "internal_node_inds": self._internal_node_inds,
                "sinks": np.asarray(self._comp_edges["sink"].to_list()),
                "data_inds": self._data_inds,
                "indices": self._indices_jax_spsolve,
                "indptr": self._indptr_jax_spsolve,
                "n_nodes": self._n_nodes,
            }
            step_voltage_implicit = step_voltage_implicit_with_jax_spsolve
        elif voltage_solver.startswith("jaxley.dhs"):
            solver_kwargs = {
                "internal_node_inds": self._internal_node_inds,
                "sinks": np.asarray(self._comp_edges["sink"].to_list()),
                "n_nodes": self._n_nodes,
                "solve_indexer": self._dhs_solve_indexer,
                "optimize_for_gpu": True if voltage_solver.endswith("gpu") else False,
            }
            step_voltage_implicit = step_voltage_implicit_with_dhs_solve
        elif voltage_solver == "jaxley.stone":
            # Our custom sparse solver requires a different format of all conductance
            # values to perform triangulation and backsubstution optimally.
            #
            # Currently, the forward Euler solver also uses this format. However,
            # this is only for historical reasons and we are planning to change this in
            # the future.
            solver_kwargs = {
                "internal_node_inds": self._internal_node_inds,
                "n_nodes": self._n_nodes,
                "sinks": np.asarray(self._comp_edges["sink"].to_list()),
                "sources": np.asarray(self._comp_edges["source"].to_list()),
                "types": np.asarray(self._comp_edges["type"].to_list()),
            }
            step_voltage_implicit = step_voltage_implicit_with_stone

        if solver in ["bwd_euler", "crank_nicolson"]:
            # Crank-Nicolson advances by half a step of backward and half a step of
            # forward Euler.
            dt = delta_t / 2 if solver == "crank_nicolson" else delta_t

            if voltage_solver == "jax.sparse":
                # The `jax.sparse` solver does not allow `vmap` (because it uses) the
                # scipy sparse solver, so we just loop here.
                num_ions = state_vals["states"].shape[0]
                updated_states = []
                for ion_ind in range(num_ions):
                    updated_states.append(
                        step_voltage_implicit(
                            state_vals["states"][ion_ind],
                            state_vals["linear_terms"][ion_ind],
                            state_vals["constant_terms"][ion_ind],
                            state_vals["axial_conductances"][ion_ind],
                            *solver_kwargs.values(),
                            dt,
                        )
                    )
                updated_states = jnp.stack(updated_states)
            else:
                # The following if-case is a bit ugly and, technically, not needed.
                # However, running a `vmapped` version of the implicit solver induces
                # significant computation cost, even if the leading dimension of the
                # `vmap` is 1 (as is the case if one has no diffusion). To ensure
                # fast runtime and compile time, the following if-case avoids the `vmap`
                # if one does not use diffusion.
                if len(self.diffusion_states) == 0:
                    updated_states = step_voltage_implicit(
                        state_vals["states"][0],
                        state_vals["linear_terms"][0],
                        state_vals["constant_terms"][0],
                        state_vals["axial_conductances"][0],
                        *solver_kwargs.values(),
                        dt,
                    )
                    # Add `vmap` dimension.
                    updated_states = jnp.expand_dims(updated_states, axis=0)
                else:
                    nones = [None] * len(solver_kwargs)
                    vmapped = vmap(
                        step_voltage_implicit, in_axes=(0, 0, 0, 0, *nones, None)
                    )
                    updated_states = vmapped(
                        *state_vals.values(), *solver_kwargs.values(), dt
                    )
            if solver == "crank_nicolson":
                # The forward Euler step in Crank-Nicolson can be performed easily as
                # `V_{n+1} = 2 * V_{n+1/2} - V_n`. See also NEURON book Chapter 4.
                updated_states = 2 * updated_states - state_vals["states"]
        elif solver == "fwd_euler":
            nones = [None] * len(solver_kwargs)
            vmapped = vmap(step_voltage_explicit, in_axes=(0, 0, 0, 0, *nones, None))
            updated_states = vmapped(
                *state_vals.values(), *solver_kwargs.values(), delta_t
            )
        else:
            raise ValueError(
                f"You specified `solver={solver}`. The only allowed solvers are "
                "['bwd_euler', 'fwd_euler', 'crank_nicolson']."
            )

        u["v"] = updated_states[0]

        # Assign the diffused ion states.
        for counter, ion_name in enumerate(self.diffusion_states):
            u[ion_name] = updated_states[counter + 1]

        # Clamp for voltages.
        if "v" in externals.keys():
            u["v"] = u["v"].at[external_inds["v"]].set(externals["v"])

        return u

    def _step_channels(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """One step of integration of the channels and of computing their current."""
        states = self._step_channels_state(
            states, delta_t, channels, channel_nodes, params
        )
        states, current_terms = self._channel_currents(
            states, delta_t, channels, channel_nodes, params
        )
        return states, current_terms

    def _step_channels_state(
        self,
        states,
        delta_t,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """One integration step of the channels."""
        voltages = states["v"]

        # Update states of the channels.
        indices = channel_nodes.index.to_numpy()
        for channel in channels:
            channel_param_names = list(channel.channel_params)
            channel_param_names += [
                "radius",
                "length",
                "axial_resistivity",
                "capacitance",
            ]
            channel_state_names = list(channel.channel_states)
            channel_state_names += self.membrane_current_names
            channel_indices = indices[channel_nodes[channel._name].astype(bool)]

            channel_params = query_channel_states_and_params(
                params, channel_param_names, channel_indices
            )
            channel_states = query_channel_states_and_params(
                states, channel_state_names, channel_indices
            )

            states_updated = channel.update_states(
                channel_states, delta_t, voltages[channel_indices], channel_params
            )
            # Rebuild state. This has to be done within the loop over channels to allow
            # multiple channels which modify the same state.
            for key, val in states_updated.items():
                states[key] = states[key].at[channel_indices].set(val)

        return states

    def _channel_currents(
        self,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        channels: List[Channel],
        channel_nodes: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return the current through each channel.

        This is also updates `state` because the `state` also contains the current.
        """
        # Compute current through channels.
        linear_terms = {}
        const_terms = {}
        for name in ["v"] + self.pumped_ions:
            modified_state = states[name]
            linear_terms[name] = jnp.zeros_like(states[name])
            const_terms[name] = jnp.zeros_like(states[name])

        current_states = {}
        for name in self.membrane_current_names:
            current_states[name] = jnp.zeros_like(modified_state)

        for channel in channels:
            name = channel._name
            if isinstance(channel, Channel):
                modified_state_name = "v"
            else:
                modified_state_name = channel.ion_name
            modified_state = states[modified_state_name]

            indices = channel_nodes.loc[channel_nodes[name]].index.to_numpy()
            current, linear_term, const_term = self._channel_current_components(
                modified_state,
                states,
                delta_t,
                channel,
                indices,
                params,
            )
            linear_terms[modified_state_name] = (
                linear_terms[modified_state_name].at[indices].add(linear_term)
            )
            const_terms[modified_state_name] = (
                const_terms[modified_state_name].at[indices].add(const_term)
            )

            # Save the current (for the unperturbed voltage) as a state that will
            # also be passed to the state update.
            current_states[channel.current_name] = (
                current_states[channel.current_name].at[indices].add(current)
            )

        # Copy the currents into the `state` dictionary such that they can be
        # recorded and used by `Channel.update_states()`.
        for name in self.membrane_current_names:
            states[name] = current_states[name]

        # * 1_000.0 to convert from mA/cm^2 to uA/cm^2.
        linear_terms["v"] *= 1000.0
        const_terms["v"] *= 1000.0
        return states, (linear_terms, const_terms)

    def _channel_current_components(
        self,
        modified_state: jnp.ndarray,
        states: Dict[str, jnp.ndarray],
        delta_t: float,
        channel: Channel,
        indices: pd.DataFrame,
        params: Dict[str, jnp.ndarray],
    ):
        """Computes current through a channel and its linear and const components.

        The linear and constant components are inferred by running the `compute_current`
        twice. They are later used for implicit Euler.
        """
        # Run with two different voltages that are `diff` apart to infer the slope and
        # offset.
        diff = 1e-3

        channel_param_names = list(channel.channel_params.keys())
        channel_state_names = list(channel.channel_states.keys())

        channel_params = {}
        for p in channel_param_names:
            channel_params[p] = params[p][indices]
        channel_params["radius"] = params["radius"][indices]
        channel_params["length"] = params["length"][indices]
        channel_params["axial_resistivity"] = params["axial_resistivity"][indices]

        channel_states = {}
        for s in channel_state_names:
            channel_states[s] = states[s][indices]

        v_and_perturbed = jnp.stack(
            [modified_state[indices], modified_state[indices] + diff]
        )
        membrane_currents = vmap(channel.compute_current, in_axes=(None, 0, None))(
            channel_states, v_and_perturbed, channel_params
        )
        voltage_term = (membrane_currents[1] - membrane_currents[0]) / diff
        constant_term = membrane_currents[0] - voltage_term * modified_state[indices]
        return membrane_currents[0], voltage_term, -constant_term

    def _step_synapse(
        self,
        u: Dict[str, jnp.ndarray],
        syn_channels: List[Channel],
        params: Dict[str, jnp.ndarray],
        delta_t: float,
        edges: pd.DataFrame,
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """One step of integration of the channels.

        `Network` overrides this method (because it actually has synapses), whereas
        `Compartment`, `Branch`, and `Cell` do not override this.
        """
        voltages = u["v"]
        return u, (jnp.zeros_like(voltages), jnp.zeros_like(voltages))

    def _synapse_currents(
        self,
        states,
        syn_channels,
        params,
        delta_t,
        edges: pd.DataFrame,
    ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        return states, (None, None)

    @staticmethod
    def _get_external_input(
        voltages: jnp.ndarray,
        i_inds: jnp.ndarray,
        i_stim: jnp.ndarray,
        area: float,
    ) -> jnp.ndarray:
        """
        Return external input to each compartment in uA / cm^2.

        Args:
            voltages: mV.
            i_stim: nA.
            radius: um.
            length_single_compartment: um.
        """
        zero_vec = jnp.zeros_like(voltages)
        current = convert_point_process_to_distributed(i_stim, area[i_inds])

        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        stim_at_timestep = scatter_add(zero_vec, i_inds[:, None], current, dnums)
        return stim_at_timestep

    def vis(
        self,
        ax: Optional[Axes] = None,
        color: str = "k",
        dims: Tuple[int] = (0, 1),
        type: str = "line",
        **kwargs,
    ) -> Axes:
        """Visualize the module.

        Modules can be visualized on one of the cardinal planes (xy, xz, yz) or
        even in 3D.

        Several options are available:
        - `line`: All points from the traced morphology (`xyzr`), are connected
        with a line plot.
        - `scatter`: All traced points, are plotted as scatter points.
        - `comp`: Plots the compartmentalized morphology, including radius
        and shape. (shows the true compartment lengths per default, but this can
        be changed via the `kwargs`, for details see
        `jaxley.utils.plot_utils.plot_comps`).
        - `morph`: Reconstructs the 3D shape of the traced morphology. For details see
        `jaxley.utils.plot_utils.plot_morph`. Warning: For 3D plots and morphologies
        with many traced points this can be very slow.

        Args:
            ax: An axis into which to plot.
            color: The color for all branches.
            dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
                two of them.
            type: The type of plot. One of ["line", "scatter", "comp", "morph"].
            kwargs: Keyword arguments passed to the plotting function.
        """
        res = 100 if "resolution" not in kwargs else kwargs.pop("resolution")
        if "comp" in type.lower():
            return plot_comps(
                self, dims=dims, ax=ax, color=color, resolution=res, **kwargs
            )
        if "morph" in type.lower():
            return plot_morph(
                self, dims=dims, ax=ax, color=color, resolution=res, **kwargs
            )

        assert not np.any([np.isnan(xyzr[:, dims]).all() for xyzr in self.xyzr]), (
            "No coordinates available. Use `vis(detail='point')` or run "
            "`.compute_xyz()` before running `.vis()`."
        )

        ax = plot_graph(
            self.xyzr,
            dims=dims,
            color=color,
            ax=ax,
            type=type,
            **kwargs,
        )

        return ax

    def compute_xyz(self):
        """Return xyz coordinates of every branch, based on the branch length.

        This function should not be called if the morphology was read from an `.swc`
        file. However, for morphologies that were constructed from scratch, this
        function **must** be called before `.vis()`. The computed `xyz` coordinates
        are only used for plotting.
        """
        max_y_multiplier = 5.0
        min_y_multiplier = 0.5

        parents = self.comb_parents
        num_children = _compute_num_children(parents)
        index_of_child = _compute_index_of_child(parents)
        levels = compute_levels(parents)

        # Extract branch.
        inds_branch = self.nodes.groupby("global_branch_index")[
            "global_comp_index"
        ].apply(list)
        branch_lens = [
            np.sum(
                self.nodes.set_index("global_comp_index").loc[np.asarray(i), "length"]
            )
            for i in inds_branch
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
            # For networks with mixed SWC and from-scratch neurons, only update those
            # branches that do not have coordinates yet.
            if np.any(np.isnan(self.xyzr[b])):
                if parents[b] > -1:
                    start_point = endpoints[parents[b]]
                    num_children_of_parent = num_children[parents[b]]
                    if num_children_of_parent > 1:
                        y_offset = (
                            ((index_of_child[b] / (num_children_of_parent - 1))) - 0.5
                        ) * y_offset_multiplier[levels[b]]
                    else:
                        y_offset = 0.0
                else:
                    start_point = [0, 0, 0]
                    y_offset = 0.0

                len_of_path = np.sqrt(y_offset**2 + 1.0)

                end_point = [
                    start_point[0] + branch_lens[b] / len_of_path * 1.0,
                    start_point[1] + branch_lens[b] / len_of_path * y_offset,
                    start_point[2],
                ]
                endpoints.append(end_point)

                self.xyzr[b][:, :3] = np.asarray([start_point, end_point])
            else:
                # Dummy to keey the index `endpoints[parent[b]]` above working.
                endpoints.append(np.zeros((2,)))

    def move(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, update_nodes: bool = False
    ):
        """Move cells or networks by adding to their (x, y, z) coordinates.

        This function is used only for visualization. It does not affect the simulation.

        Args:
            x: The amount to move in the x direction in um.
            y: The amount to move in the y direction in um.
            z: The amount to move in the z direction in um.
            update_nodes: Whether `.nodes` should be updated or not. Setting this to
                `False` largely speeds up moving, especially for big networks, but
                `.nodes` or `.show` will not show the new xyz coordinates.
        """
        for i in self._branches_in_view:
            self.base.xyzr[i][:, :3] += np.array([x, y, z])
        if update_nodes:
            self.compute_compartment_centers()

    def move_to(
        self,
        x: Union[float, np.ndarray] = 0.0,
        y: Union[float, np.ndarray] = 0.0,
        z: Union[float, np.ndarray] = 0.0,
        update_nodes: bool = False,
    ):
        """Move cells or networks to a location (x, y, z).

        If x, y, and z are floats, then the first compartment of the first branch
        of the first cell is moved to that float coordinate, and everything else is
        shifted by the difference between that compartment's previous coordinate and
        the new float location.

        If x, y, and z are arrays, then they must each have a length equal to the number
        of cells being moved. Then the first compartment of the first branch of each
        cell is moved to the specified location.

        Args:
            update_nodes: Whether `.nodes` should be updated or not. Setting this to
                `False` largely speeds up moving, especially for big networks, but
                `.nodes` or `.show` will not show the new xyz coordinates.
        """
        # Test if any coordinate values are NaN which would greatly affect moving
        if np.any(np.concatenate(self.xyzr, axis=0)[:, :3] == np.nan):
            raise ValueError(
                "NaN coordinate values detected. Shift amounts cannot be computed. "
                "Please run compute_xyzr() or assign initial coordinate values."
            )

        # can only iterate over cells for networks
        # lambda makes sure that generator can be created multiple times
        base_is_net = self.base._current_view == "network"
        cells = lambda: (self.cells if base_is_net else [self])

        root_xyz_cells = np.array([c.xyzr[0][0, :3] for c in cells()])
        root_xyz = root_xyz_cells[0] if isinstance(x, float) else root_xyz_cells
        move_by = np.array([x, y, z]).T - root_xyz

        if len(move_by.shape) == 1:
            move_by = np.tile(move_by, (len(self._cells_in_view), 1))

        for cell, offset in zip(cells(), move_by):
            for idx in cell._branches_in_view:
                self.base.xyzr[idx][:, :3] += offset
        if update_nodes:
            self.compute_compartment_centers()

    def rotate(
        self, degrees: float, rotation_axis: str = "xy", update_nodes: bool = False
    ):
        """Rotate jaxley modules clockwise. Used only for visualization.

        This function is used only for visualization. It does not affect the simulation.

        Args:
            degrees: How many degrees to rotate the module by.
            rotation_axis: Either of {`xy` | `xz` | `yz`}.
        """
        degrees = degrees / 180 * np.pi
        if rotation_axis == "xy":
            dims = [0, 1]
        elif rotation_axis == "xz":
            dims = [0, 2]
        elif rotation_axis == "yz":
            dims = [1, 2]
        else:
            raise ValueError

        rotation_matrix = np.asarray(
            [[np.cos(degrees), np.sin(degrees)], [-np.sin(degrees), np.cos(degrees)]]
        )
        for i in self._branches_in_view:
            rot = np.dot(rotation_matrix, self.base.xyzr[i][:, dims].T).T
            self.base.xyzr[i][:, dims] = rot
        if update_nodes:
            self.compute_compartment_centers()

    def copy_node_property_to_edges(
        self,
        properties_to_import: Union[str, List[str]],
        pre_or_post: Union[str, List[str]] = ["pre", "post"],
    ) -> Module:
        """Copy a property that is in `node` over to `edges`.

        By default, `.edges` does not contain the properties (radius, length, cm,
        channel properties,...) of the pre- and post-synaptic compartments. This
        method allows to copy a property of the pre- and/or post-synaptic compartment
        to the edges. It is then accessible as `module.edges.pre_property_name` or
        `module.edges.post_property_name`.

        Note that, if you modify the node property _after_ having run
        `copy_node_property_to_edges`, it will not automatically update the value in
        `.edges`.

        Note that, if this method is called on a View (e.g.
        `net.cell(0).copy_node_property_to_edges`), then it will return a View, but
        it will _not_ modify the module itself.

        Args:
            properties_to_import: The name of the node properties that should be
                imported. To list all available properties, look at
                `module.nodes.columns`.
            pre_or_post: Whether to import only the pre-synaptic property ('pre'), only
                the post-synaptic property ('post'), or both (['pre', 'post']).

        Returns:
            A new module which has the property copied to the `nodes`.
        """
        # If a string is passed, wrap it as a list.
        if isinstance(pre_or_post, str):
            pre_or_post = [pre_or_post]
        if isinstance(properties_to_import, str):
            properties_to_import = [properties_to_import]

        for pre_or_post_val in pre_or_post:
            assert pre_or_post_val in ["pre", "post"]
            for property_to_import in properties_to_import:
                # Delete the column if it already exists. Otherwise it would exist
                # twice.
                if f"{pre_or_post_val}_{property_to_import}" in self.edges.columns:
                    self.edges.drop(
                        columns=f"{pre_or_post_val}_{property_to_import}", inplace=True
                    )

                self.edges = self.edges.join(
                    self.nodes[property_to_import],
                    on=f"{pre_or_post_val}_index",
                )
                self.edges = self.edges.rename(
                    columns={
                        property_to_import: f"{pre_or_post_val}_{property_to_import}"
                    }
                )


class View(Module):
    """Views are instances of Modules which only track a subset of the
    compartments / edges of the original module. Views support the same fundamental
    operations that Modules do, i.e. `set`, `make_trainable` etc., however Views
    allow to target specific parts of a Module, i.e. setting parameters for parts
    of a cell.

    Almost all methods in View are concerned with updating the attributes of the
    base Module, i.e. `self.base`, based on the indices in view. For example,
    `_channels_in_view` lists all channels, finds the subset set to `True` in
    `self.nodes` (currently in view) and returns the updated list such that we can set
    `self.channels = self._channels_in_view()`.

    For developers: To allow seamless operation on Views and Modules as if they were
    the same, the following needs to be ensured:

    1. We consider a Module to have everything in view.
    2. Views can display and keep track of how a module is traversed. But(!),
       do not support making changes or setting variables. This still has to be
       done in the base Module, i.e. `self.base`. In order to enssure that these
       changes only affects whatever is currently in view `self._nodes_in_view`,
       or `self._edges_in_view` among others have to be used. Operating on nodes
       currently in view can for example be done with
       `self.base.node.loc[self._nodes_in_view]`
    3. Every attribute of Module that changes based on what's in view, i.e. `xyzr`,
       needs to modified when View is instantiated. I.e. `xyzr` of `cell.branch(0)`,
       should be `[self.base.xyzr[0]]` This could be achieved via:
       `[self.base.xyzr[b] for b in self._branches_in_view]`.


    For developers: Below is an example to make methods of Module compatible with View:

    .. code-block:: python
        # Use data in view to return something.
        def count_small_branches(self):
            # no need to use self.base.attr + viewed indices,
            # since no change is made to the attr in question (nodes)
            comp_lens = self.nodes["length"]
            branch_lens = comp_lens.groupby("global_branch_index").sum()
            return np.sum(branch_lens < 10)

        # Change data in view.
        def change_attr_in_view(self):
            # changes to attrs have to be made via self.base.attr + viewed indices
            a = func1(self.base.attr1[self._cells_in_view])
            b = func2(self.base.attr2[self._edges_in_view])
            self.base.attr3[self._branches_in_view] = a + b
    """

    def __init__(
        self,
        pointer: Union[Module, View],
        nodes: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        comp_edge_condition: str = "source_or_sink",
    ):
        self.base: Module = pointer.base  # Point to the base module.
        self._scope = pointer._scope  # forward view

        # attrs with a static view
        self.initialized_solver = pointer.initialized_solver
        self.initialized_syns = pointer.initialized_syns
        self.allow_make_trainable = pointer.allow_make_trainable

        # attrs affected by view
        # indices need to be update first, since they are used in the following
        self._set_inds_in_view(
            pointer, nodes, edges, comp_edge_condition=comp_edge_condition
        )
        self.ncomp = pointer.ncomp

        self.nodes = pointer.nodes.loc[self._nodes_in_view]
        ptr_edges = pointer.edges
        self.edges = (
            ptr_edges if ptr_edges.empty else ptr_edges.loc[self._edges_in_view]
        )
        ptr_edges = pointer._comp_edges
        self._comp_edges = (
            ptr_edges if ptr_edges.empty else ptr_edges.loc[self._comp_edges_in_view]
        )
        ptr_nodes = pointer._branchpoints
        self._branchpoints = (
            ptr_nodes if ptr_nodes.empty else ptr_nodes.loc[self._branchpoints_in_view]
        )
        self.xyzr = self._xyzr_in_view()
        self.ncomp = len(self.nodes)
        self.total_nbranches = len(self._branches_in_view)
        self.nbranches_per_cell = self._nbranches_per_cell_in_view()
        self._cumsum_nbranches = jnp.cumsum(np.asarray(self.nbranches_per_cell))
        self.comb_branches_in_each_level = pointer.comb_branches_in_each_level
        self.branch_edges = pointer.branch_edges.loc[self._branch_edges_in_view]
        self.ncomp_per_branch = self.base.ncomp_per_branch[self._branches_in_view]
        self.cumsum_ncomp = cumsum_leading_zero(self.ncomp_per_branch)

        self.synapse_names = np.unique(self.edges["type"]).tolist()
        self._set_synapses_in_view(pointer)

        ptr_recs = pointer.recordings
        self.recordings = (
            pd.DataFrame()
            if ptr_recs.empty
            else ptr_recs.loc[ptr_recs["rec_index"].isin(self._comps_in_view)]
        )

        self.channels = self._channels_in_view(pointer)
        self.pumps = self._pumps_in_view(pointer)
        self.pumped_ions = []
        for pump in self.pumps:
            if pump.ion_name not in self.pumped_ions:
                self.pumped_ions.append(pump.ion_name)
        # Diffusion always in entire module.
        self.diffusion_states = pointer.diffusion_states
        self.membrane_current_names = [c.current_name for c in self.channels]
        self.synapse_current_names = pointer.synapse_current_names
        self._set_trainables_in_view()  # run after synapses and channels
        self.num_trainable_params = (
            np.sum([len(inds) for inds in self.indices_set_by_trainables])
            .astype(int)
            .item()
        )

        self.ncomp_per_branch = pointer.base.ncomp_per_branch[self._branches_in_view]
        self.comb_parents = self.base.comb_parents[self._branches_in_view]
        self._set_externals_in_view()
        self.group_names = self.base.group_names

        self.jaxnodes, self.jaxedges = self._jax_arrays_in_view(
            pointer
        )  # run after trainables

        self._current_view = "view"  # if not instantiated via `comp`, `cell` etc.
        self._update_local_indices()

        # TODO FROM #447:
        self.debug_states = pointer.debug_states

        if len(self.nodes) == 0:
            raise ValueError("Nothing in view. Check your indices.")

    def _set_inds_in_view(
        self,
        pointer: Union[Module, View],
        nodes: np.ndarray,
        edges: np.ndarray,
        comp_edge_condition="source_or_sink",
    ):
        """Update node and edge indices to list only those currently in view.

        Args:
            comp_edge_condition: Either of
                {`source_and_sink`, `source_or_sink`, `endpoint`, `startpoint`}. Sets
                how the `comp_edges` are built. If `source_and_sink`, an edge between
                compartments is kept only if source and sink compartments are within
                the view. If `source_or_sink`, an edge is kept if either the source
                or the sink are within the view. If `endpoint`, then the edge is kept
                if the compartment is in source or sink and if it is an edge between
                parent compartment and branchpoint. If `startpoint`, then the edge is
                kept if the compartment is in source or sink and if it is an edge
                between child compartment and branchpoint. This is used because we
                want different treatment of the `comp_edges` depending on whether we
                index with `.branch()` (`source_or_sink`), `.comp()`
                (`source_and_sink`), `.loc(0)` (`startpoint`), or `.loc(1)`
                (`endpoint`).
        """
        # set nodes and edge indices in view
        has_node_inds = nodes is not None
        has_edge_inds = edges is not None
        self._edges_in_view = pointer._edges_in_view
        self._nodes_in_view = pointer._nodes_in_view

        self._comp_edges_in_view = pointer._comp_edges_in_view
        self._branchpoints_in_view = pointer._branchpoints_in_view

        if not has_edge_inds and has_node_inds:
            base_edges = self.base.edges
            self._nodes_in_view = nodes
            incl_comps = pointer.nodes.loc[self._nodes_in_view].index.unique()
            if not base_edges.empty:
                pre = base_edges["pre_index"].isin(incl_comps).to_numpy()
                post = base_edges["post_index"].isin(incl_comps).to_numpy()
                possible_edges_in_view = base_edges.index.to_numpy()[
                    (pre & post).flatten()
                ]
                self._edges_in_view = np.intersect1d(
                    possible_edges_in_view, self._edges_in_view
                )
            base_comp_edges = self.base._comp_edges
            base_branchpoints = self.base._branchpoints
            if not base_comp_edges.empty:
                possible_edges_in_view = _get_comp_edges_in_view(
                    base_comp_edges, incl_comps, comp_edge_condition
                )
                self._comp_edges_in_view = np.intersect1d(
                    possible_edges_in_view, self._comp_edges_in_view
                )
                all_comps = base_comp_edges.loc[self._comp_edges_in_view][
                    "sink"
                ].to_numpy()
                condition = base_branchpoints.index.isin(all_comps)
                self._branchpoints_in_view = base_branchpoints.loc[
                    condition
                ].index.to_numpy()
        elif not has_node_inds and has_edge_inds:
            base_nodes = self.base.nodes
            self._edges_in_view = edges
            incl_comps = pointer.edges.loc[
                self._edges_in_view, ["pre_index", "post_index"]
            ]
            incl_comps = np.unique(incl_comps.to_numpy().flatten())
            where_comps = base_nodes.index.isin(incl_comps)
            possible_nodes_in_view = base_nodes.index[where_comps].to_numpy()
            self._nodes_in_view = np.intersect1d(
                possible_nodes_in_view, self._nodes_in_view
            )
        elif has_node_inds and has_edge_inds:
            self._nodes_in_view = nodes
            self._edges_in_view = edges

    def _jax_arrays_in_view(self, pointer: Union[Module, View]):
        """Update jaxnodes/jaxedges to show only those currently in view."""
        a_intersects_b_at = lambda a, b: jnp.intersect1d(a, b, return_indices=True)[1]
        jaxnodes = {} if pointer.jaxnodes is not None else None
        if self.jaxnodes is not None:
            comp_inds = pointer.jaxnodes["global_comp_index"]
            common_inds = a_intersects_b_at(comp_inds, self._nodes_in_view)
            jaxnodes = {
                k: v[common_inds]
                for k, v in pointer.jaxnodes.items()
                if len(common_inds) > 0
            }

        jaxedges = {} if pointer.jaxedges is not None else None
        if pointer.jaxedges is not None:
            for key, values in self.base.jaxedges.items():
                if (syn_name := key.split("_")[0]) in self.synapse_names:
                    syn_edges = self.base.edges[self.base.edges["type"] == syn_name]
                    inds = np.intersect1d(
                        self._edges_in_view, syn_edges.index, return_indices=True
                    )[2]
                    if len(inds) > 0:
                        jaxedges[key] = values[inds]
        return jaxnodes, jaxedges

    def _set_externals_in_view(self):
        """Update external inputs to show only those currently in view."""
        self.externals = {}
        self.external_inds = {}
        for (name, inds), data in zip(
            self.base.external_inds.items(), self.base.externals.values()
        ):
            in_view = np.isin(inds, self._nodes_in_view)
            inds_in_view = inds[in_view]
            if len(inds_in_view) > 0:
                self.externals[name] = data[in_view]
                self.external_inds[name] = inds_in_view

    def _filter_trainables(
        self, is_viewed: bool = True
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Filters the trainables inside and outside of the view.

        Trainables are split between `indices_set_by_trainables` and `trainable_params`
        and can be shared between multiple compartments / branches etc, which makes it
        difficult to filter them based on the current view w.o. destroying the
        original structure.

        This method filters `indices_set_by_trainables` for the indices that are
        currently in view (or not in view) and returns the corresponding trainable
        parameters and indices such that the sharing behavior is preserved as much as
        possible.

        Args:
            is_viewed: Toggles between returning the trainables and inds
                currently inside or outside of the scope of View."""
        índices_set_by_trainables_in_view = []
        trainable_params_in_view = []
        for inds, params in zip(
            self.base.indices_set_by_trainables, self.base.trainable_params
        ):
            pkey, pval = next(iter(params.items()))
            trainable_inds_in_view = None
            if pkey in sum(
                [list(c.channel_params.keys()) for c in self.base.channels], []
            ):
                trainable_inds_in_view = np.intersect1d(inds, self._nodes_in_view)
            elif pkey in sum(
                [list(s.synapse_params.keys()) for s in self.base.synapses], []
            ):
                trainable_inds_in_view = np.intersect1d(inds, self._edges_in_view)

            in_view = is_viewed == np.isin(inds, trainable_inds_in_view)
            completely_in_view = in_view.all(axis=1)
            partially_in_view = in_view.any(axis=1) & ~completely_in_view

            trainable_params_in_view.append(
                {k: v[completely_in_view] for k, v in params.items()}
            )
            trainable_params_in_view.append(
                {k: v[partially_in_view] for k, v in params.items()}
            )

            índices_set_by_trainables_in_view.append(inds[completely_in_view])
            partial_inds = inds[partially_in_view][in_view[partially_in_view]]

            # the indexing i.e. `inds[partially_in_view]` reshapes `inds`. Since the shape
            # determines how parameters are shared, `inds` has to be returned to its
            # original shape.
            if inds.shape[0] > 1 and partial_inds.shape != (0,):
                partial_inds = partial_inds.reshape(-1, 1)
            if inds.shape[1] > 1 and partial_inds.shape != (0,):
                partial_inds = partial_inds.reshape(1, -1)

            índices_set_by_trainables_in_view.append(partial_inds)

        indices_set_by_trainables = [
            inds for inds in índices_set_by_trainables_in_view if len(inds) > 0
        ]
        trainable_params = [
            p for p in trainable_params_in_view if len(next(iter(p.values()))) > 0
        ]
        return indices_set_by_trainables, trainable_params

    def _set_trainables_in_view(self):
        """Set `trainable_params` and `indices_set_by_trainables` to show only those in view."""
        trainables = self._filter_trainables()

        # note for `branch.comp(0).make_trainable("X"); branch.make_trainable("X")`
        # `view = branch.comp(0)` will have duplicate training params.
        self.indices_set_by_trainables = trainables[0]
        self.trainable_params = trainables[1]

    def _channels_in_view(self, pointer: Union[Module, View]) -> List[Channel]:
        """Set channels to show only those in view."""
        names = [name._name for name in pointer.channels]
        channel_in_view = self.nodes[names].any(axis=0)
        channel_in_view = channel_in_view[channel_in_view].index
        return [c for c in pointer.channels if c._name in channel_in_view]

    def _pumps_in_view(self, pointer: Union[Module, View]) -> List[Pump]:
        """Set pumps to show only those in view."""
        names = [name._name for name in pointer.pumps]
        pump_in_view = self.nodes[names].any(axis=0)
        pump_in_view = pump_in_view[pump_in_view].index
        return [c for c in pointer.pumps if c._name in pump_in_view]

    def _set_synapses_in_view(self, pointer: Union[Module, View]):
        """Set synapses to show only those in view."""
        viewed_synapses = []
        viewed_params = []
        viewed_states = []
        if pointer.synapses is not None:
            for syn in pointer.synapses:
                if syn is not None:  # needed for recursive viewing
                    in_view = syn._name in self.synapse_names
                    viewed_synapses += (
                        [syn] if in_view else [None]
                    )  # padded with None to keep indices consistent
                    viewed_params += list(syn.synapse_params.keys()) if in_view else []
                    viewed_states += list(syn.synapse_states.keys()) if in_view else []
        self.synapses = viewed_synapses
        self.synapse_param_names = viewed_params
        self.synapse_state_names = viewed_states

    def _nbranches_per_cell_in_view(self) -> np.ndarray:
        cell_nodes = self.nodes.groupby("global_cell_index")
        return cell_nodes["global_branch_index"].nunique().to_list()

    def _xyzr_in_view(self) -> List[np.ndarray]:
        """Return xyzr coordinates of every branch that is in `_branches_in_view`.

        If a branch is not completely in view, the coordinates are interpolated."""
        xyzr = []
        viewed_ncomp_for_branch = self.nodes.groupby("global_branch_index").size()
        for i in self._branches_in_view:
            xyzr_i = self.base.xyzr[i]
            ncomp_i = self.base.ncomp_per_branch[i]
            global_comp_offset = self.base.cumsum_ncomp[i]
            global_comp_inds = self.nodes["global_comp_index"]
            if viewed_ncomp_for_branch.loc[i] != ncomp_i:
                local_inds = (
                    global_comp_inds.loc[
                        self.nodes["global_branch_index"] == i
                    ].to_numpy()
                    - global_comp_offset
                )
                local_ind_range = np.arange(min(local_inds), max(local_inds) + 1)
                inds = [i if i in local_inds else None for i in local_ind_range]
                comp_ends = np.linspace(0, 1, ncomp_i + 1)
                locs = np.hstack(
                    [comp_ends[[i, i + 1]] if i is not None else [np.nan] for i in inds]
                )
                xyzr.append(interpolate_xyzr(locs, xyzr_i).T)
            else:
                xyzr.append(xyzr_i)
        return xyzr

    # needs abstract method to allow init of View
    # forward to self.base for now
    def _init_morph_jax_spsolve(self):
        return self.base._init_morph_jax_spsolve()

    # needs abstract method to allow init of View
    # forward to self.base for now
    def _init_morph_jaxley_spsolve(self):
        return self.base._init_morph_jax_spsolve()

    @property
    def _branches_in_view(self) -> np.ndarray:
        """Lists the global branch indices which are currently part of the view."""
        return self.nodes["global_branch_index"].unique()

    @property
    def _cells_in_view(self) -> np.ndarray:
        """Lists the global cell indices which are currently part of the view."""
        return self.nodes["global_cell_index"].unique()

    @property
    def _comps_in_view(self) -> np.ndarray:
        """Lists the global compartment indices which are currently part of the view."""
        return self.nodes["global_comp_index"].unique()

    @property
    def _branch_edges_in_view(self) -> np.ndarray:
        """Lists the global branch edge indices which are currently part of the view."""
        incl_branches = self.nodes["global_branch_index"].unique()
        pre = self.base.branch_edges["parent_branch_index"].isin(incl_branches)
        post = self.base.branch_edges["child_branch_index"].isin(incl_branches)
        viewed_branch_inds = self.base.branch_edges.index.to_numpy()[pre & post]
        return viewed_branch_inds

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


########################################################################################
###################################### TO GRAPH ########################################
########################################################################################


def to_graph(
    module: "jx.Module", synapses: bool = False, channels: bool = False
) -> nx.DiGraph:
    """Export a `jx.Module` as a networkX compartment graph.

    Constructs a nx.DiGraph from the module. Each compartment in the module
    is represented by a node in the graph. The edges between the nodes represent
    the connections between the compartments. These edges can either be connections
    between compartments within the same branch, between different branches or
    even between different cells. In the latter case the synapse parameters
    are stored as edge attributes. Only function allows one synapse per edge!
    Additionally, global attributes of the module, for example `ncomp`, are stored as
    graph attributes.

    Exported graphs can be imported again to `jaxley` using the `from_graph` method.

    Args:
        module: A jaxley module or view instance.
        synapses: Whether to export synapses to the graph.
        channels: Whether to export ion channels to the graph.

    Returns:
        A networkx compartment. Has the same structure as a graph built with
        `build_compartment_graph()`.

    Example usage
    ^^^^^^^^^^^^^

    ::

        cell = jx.read_swc("path_to_swc.swc", ncomp=1)
        comp_graph = to_graph(cell)
    """
    module_graph = nx.DiGraph()

    # add global attrs
    module_graph.graph["type"] = module.__class__.__name__.lower()
    for attr in [
        "ncomp",
        "externals",
        "external_inds",
        "recordings",
        "trainable_params",
        "indices_set_by_trainables",
    ]:
        module_graph.graph[attr] = getattr(module, attr)

    # add nodes
    nodes = module.nodes.copy()
    nodes = nodes.drop([col for col in nodes.columns if "local" in col], axis=1)
    nodes.columns = [col.replace("global_", "") for col in nodes.columns]

    if channels:
        module_graph.graph["channels"] = module.channels
        module_graph.graph["membrane_current_names"] = [
            c.current_name for c in module.channels
        ]
    else:
        for c in module.channels:
            nodes = nodes.drop(c.name, axis=1)
            # errors="ignore" because some channels might have the same parameter or
            # state name (if the channels share parameters).
            nodes = nodes.drop(list(c.channel_params), axis=1, errors="ignore")
            nodes = nodes.drop(list(c.channel_states), axis=1, errors="ignore")

    nodes["type"] = "comp"
    for col in nodes.columns:  # col wise adding preserves dtypes
        module_graph.add_nodes_from(nodes[[col]].to_dict(orient="index").items())

    module._branchpoints["type"] = "branchpoint"
    for col in module._branchpoints.columns:
        module_graph.add_nodes_from(
            module._branchpoints[[col]].to_dict(orient="index").items()
        )

    module_graph.graph["group_names"] = module.group_names

    for i, branch_data in nodes.groupby("branch_index"):
        inds = branch_data.index.values
        # Special handling for xyzr. In the module, xyzr is currently stored in a list,
        # where each list entry indicates one _branch_. In the `comp_graph`, each
        # compartment is assigned its own `xyzr`. Here, we cast from the branch
        # representation to the compartment representation.
        xyzr = module.xyzr[i]
        ncomp_per_branch = len(branch_data)
        xyzr_per_comp = np.array_split(xyzr, ncomp_per_branch)
        for i, comp_index in enumerate(inds):
            module_graph.nodes[comp_index]["xyzr"] = xyzr_per_comp[i]

    edges = module._comp_edges.copy()
    condition1 = edges["type"].isin([2, 3])
    condition2 = edges["type"] == 0
    condition3 = edges["source"] < edges["sink"]
    edges = edges[condition1 | (condition3 & condition2)][["source", "sink"]]
    if len(edges) > 0:
        module_graph.add_edges_from(edges.to_numpy())
    module_graph.graph["type"] = module.__class__.__name__.lower()

    if synapses:
        syn_edges = module.edges.copy()
        multiple_syn_per_edge = syn_edges[["pre_index", "post_index"]].duplicated(
            keep=False
        )
        dupl_inds = multiple_syn_per_edge.index[multiple_syn_per_edge].values
        if multiple_syn_per_edge.any():
            warn(
                f"CAUTION: Synapses {dupl_inds} are connecting the same compartments. "
                "Exporting synapses to the graph only works if the same two "
                "compartments are connected by at most one synapse."
            )
        module_graph.graph["synapses"] = module.synapses
        module_graph.graph["synapse_param_names"] = module.synapse_param_names
        module_graph.graph["synapse_state_names"] = module.synapse_state_names
        module_graph.graph["synapse_names"] = module.synapse_names
        module_graph.graph["synapse_current_names"] = module.synapse_current_names

        syn_edges.columns = syn_edges.columns
        syn_edges["syn_type"] = syn_edges["type"]
        syn_edges["type"] = "synapse"
        syn_edges = syn_edges.set_index(["pre_index", "post_index"])

        if not syn_edges.empty:
            for (i, j), edge_data in syn_edges.iterrows():
                module_graph.add_edge(i, j, **edge_data.to_dict())

    return module_graph
