# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import copy
from typing import Callable, List, Optional, Tuple
from warnings import warn

import networkx as nx
import numpy as np

# from jaxley.io.graph import from_graph
import jaxley as jx


def swc_to_jaxley(
    fname: str,
    max_branch_len: float = 100.0,
    sort: bool = True,
    num_lines: Optional[int] = None,
) -> Tuple[List[int], List[float], List[Callable], List[float], List[np.ndarray]]:
    """Read an SWC file and bring morphology into `jaxley` compatible formats.

    Args:
        fname: Path to swc file.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`.
        num_lines: Number of lines of the SWC file to read.
    """
    content = np.loadtxt(fname)[:num_lines]
    types = content[:, 1]
    is_single_point_soma = types[0] == 1 and types[1] != 1

    if is_single_point_soma:
        # Warn here, but the conversion of the length happens in `_compute_pathlengths`.
        warn(
            "Found a soma which consists of a single traced point. `Jaxley` "
            "interprets this soma as a spherical compartment with radius "
            "specified in the SWC file, i.e. with surface area 4*pi*r*r."
        )
    sorted_branches, types = _split_into_branches_and_sort(
        content,
        max_branch_len=max_branch_len,
        is_single_point_soma=is_single_point_soma,
        sort=sort,
    )

    parents = _build_parents(sorted_branches)
    each_length = _compute_pathlengths(
        sorted_branches, content[:, 1:6], is_single_point_soma=is_single_point_soma
    )
    pathlengths = [np.sum(length_traced) for length_traced in each_length]
    for i, pathlen in enumerate(pathlengths):
        if pathlen == 0.0:
            warn("Found a segment with length 0. Clipping it to 1.0")
            pathlengths[i] = 1.0
    radius_fns = _radius_generating_fns(
        sorted_branches, content[:, 5], each_length, parents, types
    )

    if np.sum(np.asarray(parents) == -1) > 1.0:
        parents = np.asarray([-1] + parents)
        parents[1:] += 1
        parents = parents.tolist()
        pathlengths = [0.1] + pathlengths
        radius_fns = [lambda x: content[0, 5] * np.ones_like(x)] + radius_fns
        sorted_branches = [[0]] + sorted_branches

        # Type of padded section is assumed to be of `custom` type:
        # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        types = [5.0] + types

    all_coords_of_branches = []
    for i, branch in enumerate(sorted_branches):
        # Remove 1 because `content` is an array that is indexed from 0.
        branch = np.asarray(branch) - 1

        # Deal with additional branch that might have been added above in the lines
        # `if np.sum(np.asarray(parents) == -1) > 1.0:`
        branch[branch < 0] = 0

        # Get traced coordinates of the branch.
        coords_of_branch = content[branch, 2:6]
        all_coords_of_branches.append(coords_of_branch)

    return parents, pathlengths, radius_fns, types, all_coords_of_branches


def _split_into_branches_and_sort(
    content: np.ndarray,
    max_branch_len: float,
    is_single_point_soma: bool,
    sort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    branches, types = _split_into_branches(content, is_single_point_soma)
    branches, types = _split_long_branches(
        branches,
        types,
        content,
        max_branch_len,
        is_single_point_soma=is_single_point_soma,
    )

    if sort:
        first_val = np.asarray([b[0] for b in branches])
        sorting = np.argsort(first_val, kind="mergesort")
        sorted_branches = [branches[s] for s in sorting]
        sorted_types = [types[s] for s in sorting]
    else:
        sorted_branches = branches
        sorted_types = types
    return sorted_branches, sorted_types


def _split_long_branches(
    branches: np.ndarray,
    types: np.ndarray,
    content: np.ndarray,
    max_branch_len: float,
    is_single_point_soma: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    pathlengths = _compute_pathlengths(
        branches, content[:, 1:6], is_single_point_soma=is_single_point_soma
    )
    pathlengths = [np.sum(length_traced) for length_traced in pathlengths]
    split_branches = []
    split_types = []
    for branch, type, length in zip(branches, types, pathlengths):
        num_subbranches = 1
        split_branch = [branch]
        while length > max_branch_len:
            num_subbranches += 1
            split_branch = _split_branch_equally(branch, num_subbranches)
            lengths_of_subbranches = _compute_pathlengths(
                split_branch,
                coords=content[:, 1:6],
                is_single_point_soma=is_single_point_soma,
            )
            lengths_of_subbranches = [
                np.sum(length_traced) for length_traced in lengths_of_subbranches
            ]
            length = max(lengths_of_subbranches)
            if num_subbranches > 10:
                warn(
                    """`num_subbranches > 10`, stopping to split. Most likely your
                     SWC reconstruction is not dense and some neighbouring traced
                     points are farther than `max_branch_len` apart."""
                )
                break
        split_branches += split_branch
        split_types += [type] * num_subbranches

    return split_branches, split_types


def _split_branch_equally(branch: np.ndarray, num_subbranches: int) -> List[np.ndarray]:
    num_points_each = len(branch) // num_subbranches
    branches = [branch[:num_points_each]]
    for i in range(1, num_subbranches - 1):
        branches.append(branch[i * num_points_each - 1 : (i + 1) * num_points_each])
    branches.append(branch[(num_subbranches - 1) * num_points_each - 1 :])
    return branches


def _split_into_branches(
    content: np.ndarray, is_single_point_soma: bool
) -> Tuple[np.ndarray, np.ndarray]:
    prev_ind = None
    prev_type = None
    n_branches = 0

    # Branch inds will contain the row identifier at which a branch point occurs
    # (i.e. the row of the parent of two branches).
    branch_inds = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        current_type = c[1]
        if current_parent != prev_ind or current_type != prev_type:
            branch_inds.append(int(current_parent))
            n_branches += 1
        prev_ind = current_ind
        prev_type = current_type

    all_branches = []
    current_branch = []
    all_types = []

    # Loop over every line in the SWC file.
    for c in content:
        current_ind = c[0]  # First col is row_identifier
        current_parent = c[-1]  # Last col is parent in SWC specification.
        if current_parent == -1:
            all_types.append(c[1])
        else:
            current_type = c[1]

        if current_parent == -1 and is_single_point_soma and current_ind == 1:
            all_branches.append([int(current_ind)])
            all_types.append(int(current_type))

        # Either append the current point to the branch, or add the branch to
        # `all_branches`.
        if current_parent in branch_inds[1:]:
            if len(current_branch) > 1:
                all_branches.append(current_branch)
                all_types.append(current_type)
            current_branch = [int(current_parent), int(current_ind)]
        else:
            current_branch.append(int(current_ind))

    # Append the final branch (intermediate branches are already appended five lines
    # above.)
    all_branches.append(current_branch)
    return all_branches, all_types


def _build_parents(all_branches: List[np.ndarray]) -> List[int]:
    parents = [None] * len(all_branches)
    all_last_inds = [b[-1] for b in all_branches]
    for i, branch in enumerate(all_branches):
        parent_ind = branch[0]
        ind = np.where(np.asarray(all_last_inds) == parent_ind)[0]
        if len(ind) > 0 and ind != i:
            parents[i] = ind[0]
        else:
            assert (
                parent_ind == 1
            ), """Trying to connect a segment to the beginning of 
            another segment. This is not allowed. Please create an issue on github."""
            parents[i] = -1

    return parents


def _radius_generating_fns(
    all_branches: np.ndarray,
    radiuses: np.ndarray,
    each_length: np.ndarray,
    parents: np.ndarray,
    types: np.ndarray,
) -> List[Callable]:
    """For all branches in a cell, returns callable that return radius given loc."""
    radius_fns = []
    for i, branch in enumerate(all_branches):
        rads_in_branch = radiuses[np.asarray(branch) - 1]
        if parents[i] > -1 and types[i] != types[parents[i]]:
            # We do not want to linearly interpolate between the radius of the previous
            # branch if a new type of neurite is found (e.g. switch from soma to
            # apical). From looking at the SWC from n140.swc I believe that this is
            # also what NEURON does.
            rads_in_branch[0] = rads_in_branch[1]
        radius_fn = _radius_generating_fn(
            radiuses=rads_in_branch, each_length=each_length[i]
        )
        # Beause SWC starts counting at 1, but numpy counts from 0.
        # ind_of_branch_endpoint = np.asarray(b) - 1
        radius_fns.append(radius_fn)
    return radius_fns


def _radius_generating_fn(radiuses: np.ndarray, each_length: np.ndarray) -> Callable:
    # Avoid division by 0 with the `summed_len` below.
    each_length[each_length < 1e-8] = 1e-8
    summed_len = np.sum(each_length)
    cutoffs = np.cumsum(np.concatenate([np.asarray([0]), each_length])) / summed_len
    cutoffs[0] -= 1e-8
    cutoffs[-1] += 1e-8

    # We have to linearly interpolate radiuses, therefore we need at least two radiuses.
    # However, jaxley allows somata which consist of a single traced point (i.e.
    # just one radius). Therefore, we just `tile` in order to generate an artificial
    # endpoint and startpoint radius of the soma.
    if len(radiuses) == 1:
        radiuses = np.tile(radiuses, 2)

    def radius(loc: float) -> float:
        """Function which returns the radius via linear interpolation."""
        index = np.digitize(loc, cutoffs, right=False)
        left_rad = radiuses[index - 1]
        right_rad = radiuses[index]
        left_loc = cutoffs[index - 1]
        right_loc = cutoffs[index]
        loc_within_bin = (loc - left_loc) / (right_loc - left_loc)
        return left_rad + (right_rad - left_rad) * loc_within_bin

    return radius


def _compute_pathlengths(
    all_branches: np.ndarray, coords: np.ndarray, is_single_point_soma: bool
) -> List[np.ndarray]:
    """
    Args:
        coords: Has shape (num_traced_points, 5), where `5` is (type, x, y, z, radius).
    """
    branch_pathlengths = []
    for b in all_branches:
        coords_in_branch = coords[np.asarray(b) - 1]
        if len(coords_in_branch) > 1:
            # If the branch starts at a different neurite (e.g. the soma) then NEURON
            # ignores the distance from that initial point. To reproduce, use the
            # following SWC dummy file and read it in NEURON (and Jaxley):
            # 1 1 0.00 0.0 0.0 6.0 -1
            # 2 2 9.00 0.0 0.0 0.5 1
            # 3 2 10.0 0.0 0.0 0.3 2
            types = coords_in_branch[:, 0]
            if int(types[0]) == 1 and int(types[1]) != 1 and is_single_point_soma:
                coords_in_branch[0] = coords_in_branch[1]

            # Compute distances between all traced points in a branch.
            point_diffs = np.diff(coords_in_branch, axis=0)
            dists = np.sqrt(
                point_diffs[:, 1] ** 2 + point_diffs[:, 2] ** 2 + point_diffs[:, 3] ** 2
            )
        else:
            # Jaxley uses length and radius for every compartment and assumes the
            # surface area to be 2*pi*r*length. For branches consisting of a single
            # traced point we assume for them to have area 4*pi*r*r. Therefore, we have
            # to set length = 2*r.
            radius = coords_in_branch[0, 4]  # txyzr -> 4 is radius.
            dists = np.asarray([2 * radius])
        branch_pathlengths.append(dists)
    return branch_pathlengths


def read_swc(
    fname: str,
    nseg: int,
    max_branch_len: float = 300.0,
    min_radius: Optional[float] = None,
    assign_groups: bool = False,
):
    """Reads SWC file into a `jx.Cell`.

    Jaxley assumes cylindrical compartments and therefore defines length and radius
    for every compartment. The surface area is then 2*pi*r*length. For branches
    consisting of a single traced point we assume for them to have area 4*pi*r*r.
    Therefore, in these cases, we set lenght=2*r.

    Args:
        fname: Path to the swc file.
        nseg: The number of compartments per branch.
        max_branch_len: If a branch is longer than this value it is split into two
            branches.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        assign_groups: If True, then the identity of reconstructed points in the SWC
            file will be used to generate groups `undefined`, `soma`, `axon`, `basal`,
            `apical`, `custom`. See here:
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    """
    parents, pathlengths, radius_fns, types, coords_of_branches = swc_to_jaxley(
        fname, max_branch_len=max_branch_len, sort=True, num_lines=None
    )
    nbranches = len(parents)

    non_split = 1 / nseg
    range_ = np.linspace(non_split / 2, 1 - non_split / 2, nseg)

    comp = jx.Compartment()
    branch = jx.Branch([comp for _ in range(nseg)])
    cell = jx.Cell(
        [branch for _ in range(nbranches)], parents=parents, xyzr=coords_of_branches
    )

    radiuses = np.asarray([radius_fns[b](range_) for b in range(len(parents))])
    radiuses_each = radiuses.ravel(order="C")
    if min_radius is None:
        assert np.all(
            radiuses_each > 0.0
        ), "Radius 0.0 in SWC file. Set `read_swc(..., min_radius=...)`."
    else:
        radiuses_each[radiuses_each < min_radius] = min_radius

    lengths_each = np.repeat(pathlengths, nseg) / nseg

    cell.set("length", lengths_each)
    cell.set("radius", radiuses_each)

    # Description of SWC file format:
    # http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    ind_name_lookup = {
        0: "undefined",
        1: "soma",
        2: "axon",
        3: "basal",
        4: "apical",
        5: "custom",
    }
    types = np.asarray(types).astype(int)
    if assign_groups:
        for type_ind in np.unique(types):
            if type_ind < 5.5:
                name = ind_name_lookup[type_ind]
            else:
                name = f"custom{type_ind}"
            indices = np.where(types == type_ind)[0].tolist()
            if len(indices) > 0:
                cell.branch(indices).add_to_group(name)
    return cell


def swc_to_graph(fname, num_lines=None, sort=True) -> nx.DiGraph:
    i_id_xyzr_p = np.loadtxt(fname)[:num_lines]

    graph = nx.DiGraph()
    graph.add_nodes_from(
        (
            (int(i), {"id": int(id), "x": x, "y": y, "z": z, "r": r})
            for i, id, x, y, z, r, p in i_id_xyzr_p
        )
    )
    graph.add_edges_from([(p, i) for p, i in i_id_xyzr_p[:, [-1, 0]] if p != -1])
    graph = nx.relabel_nodes(graph, {i: i - 1 for i in graph.nodes})
    graph.graph["type"] = "swc"
    return graph


# def swc_to_graph_to_jaxley(
#     fname: str,
#     nseg: int = 4,
#     max_branch_len: float = 100.0,
#     sort: bool = True,
#     num_lines: Optional[int] = None,
# ):
#     graph = swc_to_graph(fname, num_lines, sort)
#     return from_graph(graph, nseg=nseg, max_branch_len=max_branch_len)
