from warnings import warn

import matplotlib.pyplot as plt
import numpy as np


def read_swc(fname: str, max_branch_len: float = 100.0, sort: bool = True):
    """Read an SWC file and bring morphology into `neurax` compatible formats.

    Args:
        fname: Path to swc file.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`.
    """
    content = np.loadtxt(fname)
    sorted_branches, types = _split_into_branches_and_sort(
        content, max_branch_len=max_branch_len, sort=sort
    )

    parents = _build_parents(sorted_branches)
    each_length = _compute_pathlengths(sorted_branches, content[:, 2:5])
    pathlengths = [np.sum(length_traced) for length_traced in each_length]
    for i, pathlen in enumerate(pathlengths):
        if pathlen == 0.0:
            warn("Found a segment with length 0. Clipping it to 1.0")
            pathlengths[i] = 1.0
    endpoint_radiuses = _extract_endpoint_radiuses(
        sorted_branches, content[:, 5], each_length
    )
    return parents, pathlengths, endpoint_radiuses, types


def _split_into_branches_and_sort(content, max_branch_len, sort=True):
    branches, types = _split_into_branches(content)
    branches, types = _split_long_branches(branches, types, content, max_branch_len)

    if sort:
        first_val = np.asarray([b[0] for b in branches])
        sorting = np.argsort(first_val, kind="mergesort")
        sorted_branches = [branches[s] for s in sorting]
        sorted_types = [types[s] for s in sorting]
    else:
        sorted_branches = branches
        sorted_types = types
    return sorted_branches, sorted_types


def _split_long_branches(branches, types, content, max_branch_len):
    pathlengths = _compute_pathlengths(branches, content[:, 2:5])
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
                split_branch, coords=content[:, 2:5]
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


def _split_branch_equally(branch, num_subbranches):
    num_points_each = len(branch) // num_subbranches
    branches = [branch[:num_points_each]]
    for i in range(1, num_subbranches - 1):
        branches.append(branch[i * num_points_each - 1 : (i + 1) * num_points_each])
    branches.append(branch[(num_subbranches - 1) * num_points_each - 1 :])
    return branches


def _remove_single_branch_artifacts(branches):
    """Check that all parents have two children. No only childs allowed!

    See GH #32. The reason this happens is that some branches (without branchings)
    are interrupted in their tracing. Here, we fuse these interrupted branches.
    """
    first_val = np.asarray([b[0] for b in branches])
    vals, counts = np.unique(first_val[1:], return_counts=True)
    one_vals = vals[counts == 1]
    for one_val in one_vals:
        loc = np.where(first_val == one_val)[0][0]
        solo_branch = branches[loc]
        del branches[loc]
        new_branches = []
        for b in branches:
            if b[-1] == one_val:
                new_branches.append(b + solo_branch)
            else:
                new_branches.append(b)
        branches = new_branches

    return branches


def _split_into_branches(content):
    prev_ind = None
    prev_type = None
    n_branches = 0
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
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent == -1:
            all_types.append(c[1])
        else:
            # `-1` because SWC starts counting at 1.
            current_type = c[1]
        if current_parent in branch_inds[1:]:
            if len(current_branch) > 1:
                all_branches.append(current_branch)
                all_types.append(current_type)
            current_branch = [int(current_parent), int(current_ind)]
        else:
            current_branch.append(int(current_ind))

    all_branches.append(current_branch)
    return all_branches, all_types


def _build_parents(all_branches):
    parents = [None] * len(all_branches)
    all_last_inds = [b[-1] for b in all_branches]
    for i, b in enumerate(all_branches):
        parent_ind = b[0]
        ind = np.where(np.asarray(all_last_inds) == parent_ind)[0]
        if len(ind) > 0 and ind != i:
            parents[i] = ind[0]
        else:
            parents[i] = -1

    return parents


def _extract_endpoint_radiuses(all_branches, radiuses, each_length):
    radius_fns = []
    for i in range(len(all_branches)):
        rads_in_branch = radiuses[np.asarray(all_branches[i]) - 1]
        radius_fn = _radius_generating_fn(
            radiuses=rads_in_branch, each_length=each_length[i]
        )
        # Beause SWC starts counting at 1, but numpy counts from 0.
        # ind_of_branch_endpoint = np.asarray(b) - 1
        radius_fns.append(radius_fn)
    return radius_fns


def _radius_generating_fn(radiuses, each_length):
    summed_len = np.sum(each_length)
    cutoffs = np.cumsum(np.concatenate([np.asarray([0]), each_length])) / summed_len
    cutoffs[0] -= 1e-8
    cutoffs[-1] += 1e-8

    def radius(loc):
        index = np.digitize(loc, cutoffs, right=False)
        left_rad = radiuses[index - 1]
        right_rad = radiuses[index]
        left_loc = cutoffs[index - 1]
        right_loc = cutoffs[index]
        loc_within_bin = (loc - left_loc) / (right_loc - left_loc)
        return left_rad + (right_rad - left_rad) * loc_within_bin

    return radius


def _compute_pathlengths(all_branches, coords):
    branch_pathlengths = []
    for b in all_branches:
        coords_in_branch = coords[np.asarray(b) - 1]
        point_diffs = np.diff(coords_in_branch, axis=0)
        dists = np.sqrt(
            point_diffs[:, 0] ** 2 + point_diffs[:, 1] ** 2 + point_diffs[:, 2] ** 2
        )
        branch_pathlengths.append(dists)
    return branch_pathlengths
