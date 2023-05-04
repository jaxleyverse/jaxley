from warnings import warn
import numpy as np
import matplotlib.pyplot as plt


def read_swc(fname: str, max_branch_len: float = 100.0):
    """Read an SWC file and bring morphology into `neurax` compatible formats.

    Args:
        fname: Path to swc file.
        max_branch_len: Maximal length of one branch. If a branch exceeds this length,
            it is split into equal parts such that each subbranch is below
            `max_branch_len`.
    """
    content = np.loadtxt(fname)
    sorted_branches = _split_into_branches_and_sort(
        content, max_branch_len=max_branch_len
    )

    parents = _build_parents(sorted_branches)
    pathlengths = _compute_pathlengths(sorted_branches, content[:, 2:5])
    for i in range(len(pathlengths)):
        if pathlengths[i] == 0.0:
            warn("Found a segment with length 0. Clipping it to 1.0")
            pathlengths[i] = 1.0
    endpoint_radiuses = _extract_endpoint_radiuses(sorted_branches, content[:, 5])
    start_radius = content[0, 5]
    return parents, pathlengths, endpoint_radiuses, start_radius


def plot_swc(
    fname,
    max_branch_len: float = 100.0,
    figsize=(6, 6),
    dims=(0, 1),
    cols=None,
    highlight_branch_inds=[],
):
    """Plot morphology given an SWC file."""
    highlight_cols = [
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
        "#ff7f00",
        "#6a3d9a",
        "#b15928",
        "#a6cee3",
        "#b2df8a",
        "#fb9a99",
        "#fdbf6f",
        "#cab2d6",
        "#ffff99",
    ]
    content = np.loadtxt(fname)
    sorted_branches = _split_into_branches_and_sort(
        content, max_branch_len=max_branch_len
    )

    cols = [cols] * len(sorted_branches)

    counter_highlight_branches = 0
    lines = []

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, branch in enumerate(sorted_branches):
        coords_of_branch = content[np.asarray(branch) - 1, 2:5]
        coords_of_branch = coords_of_branch[:, dims]

        col = cols[i]
        if i in highlight_branch_inds:
            col = highlight_cols[counter_highlight_branches % len(highlight_cols)]
            counter_highlight_branches += 1

        (line,) = ax.plot(
            coords_of_branch[:, 0], coords_of_branch[:, 1], c=col, label=f"ind {i}"
        )
        if i in highlight_branch_inds:
            lines.append(line)

    ax.legend(handles=lines, loc="upper left", bbox_to_anchor=(1.05, 1, 0, 0))

    return fig, ax


def _split_into_branches_and_sort(content, max_branch_len):
    branches = _split_into_branches(content)
    branches = _remove_single_branch_artifacts(branches)
    branches = _split_long_branches(branches, content, max_branch_len)

    first_val = np.asarray([b[0] for b in branches])
    sorting = np.argsort(first_val, kind="mergesort")
    sorted_branches = [branches[s] for s in sorting]
    return sorted_branches


def _split_long_branches(branches, content, max_branch_len):
    pathlengths = _compute_pathlengths(branches, content[:, 2:5])
    split_branches = []
    for branch, length in zip(branches, pathlengths):
        num_subbranches = 1
        split_branch = [branch]
        while length > max_branch_len:
            num_subbranches += 1
            split_branch = _split_branch_equally(branch, num_subbranches)
            lengths_of_subbranches = _compute_pathlengths(
                split_branch, coords=content[:, 2:5]
            )
            length = max(lengths_of_subbranches)
            if num_subbranches > 10:
                warn(
                    """`num_subbranches > 10`, stopping to split. Most likely your
                     SWC reconstruction is not dense and some neighbouring traced
                     points are farther than `max_branch_len` apart."""
                )
                break
        split_branches += split_branch

    return split_branches


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
    n_branches = 0
    branch_inds = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent != prev_ind:
            branch_inds.append(int(current_parent))
            n_branches += 1
        prev_ind = current_ind

    all_branches = []
    current_branch = []
    for c in content:
        current_ind = c[0]
        current_parent = c[-1]
        if current_parent in branch_inds[1:]:
            all_branches.append(current_branch)
            current_branch = [int(current_parent), int(current_ind)]
        else:
            current_branch.append(int(current_ind))
    all_branches.append(current_branch)

    return all_branches


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


def _extract_endpoint_radiuses(all_branches, radiuses):
    endpoint_radiuses = []
    for b in all_branches:
        branch_endpoint = b[-1]
        # Beause SWC starts counting at 1, but numpy counts from 0.
        ind_of_branch_endpoint = branch_endpoint - 1
        endpoint_radiuses.append(radiuses[ind_of_branch_endpoint])
    return endpoint_radiuses


def _compute_pathlengths(all_branches, coords):
    branch_pathlengths = []
    for b in all_branches:
        coords_in_branch = coords[np.asarray(b) - 1]
        point_diffs = np.diff(coords_in_branch, axis=0)
        dists = np.sqrt(
            point_diffs[:, 0] ** 2 + point_diffs[:, 1] ** 2 + point_diffs[:, 2] ** 2
        )
        branch_pathlengths.append(np.sum(dists))
    return branch_pathlengths
