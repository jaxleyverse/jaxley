import matplotlib.pyplot as plt
import numpy as np

from neurax.utils.cell_utils import (_compute_index_of_child,
                                     _compute_num_children, compute_levels)
from neurax.utils.swc import _build_parents, _split_into_branches_and_sort

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


def plot_morph(
    cell: "nx.Cell",
    figsize=(4, 4),
    cols="k",
    highlight_branch_inds=[],
    max_y_multiplier: float = 5.0,
    min_y_multiplier: float = 0.5,
):
    """Plot the stick representation of a morphology.

    This method operates at the branch level. It does not plot individual compartments,
    but only individual branches. It also ignores the radius, but it takes into account
    the lengths.

    Args:
        cell: The `Cell` object to be plotted.
        figsize: Size of the figure.

    Returns:
        `fig, ax` of the plot.
    """
    parents = cell.comb_parents
    num_children = _compute_num_children(parents)
    index_of_child = _compute_index_of_child(parents)
    levels = compute_levels(parents)

    # Extract branch.
    inds_branch = cell.nodes.groupby("branch_index")["comp_index"].apply(list)
    branch_lens = [np.sum(cell.params["length"][np.asarray(i)]) for i in inds_branch]
    endpoints = []

    # Different levels will get a different "angle" at which the children emerge from
    # the parents. This angle is defined by the `y_offset_multiplier`. This value
    # defines the range between y-location of the first and of the last child of a
    # parent.
    y_offset_multiplier = np.linspace(
        max_y_multiplier, min_y_multiplier, np.max(levels) + 1
    )

    cols = [cols] * len(branch_lens)
    counter_highlight_branches = 0
    lines = []

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for b in range(cell.total_nbranches):
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

        col = cols[b]
        if b in highlight_branch_inds:
            col = highlight_cols[counter_highlight_branches % len(highlight_cols)]
            counter_highlight_branches += 1
        (line,) = ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            c=col,
            label=f"ind {b}",
        )

        if b in highlight_branch_inds:
            lines.append(line)

    ax.legend(handles=lines, loc="upper left", bbox_to_anchor=(1.05, 1, 0, 0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\mu$m")
    ax.set_ylabel(r"$\mu$m")
    plt.axis("square")

    return fig, ax


def plot_swc(
    fname,
    max_branch_len: float = 100.0,
    figsize=(4, 4),
    dims=(0, 1),
    cols="k",
    highlight_branch_inds=[],
):
    """Plot morphology given an SWC file.

    Args:
        dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
            two of them.
        cols: The color for all branches except the highlighted ones.
        highlight_branch_inds: Branch indices that will be highlighted.
    """
    content = np.loadtxt(fname)
    sorted_branches, _ = _split_into_branches_and_sort(
        content, max_branch_len=max_branch_len
    )
    parents = _build_parents(sorted_branches)
    if np.sum(np.asarray(parents) == -1) > 1.0:
        sorted_branches = [[0]] + sorted_branches

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
