from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from jaxley.utils.cell_utils import (
    _compute_index_of_child,
    _compute_num_children,
    compute_levels,
)


def plot_morph(
    cell: "jx.Cell",
    col="k",
    max_y_multiplier: float = 5.0,
    min_y_multiplier: float = 0.5,
    ax=None,
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

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

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

        _ = ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            c=col,
            label=f"ind {b}",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\mu$m")
    ax.set_ylabel(r"$\mu$m")
    plt.axis("square")

    return fig, ax


def plot_swc(xyzr, dims=(0, 1), col="k", ax=None, morph_plot_kwargs: Dict = None):
    """Plot morphology given an SWC file.

    Args:
        dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
            two of them.
        cols: The color for all branches except the highlighted ones.
        highlight_branch_inds: Branch indices that will be highlighted.
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 3))
    for coords_of_branch in xyzr:
        coords_to_plot = coords_of_branch[:, dims]

        _ = ax.plot(
            coords_to_plot[:, 0], coords_to_plot[:, 1], c=col, **morph_plot_kwargs
        )

    return ax
