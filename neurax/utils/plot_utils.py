import matplotlib.pyplot as plt
import numpy as np
from neurax.utils.cell_utils import (
    _compute_num_children,
    _compute_index_of_child,
    compute_levels,
)


def plot_morph(cell: "nx.Cell", figsize=(4, 4)):
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
    endpoints = [[branch_lens[0], 0]]

    # Different levels will get a different "angle" at which the children emerge from
    # the parents. This angle is defined by the `y_offset_multiplier`. This value
    # defines the range between y-location of the first and of the last child of a
    # parent.
    y_offset_multiplier = np.linspace(5.0, 0.5, np.max(levels) + 1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot([0, branch_lens[0]], [0, 0], c="k")

    for b in range(1, cell.total_nbranches):
        start_point = endpoints[parents[b]]
        num_children_of_parent = num_children[parents[b]]
        y_offset = (
            ((index_of_child[b] / (num_children_of_parent - 1))) - 0.5
        ) * y_offset_multiplier[levels[b]]
        len_of_path = np.sqrt(y_offset ** 2 + 1.0)

        end_point = [
            start_point[0] + branch_lens[b] / len_of_path * 1.0,
            start_point[1] + branch_lens[b] / len_of_path * y_offset,
        ]
        endpoints.append(end_point)
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], c="k")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\mu$m")
    ax.set_ylabel(r"$\mu$m")
    plt.axis("square")

    return fig, ax

