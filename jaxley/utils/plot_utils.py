from typing import Dict

import matplotlib.pyplot as plt


def plot_morph(xyzr, dims=(0, 1), col="k", ax=None, morph_plot_kwargs: Dict = None):
    """Plot morphology.

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
