from typing import Dict

import matplotlib.pyplot as plt


def plot_morph(
    xyzr,
    dims=(0, 1),
    col="k",
    ax=None,
    type: str = "line",
    morph_plot_kwargs: Dict = None,
):
    """Plot morphology.

    Args:
        dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
            two of them.
        type: Either `line` or `scatter`.
        col: The color for all branches.
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 3))
    for coords_of_branch in xyzr:
        coords_to_plot = coords_of_branch[:, dims]
        x_coords = coords_to_plot[:, 0]
        y_coords = coords_to_plot[:, 1]

        if "line" in type.lower():
            _ = ax.plot(
                coords_to_plot[:, 0], coords_to_plot[:, 1], c=col, **morph_plot_kwargs
            )
        elif "scatter" in type.lower():
            _ = ax.scatter(
                coords_to_plot[:, 0], coords_to_plot[:, 1], c=col, **morph_plot_kwargs
            )
        else:
            raise NotImplementedError

    return ax
