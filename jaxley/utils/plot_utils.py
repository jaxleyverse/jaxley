from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_morph(
    xyzr,
    dims: Tuple[int] = (0, 1),
    col: str = "k",
    ax: Optional[Axes] = None,
    type: str = "line",
    morph_plot_kwargs: Dict = {},
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
        x1, x2 = coords_of_branch[:, dims].T

        if "line" in type.lower():
            _ = ax.plot(x1, x2, color=col, **morph_plot_kwargs)
        elif "scatter" in type.lower():
            _ = ax.scatter(x1, x2, color=col, **morph_plot_kwargs)
        else:
            raise NotImplementedError

    return ax
