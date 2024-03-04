from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_morph(
    xyzr,
    dims=(0, 1),
    col: Union[str, List] = "k",
    ax=None,
    type: str = "line",
    morph_plot_kwargs: Dict = None,
):
    """Plot morphology.

    Args:
        dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
            two of them.
        type: Either `line` or `scatter`.
        col: The color for branches / nodes. Accepts single color, one color per node
            or one color per branch.
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 3))
    is_str = isinstance(col, str)
    is_rgb = isinstance(col, (list, np.ndarray)) and len(col) in [3, 4]
    col = [col] * len(xyzr) if is_str or is_rgb else col

    for coords_of_branch, c in zip(xyzr, col):
        x1, x2 = coords_of_branch[:, dims].T

        if "line" in type.lower():
            _ = ax.plot(x1, x2, c=c, **morph_plot_kwargs)
        elif "scatter" in type.lower():
            _ = ax.scatter(x1, x2, c=c, **morph_plot_kwargs)
        else:
            raise NotImplementedError

    return ax
