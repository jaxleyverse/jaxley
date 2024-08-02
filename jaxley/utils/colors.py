# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import matplotlib.pyplot as plt
import numpy as np


def network_cols(num_neurons_in_layers: list):
    fn_order = [
        grey_shades,
        purple_shades,
        blue_shades,
        green_shades,
        red_shades,
        orange_shades,
    ]
    cols = []
    for num_in_layer, col_builder in zip(num_neurons_in_layers, fn_order):
        cols_in_layer = col_builder(num_in_layer)
        cols += cols_in_layer
    return cols


def blue_shades(n):
    """Return n shades of blue."""
    return plt.get_cmap("Blues")(np.linspace(1.0, 0.2, n)).tolist()


def purple_shades(n):
    """Return n shades of purple."""
    return plt.get_cmap("Purples")(np.linspace(1.0, 0.2, n)).tolist()


def red_shades(n):
    """Return n shades of red."""
    return plt.get_cmap("Reds")(np.linspace(1.0, 0.2, n)).tolist()


def green_shades(n):
    """Return n shades of green."""
    return plt.get_cmap("Greens")(np.linspace(1.0, 0.2, n)).tolist()


def grey_shades(n):
    """Return n shades of grey."""
    return plt.get_cmap("Greys")(np.linspace(1.0, 0.2, n)).tolist()


def orange_shades(n):
    """Return n shades of orange."""
    return plt.get_cmap("Oranges")(np.linspace(1.0, 0.2, n)).tolist()
