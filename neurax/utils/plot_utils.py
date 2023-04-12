import matplotlib.pyplot as plt
import numpy as np


def plot_morph(parents, num_kids, index_of_kid, levels):
    angles = [1.0, 0.6, 0.3, 0.2]
    num_branches = len(parents)
    x_fig = 2.0
    y_fig = 2.0

    fig, ax = plt.subplots(1, 1, figsize=(x_fig, y_fig))

    ax.plot([0, 1], [0, 0], c="k", linewidth=3.0)
    endpoints = [[1, 0]]

    for b in range(1, num_branches):
        start_point = endpoints[parents[b]]
        num_kids_of_parent = num_kids[parents[b]]
        y_offset = (((index_of_kid[b] / (num_kids_of_parent - 1))) - 0.5) * angles[
            levels[b]
        ]  # 1.0
        len_of_path = np.sqrt(y_offset**2 + 1.0)
        end_point = [
            start_point[0] + 1.0 / len_of_path,
            start_point[1] + y_offset / len_of_path,
        ]
        endpoints.append(end_point)
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            c="k",
            linewidth=3.0,
        )

    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.set_yticks([])
    x_plot_size = np.max(levels) + 1
    ax.set_xlim([0, x_plot_size])
    len_y = x_plot_size * y_fig / x_fig
    ax.set_ylim([-len_y / 5, len_y / 5])
