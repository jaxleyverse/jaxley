import matplotlib.pyplot as plt
import numpy as np


def _compute_num_kids(parents):
    num_branches = len(parents)
    num_kids = []
    for b in range(num_branches):
        n = np.sum(np.asarray(parents) == b)
        num_kids.append(n)
    return num_kids


def _compute_index_of_kid(parents):
    num_branches = len(parents)
    current_num_kids_for_each_branch = np.zeros((num_branches,), np.dtype("int"))
    index_of_kid = [None]
    for b in range(1, num_branches):
        index_of_kid.append(current_num_kids_for_each_branch[parents[b]])
        current_num_kids_for_each_branch[parents[b]] += 1
    return index_of_kid


def plot_morph(parents):
    num_kids = _compute_num_kids(parents)
    index_of_kid = _compute_index_of_kid(parents)
    num_branches = len(parents)
    x_fig = 6.0
    y_fig = 4.0

    fig, ax = plt.subplots(1, 1, figsize=(x_fig, y_fig))

    ax.plot([0, 1], [0, 0], c="k")
    endpoints = [[1, 0]]

    for b in range(1, num_branches):
        start_point = endpoints[parents[b]]
        num_kids_of_parent = num_kids[parents[b]]
        y_offset = (((index_of_kid[b] / (num_kids_of_parent - 1))) - 0.5) * 2.0
        len_of_path = np.sqrt(y_offset**2 + 1.0)
        end_point = [
            start_point[0] + 1.0 / len_of_path,
            start_point[1] + y_offset / len_of_path,
        ]
        endpoints.append(end_point)
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], c="k")

    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim([0, 5])
    len_y = 5.0 * y_fig / x_fig
    ax.set_ylim([-len_y / 2, len_y / 2])
    plt.show()
