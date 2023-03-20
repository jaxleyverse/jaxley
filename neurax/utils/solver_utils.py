import numpy as np
import jax.numpy as jnp


def compute_levels(parents):
    levels = np.zeros_like(parents)

    for i, p in enumerate(parents):
        if p == -1:
            levels[i] = 0
        else:
            levels[i] = levels[p] + 1
    return levels


def compute_branches_in_level(levels):
    num_branches = len(levels)
    branches_in_each_level = []
    for l in range(np.max(levels) + 1):
        branches_in_current_level = []
        for b in range(num_branches):
            if levels[b] == l:
                branches_in_current_level.append(b)
        branches_in_current_level = jnp.asarray(branches_in_current_level)
        branches_in_each_level.append(branches_in_current_level)
    return branches_in_each_level
