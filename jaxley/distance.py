import numpy as np


def distance(comp1: "View", comp2: "View", metric="euclidean") -> np.ndarray:
    """Calculate the Euclidean distance between two compartments."""

    centers1 = comp1._compute_coords_of_comp_centers()
    centers2 = comp2._compute_coords_of_comp_centers()

    if metric == "euclidean":
        return np.linalg.norm(centers1[None, ...] - centers2[:, None, ...], axis=-1)
    elif metric == "manhattan":
        return np.sum(np.abs(centers1[None, ...] - centers2[:, None, ...]), axis=-1)
    elif metric == "pathwise":
        raise NotImplementedError("Pathwise distance not yet implemented.")
    else:
        raise ValueError("Invalid metric.")
