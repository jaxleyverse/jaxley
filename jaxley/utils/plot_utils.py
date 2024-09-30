# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
from scipy.spatial import ConvexHull

from jaxley.utils.cell_utils import v_interp


def plot_morph(
    xyzr,
    dims: Tuple[int] = (0, 1),
    col: str = "k",
    ax: Optional[Axes] = None,
    type: str = "line",
    morph_plot_kwargs: Dict = {},
) -> Axes:
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


def extract_outline(points: ndarray) -> ndarray:
    """Get the outline of a 2D/3D shape.

    Extracts the subset of points which form the convex hull, i.e. the outline of
    the input points.

    Args:
        points: An array of points / corrdinates.

    Returns:
        An array of points which form the convex hull.
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return hull_points


def compute_rotation_matrix(axis: ndarray, angle: float) -> ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by the given angle.

    Can be used to rotate a coordinate vector by multiplying it with the rotation
    matrix.

    Args:
        axis: The axis of rotation.
        angle: The angle of rotation in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def plot_cylinder_projection(
    orientation: ndarray,
    length: float,
    radius: float,
    center: ndarray,
    dims: Tuple[int],
    ax: Axes = None,
    **kwargs,
) -> Axes:
    """Plot the 2D projection of a cylinder on a cardinal plane.

    Project the projection of a cylinder that is oriented in 3D space.
    - Create cylinder mesh
    - rotate cylinder mesh to orient it lengthwise along a given orientation vector.
    - move its center
    - project onto plane
    - compute outline of projected mesh.
    - fill area inside the outline

    Args:
        orientation: orientation vector. The cylinder will be oriented along this vector.
        length: The length of the cylinder.
        radius: The radius of the cylinder.
        center: The x,y,z coordinates of the center of the cylinder.
        dims: The dimensions to project the cylinder onto, i.e. [0,1] xy-plane.
        ax: The matplotlib axis to plot on.

    Returns:
        Plot of the cylinder projection.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Normalize axis vector
    orientation = np.array(orientation)
    orientation = orientation / np.linalg.norm(orientation)

    # Create a rotation matrix to align the cylinder with the given axis
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, orientation)
    rotation_angle = np.arccos(np.dot(z_axis, orientation))

    if np.allclose(rotation_axis, 0):
        rotation_matrix = np.eye(3)
    else:
        rotation_matrix = compute_rotation_matrix(rotation_axis, rotation_angle)

    # Define cylinder
    resolution = 100
    t = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(-length / 2, length / 2, resolution)
    T, Z = np.meshgrid(t, z)

    X = radius * np.cos(T)
    Y = radius * np.sin(T)

    # Rotate cylinder
    points = np.dot(rotation_matrix, np.array([X.flatten(), Y.flatten(), Z.flatten()]))
    X = points.reshape(3, -1)

    # project onto plane and move
    X = X[dims]
    X += np.array(center)[dims, np.newaxis]

    # get outline of cylinder mesh
    X = extract_outline(X.T).T

    ax.fill(X[0].flatten(), X[1].flatten(), **kwargs)
    return ax


def plot_comps(
    view,
    dims: Tuple[int] = (0, 1),
    ax: Optional[Axes] = None,
    comp_plot_kwargs: Dict = {},
    true_comp_length: bool = True,
) -> Axes:
    """Plot compartmentalized neural mrophology.

    Plots the projection of the cylindrical compartments.

    Args:
        view: The view of the module.
        dims: The dimensions to project the cylinder onto, i.e. [0,1] xy-plane.
        ax: The matplotlib axis to plot on.
        comp_plot_kwargs: The plot kwargs for plt.fill.
        true_comp_length: If True, the length of the compartment is used, i.e. the
            length of the traced neurite. This means for zig-zagging neurites the
            cylinders will be longer than the straight-line distance between the
            start and end point of the neurite. This can lead to overlapping and
            miss-aligned cylinders. Setting this False will use the straight-line
            distance instead for nicer plots.

    Returns:
        Plot of the compartmentalized morphology.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    assert not np.any(np.isnan(view.xyzr[0][:, :3])), "missing xyz coordinates."
    if "x" not in view.pointer.nodes.columns:
        view.pointer._update_nodes_with_xyz()

    branches_inds = np.unique(view.view["branch_index"].to_numpy())
    for idx in branches_inds:
        locs = view.pointer.xyzr[idx][:, :3]
        if locs.shape[0] == 1:  # assume spherical comp
            radius = view.pointer.branch(idx).view["radius"].iloc[0]
            ax.add_artist(plt.Circle(locs[0, dims], radius))
        else:
            lens = np.sqrt(np.nansum(np.diff(locs, axis=0) ** 2, axis=1))
            lens = np.cumsum([0] + lens.tolist())
            comp_ends = v_interp(
                np.linspace(0, lens[-1], view.pointer.nseg + 1), lens, locs
            ).T
            axes = np.diff(comp_ends, axis=0)
            cylinder_lens = np.sqrt(np.sum(axes**2, axis=1))

            for l, axis, (i, comp) in zip(
                cylinder_lens, axes, view.pointer.branch(idx).view.iterrows()
            ):
                center = comp[["x", "y", "z"]]
                radius = comp["radius"]
                length = comp["length"] if true_comp_length else l
                ax = plot_cylinder_projection(
                    axis, length, radius, center, np.array(dims), ax, **comp_plot_kwargs
                )
    return ax
