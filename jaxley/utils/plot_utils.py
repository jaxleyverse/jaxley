# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy import ndarray
from scipy.spatial import ConvexHull

from jaxley.utils.cell_utils import v_interp


def plot_graph(
    xyzr: ndarray,
    dims: Tuple[int] = (0, 1),
    col: str = "k",
    ax: Optional[Axes] = None,
    type: str = "line",
    morph_plot_kwargs: Dict = {},
) -> Axes:
    """Plot morphology.

    Args:
        xyzr: The coordinates of the morphology.
        dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
            two or three of them.
        col: The color for all branches.
        ax: The matplotlib axis to plot on.
        type: Either `line` or `scatter`.
        morph_plot_kwargs: The plot kwargs for plt.plot or plt.scatter.
    """

    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")

    for coords_of_branch in xyzr:
        points = coords_of_branch[:, dims].T

        if "line" in type.lower():
            _ = ax.plot(*points, color=col, **morph_plot_kwargs)
        elif "scatter" in type.lower():
            _ = ax.scatter(*points, color=col, **morph_plot_kwargs)
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


def create_cone_frustum_mesh(
    length: float,
    radius_bottom: float,
    radius_top: float,
    bottom_dome: bool = False,
    top_dome: bool = False,
) -> ndarray:
    """Generates mesh points for a cone frustum, with optional domes at either end.

    Args:
        length: The length of the frustum.
        radius_bottom: The radius of the bottom of the frustum.
        radius_top: The radius of the top of the frustum.
        bottom_dome: If True, a dome is added to the bottom of the frustum.
            The dome is a hemisphere with radius `radius_bottom`.
        top_dome: If True, a dome is added to the top of the frustum.
            The dome is a hemisphere with radius `radius_top`.

    Returns:
        An array of mesh points.
    """

    resolution = 100
    t = np.linspace(0, 2 * np.pi, resolution)

    # Determine the total height including domes
    total_height = length
    total_height += radius_bottom if bottom_dome else 0
    total_height += radius_top if top_dome else 0

    z = np.linspace(0, total_height, resolution)
    T, Z = np.meshgrid(t, z)

    # Initialize arrays
    X = np.zeros_like(T)
    Y = np.zeros_like(T)
    R = np.zeros_like(T)

    # Bottom hemisphere
    if bottom_dome:
        dome_mask = Z < radius_bottom
        arg = 1 - Z[dome_mask] / radius_bottom
        arg[np.isclose(arg, 1, atol=1e-6, rtol=1e-6)] = 1
        arg[np.isclose(arg, -1, atol=1e-6, rtol=1e-6)] = -1
        phi = np.arccos(1 - Z[dome_mask] / radius_bottom)
        R[dome_mask] = radius_bottom * np.sin(phi)
        Z[dome_mask] = Z[dome_mask]

    # Frustum
    frustum_start = radius_bottom if bottom_dome else 0
    frustum_end = total_height - (radius_top if top_dome else 0)
    frustum_mask = (Z >= frustum_start) & (Z <= frustum_end)
    Z_frustum = Z[frustum_mask] - frustum_start
    R[frustum_mask] = radius_bottom + (radius_top - radius_bottom) * (
        Z_frustum / length
    )

    # Top hemisphere
    if top_dome:
        dome_mask = Z > (total_height - radius_top)
        arg = (Z[dome_mask] - (total_height - radius_top)) / radius_top
        arg[np.isclose(arg, 1, atol=1e-6, rtol=1e-6)] = 1
        arg[np.isclose(arg, -1, atol=1e-6, rtol=1e-6)] = -1
        phi = np.arccos(arg)
        R[dome_mask] = radius_top * np.sin(phi)

    X = R * np.cos(T)
    Y = R * np.sin(T)

    return np.stack([X, Y, Z])


def create_cylinder_mesh(length: float, radius: float) -> ndarray:
    """Generates mesh points for a cylinder.

    Args:
        length: The length of the cylinder.
        radius: The radius of the cylinder.

    Returns:
        An array of mesh points.
    """
    # Define cylinder
    resolution = 100
    t = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(-length / 2, length / 2, resolution)
    T, Z = np.meshgrid(t, z)

    X = radius * np.cos(T)
    Y = radius * np.sin(T)
    return np.stack([X, Y, Z])


def create_sphere_mesh(radius: float) -> np.ndarray:
    """Generates mesh points for a sphere.

    Args:
        radius: The radius of the sphere.

    Returns:
        An array of mesh points.
    """
    resolution = 100
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)

    # Create a 2D meshgrid for phi and theta
    PHI, THETA = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates
    X = radius * np.sin(PHI) * np.cos(THETA)
    Y = radius * np.sin(PHI) * np.sin(THETA)
    Z = radius * np.cos(PHI)

    return np.stack([X, Y, Z])


def plot_mesh(
    XYZ: ndarray,
    orientation: ndarray,
    center: ndarray,
    dims: Tuple[int],
    ax: Axes = None,
    **kwargs,
) -> Axes:
    """Plot the 2D projection of a volume mesh on a cardinal plane.

    Project the projection of a cylinder that is oriented in 3D space.
    - Create cylinder mesh
    - rotate cylinder mesh to orient it lengthwise along a given orientation vector.
    - move its center
    - project onto plane
    - compute outline of projected mesh.
    - fill area inside the outline

    Args:
        XYZ: coordinates of the xyz mesh that define the volume
        orientation: orientation vector. The cylinder will be oriented along this vector.
        center: The x,y,z coordinates of the center of the cylinder.
        dims: The dimensions to plot / to project the cylinder onto,
        i.e. [0,1] xy-plane or [0,1,2] for 3D.
        ax: The matplotlib axis to plot on.

    Returns:
        Plot of the cylinder projection.
    """
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")

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

    # Rotate mesh
    X, Y, Z = XYZ
    points = np.dot(rotation_matrix, np.array([X.flatten(), Y.flatten(), Z.flatten()]))
    X = points.reshape(3, -1)

    # project onto plane and move
    X = X[dims]
    X += np.array(center)[dims, np.newaxis]

    if len(dims) < 3:
        # get outline of cylinder mesh
        X = extract_outline(X.T).T
        ax.fill(*X.reshape(X.shape[0], -1), **kwargs)
    else:
        # plot 3d mesh
        ax.plot_surface(*X.reshape(*XYZ.shape), **kwargs)
    return ax


def plot_comps(
    module_or_view: Union["jx.Module", "jx.View"],
    view: pd.DataFrame,
    dims: Tuple[int] = (0, 1),
    col: str = "k",
    ax: Optional[Axes] = None,
    comp_plot_kwargs: Dict = {},
    true_comp_length: bool = True,
) -> Axes:
    """Plot compartmentalized neural mrophology.

    Plots the projection of the cylindrical compartments.

    Args:
        module_or_view: The module or view to plot.
        view: The view of the module.
        dims: The dimensions to plot / to project the cylinder onto,
            i.e. [0,1] xy-plane or [0,1,2] for 3D.
        col: The color for all compartments
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
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")

    module = (
        module_or_view.pointer
        if "pointer" in module_or_view.__dict__
        else module_or_view
    )
    assert not np.any(np.isnan(module.xyzr[0][:, :3])), "missing xyz coordinates."
    if "x" not in module.nodes.columns:
        module._update_nodes_with_xyz()
        view[["x", "y", "z"]] = module.nodes.loc[view.index, ["x", "y", "z"]]

    branches_inds = np.unique(view["branch_index"].to_numpy())
    for idx in branches_inds:
        locs = module.xyzr[idx][:, :3]
        if locs.shape[0] == 1:  # assume spherical comp
            radius = module.xyzr[idx][:, -1]
            center = module.xyzr[idx][0, :3]
            if len(dims) == 3:
                xyz = create_sphere_mesh(radius)
                ax = plot_mesh(
                    xyz,
                    np.zeros(3),
                    center,
                    np.array(dims),
                    ax,
                    color=col,
                    **comp_plot_kwargs,
                )
            else:
                ax.add_artist(plt.Circle(locs[0, dims], radius, color=col))
        else:
            lens = np.sqrt(np.nansum(np.diff(locs, axis=0) ** 2, axis=1))
            lens = np.cumsum([0] + lens.tolist())
            comp_ends = v_interp(
                np.linspace(0, lens[-1], module.nseg + 1), lens, locs
            ).T
            axes = np.diff(comp_ends, axis=0)
            cylinder_lens = np.sqrt(np.sum(axes**2, axis=1))

            branch_df = view[view["branch_index"] == idx]
            for l, axis, (i, comp) in zip(cylinder_lens, axes, branch_df.iterrows()):
                center = comp[["x", "y", "z"]]
                radius = comp["radius"]
                length = comp["length"] if true_comp_length else l
                xyz = create_cylinder_mesh(length, radius)
                ax = plot_mesh(
                    xyz,
                    axis,
                    center,
                    np.array(dims),
                    ax,
                    color=col,
                    **comp_plot_kwargs,
                )
    return ax


def plot_morph(
    module_or_view: Union["jx.Module", "jx.View"],
    view: pd.DataFrame,
    dims: Tuple[int] = (0, 1),
    col: str = "k",
    ax: Optional[Axes] = None,
    morph_plot_kwargs: Dict = {},
) -> Axes:
    """Plot the detailed morphology.

    Plots the traced morphology it was traced. That means at every point that was
    traced a disc of radius `r` is plotted. The outline of the discs are then
    connected to form the morphology. This means every trace segement can be
    represented by a cone frustum. To prevent breaks in the morphology, each
    segement is connected with a ball joint.

    Args:
        module_or_view: The module or view to plot.
        view: The view dataframe of the module.
        dims: The dimensions to plot / to project the cylinder onto,
            i.e. [0,1] xy-plane or [0,1,2] for 3D.
        col: The color for all branches
        ax: The matplotlib axis to plot on.
        morph_plot_kwargs: The plot kwargs for plt.fill.

    Returns:
        Plot of the detailed morphology."""
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")
    if len(dims) == 3:
        warn(
            "rendering large morphologies in 3D can take a while. Consider projecting to 2D instead."
        )

    module = (
        module_or_view.pointer
        if "pointer" in module_or_view.__dict__
        else module_or_view
    )
    assert not np.any(np.isnan(module.xyzr[0][:, :3])), "missing xyz coordinates."

    branches_inds = np.unique(view["branch_index"].to_numpy())

    for idx in branches_inds:
        xyzrs = module.xyzr[idx]
        if len(xyzrs) > 1:
            for xyzr1, xyzr2 in zip(xyzrs[1:, :], xyzrs[:-1, :]):
                dxyz = xyzr2[:3] - xyzr1[:3]
                length = np.sqrt(np.sum(dxyz**2))
                points = create_cone_frustum_mesh(
                    length, xyzr1[-1], xyzr2[-1], bottom_dome=True, top_dome=True
                )
                plot_mesh(
                    points,
                    dxyz,
                    xyzr1[:3],
                    np.array(dims),
                    color=col,
                    ax=ax,
                    **morph_plot_kwargs,
                )
        else:
            points = create_cone_frustum_mesh(
                0, xyzrs[:, -1], xyzrs[:, -1], bottom_dome=True, top_dome=True
            )
            plot_mesh(
                points,
                np.ones(3),
                xyzrs[0, :3],
                dims=np.array(dims),
                color=col,
                ax=ax,
                **morph_plot_kwargs,
            )

    return ax
