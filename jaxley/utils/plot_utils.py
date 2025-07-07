# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional, Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from numpy import ndarray
from scipy.spatial import ConvexHull

from jaxley.utils.cell_utils import v_interp


def plot_graph(
    xyzr: ndarray,
    dims: Tuple[int] = (0, 1),
    color: str = "k",
    ax: Optional[Axes] = None,
    type: str = "line",
    **kwargs,
) -> Axes:
    """Plot morphology.

    Args:
        xyzr: The coordinates of the morphology.
        dims: Which dimensions to plot. 1=x, 2=y, 3=z coordinate. Must be a tuple of
            two or three of them.
        color: The color for all branches.
        ax: The matplotlib axis to plot on.
        type: Either `line` or `scatter`.
        kwargs: The plot kwargs for plt.plot or plt.scatter.
    """

    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")

    for coords_of_branch in xyzr:
        points = coords_of_branch[:, dims].T

        if "line" in type.lower():
            std_dev = np.std(coords_of_branch[:, :3], axis=0)
            if points.shape[1] > 1 and np.any(std_dev > 0.0):
                _ = ax.plot(*points, color=color, **kwargs)
            else:
                # Single point somata are drawn as circles with appropriate radius.
                circle = Circle(
                    (points[0, 0], points[1, 0]),
                    radius=coords_of_branch[0, -1],
                    facecolor=color,
                    **kwargs,
                )
                _ = ax.add_patch(circle)
        elif "scatter" in type.lower():
            _ = ax.scatter(*points, color=color, **kwargs)
        else:
            raise NotImplementedError

    return ax


def extract_outline(points: ndarray) -> ndarray:
    """Get the outline of a 2D/3D shape.

    Extracts the subset of points which form the convex hull, i.e. the outline of
    the input points.

    Args:
        points: An array of points / coordinates.

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
    resolution: int = 100,
) -> ndarray:
    """Generates mesh points for a cone frustum, with optional domes at either end.

    This is used to render the traced morphology in 3D (and to project it to 2D)
    as part of `plot_morph`. Sections between two traced coordinates with two
    different radii can be represented by a cone frustum. Additionally, the ends
    of the frustum can be capped with hemispheres to ensure that two neighbouring
    frustums are connected smoothly (like ball joints).

    Args:
        length: The length of the frustum.
        radius_bottom: The radius of the bottom of the frustum.
        radius_top: The radius of the top of the frustum.
        bottom_dome: If True, a dome is added to the bottom of the frustum.
            The dome is a hemisphere with radius `radius_bottom`.
        top_dome: If True, a dome is added to the top of the frustum.
            The dome is a hemisphere with radius `radius_top`.
        resolution: defines the resolution of the mesh.
            If too low (typically <10), can result in errors.
            Useful too have a simpler mesh for plotting.

    Returns:
        An array of mesh points.
    """

    t = np.linspace(0, 2 * np.pi, resolution)

    # Determine the total height including domes
    total_height = length
    total_height += radius_bottom if bottom_dome else 0
    total_height += radius_top if top_dome else 0

    z = np.linspace(0, total_height, resolution)
    t_grid, z_coords = np.meshgrid(t, z)

    # Initialize arrays
    x_coords = np.zeros_like(t_grid)
    y_coords = np.zeros_like(t_grid)
    r_coords = np.zeros_like(t_grid)

    # Bottom hemisphere
    if bottom_dome:
        dome_mask = z_coords < radius_bottom
        arg = 1 - z_coords[dome_mask] / radius_bottom
        arg[np.isclose(arg, 1, atol=1e-6, rtol=1e-6)] = 1
        arg[np.isclose(arg, -1, atol=1e-6, rtol=1e-6)] = -1
        phi = np.arccos(1 - z_coords[dome_mask] / radius_bottom)
        r_coords[dome_mask] = radius_bottom * np.sin(phi)
        z_coords[dome_mask] = z_coords[dome_mask]

    # Frustum
    frustum_start = radius_bottom if bottom_dome else 0
    frustum_end = total_height - (radius_top if top_dome else 0)
    frustum_mask = (z_coords >= frustum_start) & (z_coords <= frustum_end)
    z_frustum = z_coords[frustum_mask] - frustum_start
    r_coords[frustum_mask] = radius_bottom + (radius_top - radius_bottom) * (
        z_frustum / length
    )

    # Top hemisphere
    if top_dome:
        dome_mask = z_coords > (total_height - radius_top)
        arg = (z_coords[dome_mask] - (total_height - radius_top)) / radius_top
        arg[np.isclose(arg, 1, atol=1e-6, rtol=1e-6)] = 1
        arg[np.isclose(arg, -1, atol=1e-6, rtol=1e-6)] = -1
        phi = np.arccos(arg)
        r_coords[dome_mask] = radius_top * np.sin(phi)

    x_coords = r_coords * np.cos(t_grid)
    y_coords = r_coords * np.sin(t_grid)

    return np.stack([x_coords, y_coords, z_coords])


def create_cylinder_mesh(
    length: float, radius: float, resolution: int = 100
) -> ndarray:
    """Generates mesh points for a cylinder.

    This is used to render cylindrical compartments in 3D (and to project it to 2D)
    as part of `plot_comps`.

    Args:
        length: The length of the cylinder.
        radius: The radius of the cylinder.
        resolution: defines the resolution of the mesh.
            If too low (typically <10), can result in errors.
            Useful too have a simpler mesh for plotting.

    Returns:
        An array of mesh points.
    """
    # Define cylinder
    t = np.linspace(0, 2 * np.pi, resolution)
    z_coords = np.linspace(-length / 2, length / 2, resolution)
    t_grid, z_coords = np.meshgrid(t, z_coords)

    x_coords = radius * np.cos(t_grid)
    y_coords = radius * np.sin(t_grid)
    return np.stack([x_coords, y_coords, z_coords])


def create_sphere_mesh(radius: float, resolution: int = 100) -> np.ndarray:
    """Generates mesh points for a sphere.

    This is used to render spherical compartments in 3D (and to project it to 2D)
    as part of `plot_comps`.

    Args:
        radius: The radius of the sphere.
        resolution: defines the resolution of the mesh.
            If too low (typically <10), can result in errors.
            Useful too have a simpler mesh for plotting.

    Returns:
        An array of mesh points.
    """
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)

    # Create a 2D meshgrid for phi and theta
    phi_coords, theta_coords = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates
    x_coords = radius * np.sin(phi_coords) * np.cos(theta_coords)
    y_coords = radius * np.sin(phi_coords) * np.sin(theta_coords)
    z_coords = radius * np.cos(phi_coords)

    return np.stack([x_coords, y_coords, z_coords])


def plot_mesh(
    mesh_points: ndarray,
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
        mesh_points: coordinates of the xyz mesh that define the volume
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
    x_mesh, y_mesh, z_mesh = mesh_points
    rotated_mesh_points = np.dot(
        rotation_matrix,
        np.array([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()]),
    )
    rotated_mesh_points = rotated_mesh_points.reshape(3, -1)

    # project onto plane and move
    rotated_mesh_points = rotated_mesh_points[dims]
    rotated_mesh_points += np.array(center)[dims, np.newaxis]

    if len(dims) < 3:
        # get outline of cylinder mesh
        mesh_outline = extract_outline(rotated_mesh_points.T).T
        ax.fill(*mesh_outline.reshape(mesh_outline.shape[0], -1), **kwargs)
    else:
        # plot 3d mesh
        ax.plot_surface(*rotated_mesh_points.reshape(*mesh_points.shape), **kwargs)
    return ax


def plot_comps(
    module_or_view: Union["jx.Module", "jx.View"],
    dims: Tuple[int] = (0, 1),
    color: str = "k",
    ax: Optional[Axes] = None,
    true_comp_length: bool = True,
    resolution: int = 100,
    **kwargs,
) -> Axes:
    """Plot compartmentalized neural morphology.

    Plots the projection of the cylindrical compartments.

    Args:
        module_or_view: The module or view to plot.
        dims: The dimensions to plot / to project the cylinder onto,
            i.e. [0,1] xy-plane or [0,1,2] for 3D.
        color: The color for all compartments
        ax: The matplotlib axis to plot on.
        true_comp_length: If True, the length of the compartment is used, i.e. the
            length of the traced neurite. This means for zig-zagging neurites the
            cylinders will be longer than the straight-line distance between the
            start and end point of the neurite. This can lead to overlapping and
            miss-aligned cylinders. Setting this False will use the straight-line
            distance instead for nicer plots.
        resolution: defines the resolution of the mesh.
            If too low (typically <10), can result in errors.
            Useful too have a simpler mesh for plotting.
        kwargs: The plot kwargs for plt.fill.

    Returns:
        Plot of the compartmentalized morphology.
    """
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")

    assert not np.any(
        np.isnan(module_or_view.xyzr[0][:, :3])
    ), "missing xyz coordinates."
    if "x" not in module_or_view.nodes.columns:
        module_or_view.compute_compartment_centers()

    for idx, xyzr in zip(module_or_view._branches_in_view, module_or_view.xyzr):
        locs = xyzr[:, :3]
        if locs.shape[0] == 1:  # assume spherical comp
            radius = xyzr[:, -1]
            center = xyzr[0, :3]
            if len(dims) == 3:
                xyz = create_sphere_mesh(radius, resolution)
                ax = plot_mesh(
                    xyz,
                    np.array([0, 0, 1]),
                    center,
                    np.array(dims),
                    ax,
                    color=color,
                    **kwargs,
                )
            else:
                ax.add_artist(plt.Circle(locs[0, dims], radius, color=color))
        else:
            lens = np.sqrt(np.nansum(np.diff(locs, axis=0) ** 2, axis=1))
            lens = np.cumsum([0] + lens.tolist())
            comp_ends = v_interp(
                np.linspace(0, lens[-1], module_or_view.ncomp + 1), lens, locs
            ).T
            axes = np.diff(comp_ends, axis=0)
            cylinder_lens = np.sqrt(np.sum(axes**2, axis=1))

            branch_df = module_or_view.nodes[
                module_or_view.nodes["global_branch_index"] == idx
            ]
            for l, axis, (i, comp) in zip(cylinder_lens, axes, branch_df.iterrows()):
                center = comp[["x", "y", "z"]].astype(float)
                radius = comp["radius"]
                length = comp["length"] if true_comp_length else l
                xyz = create_cylinder_mesh(length, radius, resolution)
                ax = plot_mesh(
                    xyz,
                    axis,
                    center,
                    np.array(dims),
                    ax,
                    color=color,
                    **kwargs,
                )
    return ax


def plot_morph(
    module_or_view: Union["jx.Module", "jx.View"],
    dims: Tuple[int] = (0, 1),
    color: str = "k",
    ax: Optional[Axes] = None,
    resolution: int = 100,
    **kwargs,
) -> Axes:
    """Plot the detailed morphology.

    Plots the traced morphology it was traced. That means at every point that was
    traced a disc of radius `r` is plotted. The outline of the discs are then
    connected to form the morphology. This means every trace segment can be
    represented by a cone frustum. To prevent breaks in the morphology, each
    segment is connected with a ball joint.

    Args:
        module_or_view: The module or view to plot.
        dims: The dimensions to plot / to project the cylinder onto,
            i.e. [0,1] xy-plane or [0,1,2] for 3D.
        color: The color for all branches
        ax: The matplotlib axis to plot on.
        kwargs: The plot kwargs for plt.fill.

        resolution: defines the resolution of the mesh.
            If too low (typically <10), can result in errors.
            Useful too have a simpler mesh for plotting.

    Returns:
        Plot of the detailed morphology."""
    if ax is None:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) if len(dims) < 3 else plt.axes(projection="3d")
    if len(dims) == 3:
        warn(
            "rendering large morphologies in 3D can take a while. Consider projecting to 2D instead."
        )

    assert not np.any(
        np.isnan(module_or_view.xyzr[0][:, :3])
    ), "missing xyz coordinates."

    for xyzr in module_or_view.xyzr:
        if len(xyzr) > 1:
            for xyzr1, xyzr2 in zip(xyzr[1:, :], xyzr[:-1, :]):
                dxyz = xyzr2[:3] - xyzr1[:3]
                length = np.sqrt(np.sum(dxyz**2))
                points = create_cone_frustum_mesh(
                    length,
                    xyzr1[-1],
                    xyzr2[-1],
                    bottom_dome=True,
                    top_dome=True,
                    resolution=resolution,
                )
                plot_mesh(
                    points,
                    dxyz,
                    xyzr1[:3],
                    np.array(dims),
                    color=color,
                    ax=ax,
                    **kwargs,
                )
        else:
            points = create_cone_frustum_mesh(
                0,
                xyzr[:, -1],
                xyzr[:, -1],
                bottom_dome=True,
                top_dome=True,
                resolution=resolution,
            )
            plot_mesh(
                points,
                np.ones(3),
                xyzr[0, :3],
                dims=np.array(dims),
                color=color,
                ax=ax,
                **kwargs,
            )

    return ax
