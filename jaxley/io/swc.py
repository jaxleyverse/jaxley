# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional

from jaxley.io.graph import build_compartment_graph, from_graph, to_swc_graph
from jaxley.modules import Cell


def read_swc(
    fname: str,
    ncomp: Optional[int],
    max_branch_len: Optional[float] = None,
    min_radius: Optional[float] = None,
    assign_groups: bool = True,
    backend: str = "graph",
    ignore_swc_tracing_interruptions: bool = True,
    relevant_type_ids: Optional[List[int]] = None,
) -> Cell:
    """Reads SWC file into a `Cell`.

    Jaxley assumes cylindrical compartments and therefore defines length and radius
    for every compartment. The surface area is then 2*pi*r*length. For branches
    consisting of a single traced point we assume for them to have area 4*pi*r*r.
    Therefore, in these cases, we set length=2*r.

    Args:
        fname: Path to the swc file.
        ncomp: The number of compartments per branch.
        max_branch_len: If a branch is longer than this value it is split into two
            branches.
        min_radius: If the radius of a reconstruction is below this value it is clipped.
        assign_groups: If True, then the identity of reconstructed points in the SWC
            file will be used to generate groups `soma`, `axon`, `basal`, `apical`. See
            here:
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        backend: The backend to use. Currently only `graph` is supported.
        ignore_swc_tracing_interruptions: Whether to ignore discontinuities in the swc
            tracing order. If False, this will result in split branches at these points.
        relevant_type_ids: All type ids that are not in this list will be ignored for
            tracing the morphology. This means that branches which have multiple type
            ids (which are not in `relevant_type_ids`) will be considered as one branch.
            If `None`, we default to `[1, 2, 3, 4]`.

    Returns:
        A `Cell` object."""

    if backend == "graph":
        swc_graph = to_swc_graph(fname)
        comp_graph = build_compartment_graph(
            swc_graph,
            ncomp=ncomp,
            root=None,
            min_radius=min_radius,
            max_len=max_branch_len,
            ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
            relevant_type_ids=relevant_type_ids,
        )
        module = from_graph(
            comp_graph,
            assign_groups=assign_groups,
            solve_root=None,
            traverse_for_solve_order=True,  # Traverse to fix potential tracing errors.
        )
        return module
    else:
        raise ValueError(f"Unknown backend: {backend}. Use either `custom` or `graph`.")
