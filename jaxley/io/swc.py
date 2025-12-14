# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional

import jaxley.io as io
from jaxley.io.neuron import assert_NEURON
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
        backend: The backend to use. Currently we support `graph` and `neuron` backends.
            The `graph` backend uses the `NetworkX` module to read the SWC file and
            construct the compartment graph. The `neuron` backend uses `NEURON`'s
            `h.Import3d_SWC_read()` to do this.
        ignore_swc_tracing_interruptions: Whether to ignore discontinuities in the swc
            tracing order. If False, this will result in split branches at these points.
        relevant_type_ids: All type ids that are not in this list will be ignored for
            tracing the morphology. This means that branches which have multiple type
            ids (which are not in `relevant_type_ids`) will be considered as one branch.
            If `None`, we default to `[1, 2, 3, 4]`.

    Returns:
        A `Cell` object."""

    if backend.lower() == "graph":
        swc_df = io.tmp.read_swc(fname)
        swc_graph = io.tmp.swc_to_nx(swc_df, relevant_ids=relevant_type_ids)
        comp_graph = io.tmp.build_compartment_graph(
            swc_graph,
            ncomp=ncomp,
            min_radius=min_radius,
            max_len=max_branch_len,
            ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        )
        module = io.tmp.from_graph(
            comp_graph,
            assign_groups=assign_groups,
        )
    elif backend.lower() == "legacy":
        swc_graph = io.graph.to_swc_graph(fname)
        comp_graph = io.graph.build_compartment_graph(
            swc_graph,
            ncomp=ncomp,
            min_radius=min_radius,
            max_len=max_branch_len,
            ignore_swc_tracing_interruptions=ignore_swc_tracing_interruptions,
        )
        module = io.graph.from_graph(
            comp_graph,
            assign_groups=assign_groups,
        )
    elif backend.lower() == "neuron":
        # Check if NEURON is available
        assert_NEURON()

        io.neuron._load_swc_into_neuron(fname)
        swc_graph = io.neuron.h_allsec_to_nx(relevant_ids=relevant_type_ids)
        comp_graph = io.neuron.build_compartment_graph(ncomp=ncomp)
        module = io.tmp.from_graph(
            comp_graph,
            assign_groups=assign_groups,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use either `neuron` or `graph`.")

    return module
