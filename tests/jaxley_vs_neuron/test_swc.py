# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
import numpy as np
import pytest
from neuron import h
from scipy.spatial.distance import cdist

import jaxley as jx
from jaxley.channels import HH
from tests.helpers import (
    neuron_section_graph,
    neuron_seg_xyz,
    select_evenly_spaced_nodes,
)

_ = h.load_file("stdlib.hoc")
_ = h.load_file("import3d.hoc")


@pytest.mark.parametrize("backend", ["graph", "neuron", "legacy"])
@pytest.mark.parametrize(
    "file",
    [
        "morph_ca1_n120_single_point_soma.swc",
        "morph_ca1_n120.swc",
        "morph_l5pc_with_axon.swc",
        "morph_allen_485574832.swc",
        "morph_variable_radiuses_within_branch.swc",
    ],
)
def test_swc_voltages(file, backend):
    """Check if voltages of SWC recording match, for every SWC reader backend.

    Recording sites are picked in NEURON (spread evenly over the morphology) and
    matched to jaxley by their nearest compartment center.

    It tests whether, on average over time, the voltage is off by less than 0.3 mV
    for every recording.
    """
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, "../swc_files", file)  # n120

    i_delay = 5.0
    i_dur = 20.0
    i_amp = 1.0
    t_max = 30.0
    dt = 0.025

    if file == "morph_variable_radiuses_within_branch.swc":
        ncomp_per_branch = 1
    else:
        ncomp_per_branch = 3

    # needs to be run before NEURON import, since jaxley's NEURON backend overwrites
    # the NEURON `h` object otherwise
    cell = jx.read_swc(
        fname,
        ncomp=ncomp_per_branch,
        backend=backend,
        max_branch_len=2_000.0,
        ignore_swc_tracing_interruptions=False,
    )

    ##################### NEURON ##################
    h.secondorder = 0

    for sec in h.allsec():
        h.delete_section(sec=sec)

    nrn_cell = h.Import3d_SWC_read()
    nrn_cell.input(fname)
    i3d = h.Import3d_GUI(nrn_cell, False)
    i3d.instantiate(None)

    for sec in h.allsec():
        sec.nseg = ncomp_per_branch

    sections = list(h.allsec())
    middle = lambda sec: [sec(seg.x) for seg in sec][ncomp_per_branch // 2]

    # Sites spread over the morphology, and the xyz of each so jaxley can be matched.
    neuron_inds = select_evenly_spaced_nodes(neuron_section_graph(), 10)
    rec_xyz = np.stack([neuron_seg_xyz(middle(sections[i])) for i in neuron_inds])
    stim_xyz = neuron_seg_xyz(middle(h.soma[0]))

    ####################### jaxley ##################
    cell.insert(HH())
    # Match the NEURON sites to the nearest jaxley compartment.
    comp_xyz = cell.nodes[["x", "y", "z"]].to_numpy()
    jaxley_inds = cell.nodes.index[cdist(rec_xyz, comp_xyz).argmin(axis=1)].to_numpy()
    stim_ind = cell.nodes.index[cdist([stim_xyz], comp_xyz).argmin(axis=1)][0]

    cell.set("axial_resistivity", 100.0)
    cell.set("v", -62.0)
    cell.set("HH_m", 0.074901)
    cell.set("HH_h", 0.4889)
    cell.set("HH_n", 0.3644787)
    cell.scope("global").comp(stim_ind).stimulate(
        jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    )
    for i in jaxley_inds:
        cell.scope("global").comp(i).record(verbose=False)

    voltages_jaxley = jx.integrate(cell, delta_t=dt, voltage_solver="jaxley.dhs.cpu")

    ################### NEURON #################
    stim = h.IClamp(middle(h.soma[0]))
    stim.delay = i_delay
    stim.dur = i_dur
    stim.amp = i_amp

    voltage_recs = {}
    for counter, r in enumerate(neuron_inds):
        v = h.Vector()
        v.record(middle(sections[r])._ref_v)
        voltage_recs[f"v{counter}"] = v

    for sec in h.allsec():
        sec.insert("hh")
        sec.Ra = 100.0

        sec.gnabar_hh = 0.120  # S/cm2
        sec.gkbar_hh = 0.036  # S/cm2
        sec.gl_hh = 0.0003  # S/cm2
        sec.ena = 50  # mV
        sec.ek = -77.0  # mV
        sec.el_hh = -54.3  # mV

    h.dt = dt
    tstop = t_max
    v_init = -62.0

    def initialize():
        h.finitialize(v_init)
        h.fcurrent()

    def integrate():
        while h.t < tstop:
            h.fadvance()

    initialize()
    integrate()
    voltages_neuron = np.asarray([voltage_recs[key] for key in voltage_recs])
    errors = np.mean(np.abs(voltages_jaxley - voltages_neuron), axis=1)

    print(f"Errors: {errors}")

    ###################### check ################
    assert all(
        errors < 0.3
    ), f"{backend}: error {np.max(errors)} > 0.3. Voltages do not match."
