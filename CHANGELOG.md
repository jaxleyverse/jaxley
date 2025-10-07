# 0.11.5 (pre-release)

### 🐛 Bug fixes

- safe softplus, use linear function above certain threshold. This avoids an unwanted
clipping  operation due to the save_exp (#714 @matthijspals)

### 📚 Documentation

- typo fixes for several tutorial notebooks (#721, @michaeldeistler, thanks @martricks
for reporting)


# 0.11.4

### 🐛 Bug fixes

- bugfix for indexing when `init_states()` is run on a `jx.Network` (#711,
@michaeldeistler)

### 📚 Documentation

- add an example on fitting a morphologically detailed cell with gradient descent (#705,
@michaeldeistler)


# 0.11.3

### 🛠️ Internal updates

-  follow jax typing practices with Array and ArrayLike (#693, @alexpejovic)

### 🐛 Bug fixes

- fix for networks that mix point neurons and morphologically detailed neurons (#702,
@michaeldeistler)
- carry over groups from constituents of a module (e.g., `jx.Cell` groups get carried
over to `jx.Network`) (#703, @michaeldeistler)


# 0.11.2

### 🐛 Bug fixes

- Bugfix for `Network`s on `GPU`: since `v0.9.0`, networks had been very slow on GPU
because the voltage equations of cells had been processed in sequence, not in parallel.
This is now solved, giving a large speed-up for networks consisting of many cells (#691,
@michaeldeistler, thanks to @VENOM314 for reporting)

### 📚 Documentation

- Remove all content from the old mkdocs documentation website (#689, @michaeldeistler)


# 0.11.1

### 🐛 Bug fixes

- bugfix for `set_ncomp()` when the cell consists of a single branch (#686,
@michaeldeistler)

### 🛠️ Internal updates

- fix all typos in the codebase by using the `typos` project (#682, @alexpejovic)


# 0.11.0

### 🧩 New features

- simple conductance synapse added (#659, @kyralianaka)

### 📚 Documentation

- add a how-to guide on converting `NMODL` files to `Jaxley`, see
[here](https://jaxley.readthedocs.io/en/latest/how_to_guide/import_channels_from_neuron.html)
(#669, @michaeldeistler, special thanks to @r-makarov for building the tool)

### 🛠️ Internal updates

- changes to how the membrane area from SWC files is computed when the radius within a
compartment is not constant. This fix can have an impact on simulation results. The
updated computation of membrane area matches that of the NEURON simulator (#662,
@michaeldeistler, thanks to @VENOM314 for reporting).
- remove the `custom` SWC reader (which had been deprecated in `v0.9.0`, #662,
@michaeldeistler).
- fix bug in the `.ncomp` attribute after `set_ncomp()` had been run (#676,
@manuelgloeckler)


# 0.10.0

### 🧩 New features

- functionality to compute the pathwise distance between compartments (#648,
@michaeldeistler):
```python
from jaxley.morphology import distance_pathwise
path_dists = distance_pathwise(cell.soma.branch(0).comp(0), cell)
cell.nodes["path_dist_from_soma"] = path_dists
```

### 🐛 Bug fixes

- fixed synapse recording indices to be within type (#643, @kyralianaka)
- Fix inheriting from a Module #590 (#642, @jnsbck)

### 🛠️ Internal updates

- `module.distance()` is now deprecated in favor of `jx.morphology.distance_direct()`
(#648, @michaeldeistler)


# 0.9.0

### ✨ Highlights

- This PR implements a new solver, which is now used by default (#625,
@michaeldeistler). The new solver has the following advantages:
  - Much lower runtime. Across several morphologies, we get a 20% runtime speedup on
    CPU and a 50% runtime speedup on GPU.
  - Almost zero compile time. We get a 50x compile time speedup on CPU and a 3x compile
    time speedup on GPU.
- Utility to delete parts of a morphology (#612, @michaeldeistler):
```python
from jaxley.morphology import morph_delete
cell = morph_delete(cell.branch([1, 2]))
```
- Utility to connect two cells into a single cell (#612, @michaeldeistler):
```python
from jaxley.morphology import morph_connect
cell = morph_connect(cell1.branch(1).loc(0.0), cell2.branch(2).loc(1.0))
```

### 🧩 New features

- the default SWC reader has changed. To use the previous SWC reader, run
`jx.read_swc(..., backend="custom")`. However, note that we will remove this reader
in the future. If the new SWC reader is causing issues for you, please open an issue
(#612, @michaeldeistler)
- radiuses are now integrated across SWC coordinates, not interpolated
(#612, @michaeldeistler)
- remove pin of `JAX` version. New `JAX` versions (`JAX>=0.6.0`) resolve slow CPU
runtime, see [here](https://github.com/jax-ml/jax/issues/26145) (#623, @michaeldeistler)
- running the `d_lambda` rule is now much faster, see
[the how-to guide](https://jaxley.readthedocs.io/en/latest/how_to_guide/set_ncomp.html)
(#625, @michaeldeistler)

### 📚 Documentation

- Introduce the `how-to guide` on the website (#612, @michaeldeistler)
- reorganize the advanced tutorials into subgroups (#612, @michaeldeistler)
- split the morphology handling tutorials into two notebooks (#612, @michaeldeistler)

### 🛠️ Internal updates

- improvements to graph-backend for more flexibility in modifying morphologies (#613,
@michaeldeistler)
- `jx.read_swc()` now ignores `type_id > 4` by default. This ensures compatibility
with flywire, which highjacks `type_id > 4` to indicate synaptic contacts (#612,
@michaeldeistler)
- remove root compartment for SWC files (#613, @michaeldeistler)
- enable traversing compartmentalized graph for optimizing solve order (#613,
@michaeldeistler)
- `._comp_edges` are being tracked in the `View` (#621, @michaeldeistler)
- introduce `._branchpoints_` attribute and track in the `View` (#612, @michaeldeistler)
- `pre_global_comp_index` and `post_global_comp_index` are no longer part of `.edges`
by default. To get them, do `net.copy_node_property_to_edges("global_comp_index")`
(#625, @michaeldeistler)

### 🐛 Bug fixes

- `ChainTransform` forward now working as mirror to inverse (#628, @kyralianaka)
- allow `data_set` with vectors of values (#606, @chaseking)

### 🎉 New Contributors

- @chaseking made their first contribution in #606


# 0.8.2

- enable intersections of groups, e.g. `net.exc.soma` (#608, @michaeldeistler)
- allow ion diffusion to be jitted (#609, @michaeldeistler)


# 0.8.1

- fixups to tutorial notebooks (#604, @michaeldeistler)
- remove `delete_channel()` (#604, @michaeldeistler)


# 0.8.0

### 🧩 New features

- add leaky integrate-and-fire neurons (#564, @jnsbck), Izhikevich neurons, and
rate-based neurons (#601, @michaeldeistler)

### 🛠️ Minor updates

- make `delta` and `v_th` in `IonotropicSynapse` trainable parameters (#599, @jnsbck)
- make random postsnaptic compartment selection optional in connectivity functions
(#489, @kyralianaka)

### 🐛 Bug fixes

- Fix bug for `groups` when `.set_ncomp` was run (#587, @michaeldeistler)
- allow `.distance` to be jitted (#603, @michaeldeistler)


# 0.7.0

### 🧩 New Features

- Allow ion diffusion with `cell.diffuse()` and add tutorials (#438, @michaeldeistler):
```python
from jaxley.channels import CaNernstReversal
from jaxley.pumps import CaFaradayConcentrationChange

cell.insert(CaFaradayConcentrationChange())
cell.insert(CaNernstReversal())
cell.diffuse("CaCon_i")
cell.set("axial_diffusion_CaCon_i", 1.0)
```
- Introduce ion pumps (#438, @michaeldeistler)

### 🛠️ Minor changes

- rename `delete_channel()` to `delete()` (#438, @michaeldeistler)

### 🐛 Bug fixes

- Fix for simulation of morphologies with inhomogeneous numbers of compartments (#438,
@michaeldeistler)
- Bugfix for types assigned by SWC reader when soma is traced by a single point (#582,
@Kartik-Sama, @michaeldeistler).

### Code health

- add new release workflow (#588, @jnsbck)
- update FAQ and tutorials (#593, @michaeldeistler)
- make random post compartment selection optional in connectivity functions (#489,
@kyralianaka)

### 🎉 New Contributors

- @Kartik-Sama made their first contribution in #582


# 0.6.2

- also remove `nseg` from tutorials (#580, @jnsbck)


# 0.6.1

- Fixup for failing readthedocs build of v0.6.0 (#579, @michaeldeistler)
- Deprecate `nseg` in favor of `ncomp`.


# 0.6.0

### Pin of JAX version

Installing `Jaxley` will no longer install the newest version of `JAX`. We realized
that, on CPU, with version `jax==0.4.32` or newer, simulation time in `Jaxley` is 10x
slower and gradient time is 5x slower as compared to older versions of JAX. Newer
versions of `JAX` can be made equally fast as older versions by setting
`os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'` at the beginning of
your jupyter notebook (#570, @michaeldeistler).

### 🧩 New Features

- Add ability to record synaptic currents (#523, @ntolley). Recordings can be turned on
with
```python
net.record("i_IonotropicSynapse")
```

- refactor plotting (#539, @jnsbck).
  - rm networkx dependency
  - add `Network.arrange_in_layers`
  - disentangle moving of cells and plotting in `Network.vis`. To get the same as
  `net.vis(layers=[3,3])`, one now has to do:
```python
net.arrange_in_layers([3,3])
net.vis()
```

- Allow parameter sharing for groups of different sizes, i.e. due to inhomogeneous
numbers of compartments or for synapses with the same (pre-)synaptic parameters but
different numbers of post-synaptic partners. (#514, @jnsbck)

- Add `jaxley.io.graph` for exporting and importing of jaxley modules to and from
`networkx` graph objects (#355, @jnsbck).
  - Adds a new (and improved) SWC reader, which is more flexible and should also be
  easier to extend in the future.
  ```python
  from jaxley.io.graph import swc_to_graph, from_graph
  graph = swc_to_graph(fname)
  # do something to the swc graph, i.e. prune it
  pruned_graph = do_something_to_graph(graph)
  cell = from_graph(pruned_graph, ncomp=4)
  ```
  - Adds a new `to_graph` method for jaxley modules, which exports a module to a
  `networkX` graph. This allows to seamlessly work with `networkX`'s graph manipulation
  or visualization functions.
  - `"graph"` can now also be selected as a backend in the `read_swc`.
  - See the improved SWC reader
  [tutorial](https://jaxley.readthedocs.io/en/latest/tutorials/08_importing_morphologies.html)
  for more details.

### 🛠️ Code Health

- changelog added to CI (#537, #558, @jnsbck)

- Add regression tests and supporting workflows for maintaining baselines (#475,
#546, @jnsbck).
  - Regression tests can be triggered by commenting "/test_regression" on a PR.
  - Regression tests can be done locally by running
  `NEW_BASELINE=1 pytest -m regression` i.e. on `main` and then `pytest -m regression`
  on `feature`, which will produce a test report (printed to the console and saved
  to .txt).

- Allow inspecting the version via `import jaxley as jx; print(jx.__version__)` (#577,
@michaeldeistler).

### 🐛 Bug fixes

- Fixed inconsistency with *type* assertions arising due to `numpy` functions returning
different `dtypes` on platforms like Windows (#567, @Kartik-Sama)

### 🎉 New Contributors

- @ntolley made their first contribution in #523


# 0.5.0

### 🛠️ API changes

- Synapse views no longer exist (#447, #453, @jnsbck). Previous code such as
```python
net.IonotropicSynapse.set("IonotropicSynapse_s", 0.2)
```
must be updated to:
```python
net.set("IonotropicSynapse_s", 0.2)
```
For a more detailed tutorial on how to index synapses, see
[this new tutorial](https://jaxley.readthedocs.io/en/latest/tutorials/09_advanced_indexing.html).  
- Throughout the codebase, we renamed any occurrence of `seg` (for `segment`) to `comp`
(for `compartment`). The most notable user-facing changes are:
  - `branch = jx.Branch(comp, ncomp=4)`
  - `cell = jx.read_swc(fname, ncomp=4)`
- New defaults for the SWC reader with `jx.read_swc()`. By default, we now have
`assign_groups=True` (previously `False`) and `max_branch_len=None` (previously
`300.0`).
- We renamed `.view` to `.nodes`, e.g., `cell.branch(0).nodes` (#447, #453, @jnsbck).
- We renamed `_update_nodes_with_xyz()` to `compute_compartment_centers()` (#520,
@jnsbck)
- We updated the way in which transformations are built (#455, @manuelgloeckler).
Previous code such as
```python
tf = jx.ParamTransform(
    lower={"radius": 0.1, "length": 2.0},
    lower={"radius": 3.0, "length": 20.0},
)
```
must be updated to:
```python
from jaxley.optimize.transforms import ParamTransform, SigmoidTransform
transforms = [
    {"radius": SigmoidTransform(lower=0.1, upper=3.0)},
    {"length": SigmoidTransform(lower=2.0, upper=20.0)},
]
tf = jt.ParamTransform(transforms)
```

### 🧩 New features

- Added a new `delete_channel()` method (#521, @jnsbck)
- Allow to write trainables to the module (#470, @michaeldeistler):
```python
net.make_trainable("radius")
params = net.get_parameters()
net.write_trainables(params)
```
- Expose the step function to allow for fine-grained simulation (#466, @manuelgloeckler)
- More flexible and thorough viewing (#447, #453, @jnsbck)
- Boolean indexing for cells, branches, and comps (@494, @jnsbck):
```python
r_greater_1 = net.nodes.groupby("global_cell_index")["radius"].mean() > 1
net[r_greater_1].nodes.vis()
```
- check if recordings are empty (#460, @deezer257)
- enable `clamp` to be jitted and vmapped with `data_clamp()` (#374, @kyralianaka)

### 🐛 Bug fixes

- allow for cells that were read from swc to be pickled (#525, @jnsbck)
- fix units of `compute_current()` in channels (#461, @michaeldeistler)
- fix issues with plotting when the morphology has a different number of compartments
(#513, @jnsbck)

### 📚 Documentation

- new tutorial on synapse indexing (#464, @michaeldeistler, @zinaStef)
- new tutorial on parameter sharing (#464, @michaeldeistler, @zinaStef)
- new tutorial on modules and views (#493, @jnsbck)
- improved tutorial on building channel models (#473, @simoneeb)
- get rid of tensorflow dependency by defining our simple dataloader in the tutorial
(#484, @jnsbck)
- new FAQ about rate-based networks (#531, @michaeldeistler)

### Code health

- refactor tests with fixtures (@479, #499, @fabioseel, @jnsbck)
- make several attributes private (#495, @ntolley)
- move `read_swc.py` to new `io` folder (#524, @jnsbck)
- faster testing for SWC and plotting (#479, @fabioseel)
- automated tests to check if tutorials can be run (#480, @jnsbck)
- add helpers to deprecate functions and kwargs (#516, @jnsbck)

### 🎉 New Contributors

- @simoneeb made their first contribution in #473
- @zinaStef made their first contribution in #464
- @fabioseel made their first contribution in #479
- @deezer257 made their first contribution in #460


# 0.4.0

### 🧩 New features

- Changing the number of compartments: `cell.branch(0).set_ncomp(4)` (#436, #440, #445,
@michaeldeistler, @jnsbck)
- New options for plotting: `cell.vis(type='comp')` and `cell.vis(type='morph')` (#432,
#437, @jnsbck)
- Speed optimization for `jx.integrate(..., voltage_solver="jaxley.stone")` (#442,
@michaeldeistler)

### 📚 Documentation

- new website powered by sphinx:
[`jaxley.readthedocs.io`](https://jaxley.readthedocs.io/) (#434, #435, @michaeldeistler)


# v0.3.0

### 🧩 New features

- New solver: `jx.integrate(..., voltage_solver="jax.sparse")` which has very low
compile time (#418, @michaeldeistler)
- Support for different number of compartments per branch at initialization (modifying
the number of compartments after initialization is not yet supported, #418, #426, 
@michaeldeistler)

### 🐛 Bug fixes

- Bugfix for capacitances and their interplay with axial conductances (Thanks @Tunenip,
#426, @michaeldeistler)
- Bugfixes for tutorials on website


# v0.2.1

- Bugfix for using `states` in `init_state` of the channel (#421, @michaeldeistler)
- Bugfix for tutorial on building custom channels (#421, @michaeldeistler)
- Add `jaxley-mech` as `dev` dependency to use it in testing (#421, @michaeldeistler)


# v0.2.0

### 🧩 New features

- Cranck-Nicolson solver (#413, @michaeldeistler)
- Forward Euler solver for compartments and branches (#413, @michaeldeistler)
- Add option to access `states` in `channel.init_state` (#416, @michaeldeistler)

### 🐛 Bug fixes

- Bugfix for interpolation of x, y, z values (#411, @jnsbck)


# v0.1.2

- Minor fixups for README, website, and PyPI landing page


# v0.1.1

- Minor fixups for README, website, and PyPI landing page


# v0.1.0

- First public version
