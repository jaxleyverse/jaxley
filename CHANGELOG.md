# 0.6.0

### New Features

- Add ability to record synaptic currents (#523, @ntolley). Recordings can be turned on with
```python
net.record("i_IonotropicSynapse")
```
- Add regression tests and supporting workflows for maintaining baselines (#475, #546, @jnsbck).
  - Regression tests can be triggered by commenting "/test_regression" on a PR.
  - Regression tests can be done locally by running `NEW_BASELINE=1 pytest -m regression` i.e. on `main` and then `pytest -m regression` on `feature`, which will produce a test report (printed to the console and saved to .txt).

- refactor plotting (#539, @jnsbck).
  - rm networkx dependency
  - add `Network.arrange_in_layers`
  - disentangle moving of cells and plotting in `Network.vis`. To get the same as `net.vis(layers=[3,3])`, one now has to do:
```python
net.arrange_in_layers([3,3])
net.vis()
```

- Allow parameter sharing for groups of different sizes, i.e. due to inhomogenous numbers of compartments or for synapses with the same (pre-)synaptic parameters but different numbers of post-synaptic partners. (#514, @jnsbck)

- changelog added to CI (#537, #558,  @jnsbck)

### Bug fixes
- Fixed inconsistency with *type* assertions arising due to `numpy` functions returning different `dtypes` on platforms like Windows (#567, @Kartik-Sama)

# 0.5.0

### API changes

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
- Throughout the codebase, we renamed any occurance of `seg` (for `segment`) to `comp`
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

### New features

- Added a new `delete_channel()` method (#521, @jnsbck)
- Allow to write trainables to the module (#470, @michaeldeistler):
```python
net.make_trainable("radius")
params = net.get_parameters()
net.write_trainables(params)
```
- Expose the step function to allow for fine-grained simulation (#466, @manuelgloeckler)
- More flexible and throrough viewing (#447, #453, @jnsbck)
- Boolean indexing for cells, branches, and comps (@494, @jnsbck):
```python
r_greater_1 = net.nodes.groupby("global_cell_index")["radius"].mean() > 1
net[r_greater_1].nodes.vis()
```  
- check if recordings are empty (#460, @deezer257)
- enable `clamp` to be jitted and vmapped with `data_clamp()` (#374, @kyralianaka)

### Bug fixes

- allow for cells that were read from swc to be pickled (#525, @jnsbck)
- fix units of `compute_current()` in channels (#461, @michaeldeistler)
- fix issues with plotting when the morphology has a different number of compartments
(#513, @jnsbck)

### Documentation

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


# 0.4.0

### New features

- Changing the number of compartments: `cell.branch(0).set_ncomp(4)` (#436, #440, #445,
@michaeldeistler, @jnsbck)
- New options for plotting: `cell.vis(type='comp')` and `cell.vis(type='morph')` (#432,
#437, @jnsbck)
- Speed optimization for `jx.integrate(..., voltage_solver="jaxley.stone")` (#442,
@michaeldeistler)

### Documentation

- new website powered by sphinx:
[`jaxley.readthedocs.io`](https://jaxley.readthedocs.io/) (#434, #435, @michaeldeistler)


# v0.3.0

### New features

- New solver: `jx.integrate(..., voltage_solver="jax.sparse")` which has very low
compile time (#418, @michaeldeistler)
- Support for different number of compartments per branch at initilization (modifying
the number of compartments after initialization is not yet supported, #418, #426, 
@michaeldeistler)

### Bugfixes

- Bugfix for capacitances and their interplay with axial conductances (Thanks @Tunenip, 
#426, @michaeldeistler)
- Bugfixes for tutorials on website


# v0.2.1

- Bugfix for using `states` in `init_state` of the channel (#421, @michaeldeistler)
- Bugfix for tutorial on building custom channels (#421, @michaeldeistler)
- Add `jaxley-mech` as `dev` dependency to use it in testsing (#421, @michaeldeistler)


# v0.2.0

### New features

- Cranck-Nicolson solver (#413, @michaeldeistler)
- Forward Euler solver for compartments and branches (#413, @michaeldeistler)
- Add option to access `states` in `channel.init_state` (#416, @michaeldeistler)

### Bugfixes

- Bugfix for interpolation of x, y, z values (#411, @jnsbck)


# v0.1.2

- Minor fixups for README, website, and PyPI landing page


# v0.1.1

- Minor fixups for README, website, and PyPI landing page


# v0.1.0

- First public version