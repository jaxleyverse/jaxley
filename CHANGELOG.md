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