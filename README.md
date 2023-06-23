[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/neurax/blob/main/CONTRIBUTING.md)
[![Tests](https://github.com/mackelab/neurax/workflows/Tests/badge.svg?branch=main)](https://github.com/mackelab/neurax/actions)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/mackelab/neurax/blob/main/LICENSE)


# neurax
`neurax` is a differentiable simulator for networks of multicompartment neurons in JAX. Its key features are:
- automatic differentiation, allowing gradient-based optimization of thousands of parameters
- support for CPU and GPU without any changes to the code
-`jit`-compilation, making it as fast as other packages while being written fully in python
- backward-Euler solver for stable numerical solution of multicompartment neurons
- elegant mechanisms for parameter sharing

### Tutorial

Tutorial notebooks with some explanation are in `tutorials`.

### Units

`neurax` uses the same [units as `NEURON`](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).

### Installation

```sh
git clone https://github.com/mackelab/neurax.git
cd neurax
pip install -e .
```
