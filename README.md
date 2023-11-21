[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/jaxley/blob/main/CONTRIBUTING.md)
[![Tests](https://github.com/mackelab/jaxley/workflows/Tests/badge.svg?branch=main)](https://github.com/mackelab/jaxley/actions)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/mackelab/jaxley/blob/main/LICENSE)


<p align="center">
  <img src="docs/logo.png?raw=true" width="360">
</p>

`jaxley` is a differentiable simulator for networks of multicompartment neurons in [JAX](https://github.com/google/jax). Its key features are:
- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU and GPU without any changes to the code  
- `jit`-compilation, making it as fast as other packages while being fully written in python  
- backward-Euler solver for stable numerical solution of multicompartment neurons  
- elegant mechanisms for parameter sharing


### Tutorial

Tutorial notebooks with some explanation are in [`tutorials`](https://github.com/mackelab/jaxley/tree/main/tutorials). We currently have tutorials on how to:
- [run a simple network simulation](https://github.com/mackelab/jaxley/blob/main/tutorials/01_small_network.ipynb)
- [set parameters](https://github.com/mackelab/jaxley/blob/main/tutorials/02_setting_parameters.ipynb)
- [obtain a gradient and train](https://github.com/mackelab/jaxley/blob/main/tutorials/03_gradient.ipynb)
- [define groups (aka sectionlists)](https://github.com/mackelab/jaxley/blob/main/tutorials/04_groups.ipynb)
- [define your own channels and synapses](https://github.com/mackelab/jaxley/blob/main/tutorials/05_new_mechanisms.ipynb)
- [use diverse channels](https://github.com/mackelab/jaxley/blob/main/tutorials/06_diverse_channels.ipynb)


### Units

`jaxley` uses the same [units as `NEURON`](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).


### Installation
`jaxley` requires that you first download and install [tridiax](https://github.com/mackelab/tridiax). Then, install `jaxley` via:
```sh
git clone https://github.com/mackelab/jaxley.git
cd jaxley
pip install -e .
```


### Feedback and Contributions

We welcome any feedback on how jaxley is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see [contribute](https://github.com/mackelab/jaxley/blob/main/CONTRIBUTING.md)). We wish to maintain a positive community, please read our [Code of Conduct](https://github.com/mackelab/jaxley/blob/main/CODE_OF_CONDUCT.md).


### Acknowledgements

We greatly benefited from previous toolboxes for simulating multicompartment neurons, in particular [NEURON](https://github.com/neuronsimulator/nrn).


### License

[MIT License](https://github.com/mackelab/jaxley/blob/main/LICENSE)


### Citation

If you use `jaxley`, consider citing the corresponding paper:
```
@article{}
```