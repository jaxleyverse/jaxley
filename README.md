[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mackelab/neurax/blob/main/CONTRIBUTING.md)
[![Tests](https://github.com/mackelab/neurax/workflows/Tests/badge.svg?branch=main)](https://github.com/mackelab/neurax/actions)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/mackelab/neurax/blob/main/LICENSE)


<h1 align="center">
neurax
</h1>

`neurax` is a differentiable simulator for networks of multicompartment neurons in [JAX](https://github.com/google/jax). Its key features are:
- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU and GPU without any changes to the code  
- `jit`-compilation, making it as fast as other packages while being fully written in python  
- backward-Euler solver for stable numerical solution of multicompartment neurons  
- elegant mechanisms for parameter sharing


### Tutorial

Tutorial notebooks with some explanation are in [`tutorials`](https://github.com/mackelab/neurax/tree/main/tutorials). We currently have tutorials on how to:
- [run a simple network simulation](https://github.com/mackelab/neurax/blob/main/tutorials/01_small_network.ipynb)
- [set parameters](https://github.com/mackelab/neurax/blob/main/tutorials/02_setting_parameters.ipynb)
- [obtain a gradient and train](https://github.com/mackelab/neurax/blob/main/tutorials/03_gradient.ipynb)
- [define groups (aka sectionlists)](https://github.com/mackelab/neurax/blob/main/tutorials/04_groups.ipynb)


### Units

`neurax` uses the same [units as `NEURON`](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).


### Installation
`neurax` requires that you first download and install [tridiax](https://github.com/mackelab/tridiax). Then, install `neurax` via:
```sh
git clone https://github.com/mackelab/neurax.git
cd neurax
pip install -e .
```


### Feedback and Contributions

We welcome any feedback on how neurax is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see [contribute](https://github.com/mackelab/neurax/blob/main/CONTRIBUTING.md)). We wish to maintain a positive community, please read our [Code of Conduct](https://github.com/mackelab/neurax/blob/main/CODE_OF_CONDUCT.md).


### Acknowledgements

We greatly benefited from previous toolboxes for simulating multicompartment neurons, in particular [NEURON](https://github.com/neuronsimulator/nrn).


### License

[MIT License](https://github.com/mackelab/neurax/blob/main/LICENSE)


### Citation

If you use `neurax`, consider citing the corresponding paper:
```
@article{}
```