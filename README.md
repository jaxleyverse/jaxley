[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md)
[![Tests](https://github.com/jaxleyverse/jaxley/workflows/Tests/badge.svg?branch=main)](https://github.com/jaxleyverse/jaxley/actions)
[![GitHub license](https://img.shields.io/github/license/sbi-dev/sbi)](https://github.com/jaxleyverse/jaxley/blob/main/LICENSE)


<p align="center">
  <img src="https://raw.githubusercontent.com/jaxleyverse/jaxley/main/docs/logo.png" width="360">
</p>

[Getting Started](https://jaxleyverse.github.io/jaxley/tutorial/01_morph_neurons/) | [Documentation](https://jaxleyverse.github.io/jaxley/)

`Jaxley` is a differentiable simulator for biophysical neuron models in [JAX](https://github.com/google/jax). Its key features are:

- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU, GPU, or TPU without any changes to the code  
- `jit`-compilation, making it as fast as other packages while being fully written in python  
- backward-Euler solver for stable numerical solution of multicompartment neurons  
- elegant mechanisms for parameter sharing


### Tutorials

Tutorials are available [on our website](https://jaxleyverse.github.io/jaxley/). We currently have tutorials on how to:

- [simulate morphologically detailed neurons](https://jaxleyverse.github.io/jaxley/tutorial/01_morph_neurons/)
- [simulate networks of such neurons](https://jaxleyverse.github.io/jaxley/tutorial/02_small_network/)
- [set parameters of cells and networks](https://jaxleyverse.github.io/jaxley/tutorial/03_setting_parameters/)
- [speed up simulations with jit and vmap](https://jaxleyverse.github.io/jaxley/tutorial/04_jit_and_vmap/)
- [define your own channels and synapses](https://jaxleyverse.github.io/jaxley/tutorial/05_channel_and_synapse_models/)
- [define groups](https://jaxleyverse.github.io/jaxley/tutorial/06_groups/)
- [read and handle SWC files](https://jaxleyverse.github.io/jaxley/tutorial/08_importing_morphologies/)
- [train biophysical models](https://jaxleyverse.github.io/jaxley/tutorial/07_gradient_descent/)


### Units

`Jaxley` uses the same [units as `NEURON`](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).


### Installation
`Jaxley` is available on [`PyPI`](https://pypi.org/project/jaxley/):
```sh
pip install jaxley
```
This will install `Jaxley` with CPU support. If you want GPU support, follow the instructions on the [`JAX` Github repository](https://github.com/google/jax) to install `JAX` with GPU support (in addition to installing `Jaxley`). For example, for NVIDIA GPUs, run
```sh
pip install -U "jax[cuda12]"
```


### Feedback and Contributions

We welcome any feedback on how Jaxley is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see [contribute](https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md)). We wish to maintain a positive community, please read our [Code of Conduct](https://github.com/jaxleyverse/jaxley/blob/main/CODE_OF_CONDUCT.md).


### Acknowledgements

We greatly benefited from previous toolboxes for simulating multicompartment neurons, in particular [NEURON](https://github.com/neuronsimulator/nrn).


### License

[Apache License](https://github.com/jaxleyverse/jaxley/blob/main/LICENSE)
