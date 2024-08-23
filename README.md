<p align="center">
  <img src="https://raw.githubusercontent.com/jaxleyverse/jaxley/main/docs/logo.png" width="360">
</p>

<h1 align="center">Differentiable neuron simulations on CPU, GPU, or TPU</h1>

[![PyPI version](https://badge.fury.io/py/jaxley.svg)](https://badge.fury.io/py/jaxley)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md)
[![Tests](https://github.com/jaxleyverse/jaxley/workflows/Tests/badge.svg?branch=main)](https://github.com/jaxleyverse/jaxley/actions)
[![GitHub license](https://img.shields.io/github/license/jaxleyverse/jaxley)](https://github.com/jaxleyverse/jaxley/blob/main/LICENSE)

[**Documentation**](https://jaxleyverse.github.io/jaxley/)
 | [**Getting Started**](https://jaxleyverse.github.io/jaxley/tutorial/01_morph_neurons/)
 | [**Install guide**](https://jaxleyverse.github.io/jaxley/install/)
 | [**Reference docs**](https://jaxleyverse.github.io/jaxley/reference/modules/)
 | [**FAQ**](https://jaxleyverse.github.io/jaxley/faq/)


## What is Jaxley?

`Jaxley` is a differentiable simulator for biophysical neuron models, written in the Python library [JAX](https://github.com/google/jax). Its key features are:

- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU, GPU, or TPU without any changes to the code  
- `jit`-compilation, making it as fast as other packages while being fully written in python  
- support for multicompartment neurons  
- elegant mechanisms for parameter sharing  


## Getting started

`Jaxley` allows to simulate biophysical neuron models on CPU, GPU, or TPU:
```python
import matplotlib.pyplot as plt
from jax import config

import jaxley as jx
from jaxley.channels import HH

config.update("jax_platform_name", "cpu")  # Or "gpu" / "tpu".

cell = jx.Cell()  # Define cell.
cell.insert(HH())  # Insert channels.

current = jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.1, delta_t=0.025, t_max=10.0)
cell.stimulate(current)  # Stimulate with step current.
cell.record("v")  # Record voltage.

v = jx.integrate(cell)  # Run simulation.
plt.plot(v.T)  # Plot voltage trace.
```

If you want to learn more, we recommend you to check out our tutorials on how to:

- [get started with `Jaxley`](https://jaxleyverse.github.io/jaxley/tutorial/01_morph_neurons/)
- [simulate networks of neurons](https://jaxleyverse.github.io/jaxley/tutorial/02_small_network/)
- [speed up simulations with GPUs and `jit`](https://jaxleyverse.github.io/jaxley/tutorial/04_jit_and_vmap/)
- [define your own channels and synapses](https://jaxleyverse.github.io/jaxley/tutorial/05_channel_and_synapse_models/)
- [compute the gradient and train biophysical models](https://jaxleyverse.github.io/jaxley/tutorial/07_gradient_descent/)


## Installation

`Jaxley` is available on [`PyPI`](https://pypi.org/project/jaxley/):
```sh
pip install jaxley
```
This will install `Jaxley` with CPU support. If you want GPU support, follow the instructions on the [`JAX` Github repository](https://github.com/google/jax) to install `JAX` with GPU support (in addition to installing `Jaxley`). For example, for NVIDIA GPUs, run
```sh
pip install -U "jax[cuda12]"
```


## Feedback and Contributions

We welcome any feedback on how Jaxley is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see [contribute](https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md)). We wish to maintain a positive community, please read our [Code of Conduct](https://github.com/jaxleyverse/jaxley/blob/main/CODE_OF_CONDUCT.md).


## License

[Apache License Version 2.0 (Apache-2.0)](https://github.com/jaxleyverse/jaxley/blob/main/LICENSE)


## Citation

If you use `Jaxley`, consider citing the [corresponding paper](https://www.biorxiv.org/content/10.1101/2024.08.21.608979):

```
@article{deistler2024differentiable,
  doi = {10.1101/2024.08.21.608979},
  year = {2024},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Deistler, Michael and Kadhim, Kyra L. and Pals, Matthijs and Beck, Jonas and Huang, Ziwei and Gloeckler, Manuel and Lappalainen, Janne K. and Schr{\"o}der, Cornelius and Berens, Philipp and Gon{\c c}alves, Pedro J. and Macke, Jakob H.},
  title = {Differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics},
  journal = {bioRxiv}
}
```