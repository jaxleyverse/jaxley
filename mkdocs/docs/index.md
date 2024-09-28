
<p align="center">
  <img src="logo.png?raw=true" width="360">
</p>

`Jaxley` is a differentiable simulator for biophysical neuron models in [JAX](https://github.com/google/jax). Its key features are:

- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU, GPU, or TPU without any changes to the code  
- `jit`-compilation, making it as fast as other packages while being fully written in python  
- backward-Euler solver for stable numerical solution of multicompartment neurons  
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

If you want to learn more, we have tutorials on how to:

- [simulate morphologically detailed neurons](https://jaxleyverse.github.io/jaxley/tutorial/01_morph_neurons/)
- [simulate networks of such neurons](https://jaxleyverse.github.io/jaxley/tutorial/02_small_network/)
- [set parameters of cells and networks](https://jaxleyverse.github.io/jaxley/tutorial/03_setting_parameters/)
- [speed up simulations with GPUs and jit](https://jaxleyverse.github.io/jaxley/tutorial/04_jit_and_vmap/)
- [define your own channels and synapses](https://jaxleyverse.github.io/jaxley/tutorial/05_channel_and_synapse_models/)
- [define groups](https://jaxleyverse.github.io/jaxley/tutorial/06_groups/)
- [read and handle SWC files](https://jaxleyverse.github.io/jaxley/tutorial/08_importing_morphologies/)
- [compute the gradient and train biophysical models](https://jaxleyverse.github.io/jaxley/tutorial/07_gradient_descent/)


## Installation

`Jaxley` is available on [`pypi`](https://pypi.org/project/jaxley/):
```sh
pip install jaxley
```
This will install `Jaxley` with CPU support. If you want GPU support, follow the instructions on the [`JAX` github repository](https://github.com/google/jax) to install `JAX` with GPU support (in addition to installing `Jaxley`). For example, for NVIDIA GPUs, run
```sh
pip install -U "jax[cuda12]"
```


## Feedback and Contributions

We welcome any feedback on how `Jaxley` is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see [contribute](https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md)). We wish to maintain a positive community, please read our [Code of Conduct](https://github.com/jaxleyverse/jaxley/blob/main/CODE_OF_CONDUCT.md).


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