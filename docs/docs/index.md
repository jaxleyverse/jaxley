
![jaxley_logo](logo.png)

`jaxley` is a differentiable simulator for networks of multicompartment neurons in [JAX](https://github.com/google/jax). Its key features are:

- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU and GPU without any changes to the code  
- `jit`-compilation, making it as fast as other packages while being fully written in python  
- backward-Euler solver for stable numerical solution of multicompartment neurons  
- elegant mechanisms for parameter sharing


### Tutorial

Tutorial notebooks with some explanation are in [`tutorials`](https://github.com/jaxleyverse/jaxley/tree/main/tutorials). We currently have tutorials on how to:

- [simulate morphologically detailed neurons](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/01_morph_neurons.ipynb)
- [simulate networks of such neurons](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/02_small_network.ipynb)
- [set parameters of cells and networks](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/03_setting_parameters.ipynb)
- [speed up simulations with jit and vmap](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/04_jit_and_vmap.ipynb)
- [define your own channels and synapses](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/05_channel_and_synapse_models.ipynb)
- [define groups](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/06_groups.ipynb)
- [train biophysical models](https://github.com/jaxleyverse/jaxley/blob/main/tutorials/07_gradient_descent.ipynb)


### Units

`jaxley` uses the same [units as `NEURON`](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).


### Installation
`jaxley` requires that you first download and install [tridiax](https://github.com/jaxleyverse/tridiax). Then, install `jaxley` via:
```sh
git clone https://github.com/jaxleyverse/jaxley.git
cd jaxley
pip install -e .
```

Note that `pip>=21.3` is required to install the editable version with `pyproject.toml` see [pip docs](https://pip.pypa.io/en/latest/reference/build-system/pyproject-toml/#editable-installation). 


### Feedback and Contributions

We welcome any feedback on how jaxley is working for your neuron models and are happy to receive bug reports, pull requests and other feedback (see [contribute](https://github.com/jaxleyverse/jaxley/blob/main/CONTRIBUTING.md)). We wish to maintain a positive community, please read our [Code of Conduct](https://github.com/jaxleyverse/jaxley/blob/main/CODE_OF_CONDUCT.md).


### Acknowledgements

We greatly benefited from previous toolboxes for simulating multicompartment neurons, in particular [NEURON](https://github.com/neuronsimulator/nrn).


### License

[MIT License](https://github.com/jaxleyverse/jaxley/blob/main/LICENSE)


### Citation

If you use `jaxley`, consider citing the corresponding paper:
```
@article{}
```