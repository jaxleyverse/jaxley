# What kinds of models can be implemented in Jaxley?

### Biophysical neuron models

`Jaxley` focuses on biophysical, Hodgkin-Huxley-type models, as well as networks thereof. In particular, `Jaxley` supports:

- single-compartment (point neuron) Hodgkin-Huxley models,
- multi-compartment Hodgkin-Huxley models.

For these models, `Jaxley` is flexible and accurate. For example, it can flexibly [add new channel models](https://jaxley.readthedocs.io/en/latest/tutorials/05_channel_and_synapse_models.html), use [different kinds of synapses (conductance-based, tanh, ...)](https://github.com/jaxleyverse/jaxley/tree/main/jaxley/synapses), it can [insert different kinds of channels in different branches](https://jaxley.readthedocs.io/en/latest/tutorials/01_morph_neurons.html) (or compartments) within single cells, and it can [simulate complex ion dynamics (diffusion, pumps,...)](https://jaxley.readthedocs.io/en/latest/tutorials/11_ion_dynamics.html). You can think of `Jaxley` like [the `NEURON` simulator](https://neuron.yale.edu/neuron/) written in `JAX`. Like `NEURON`, `Jaxley` implements a backward-Euler solver for stable numerical solution of multi-compartment neurons.

### Simplified neuron models

`Jaxley` can also simulate simplified models, see the tutorial [here](https://jaxley.readthedocs.io/en/latest/tutorials/12_simplified_models.html). In particular, `Jaxley` supports:
- Leaky-integrate-and-fire (LIF) neurons,  
- Izhikevich neurons,  
- Rate-based neurons.  

`Jaxley` also supports networks of these neurons.

Note that, for LIF neurons, `Jaxley` additionally supports surrogate gradient descent as of version 0.11.7, which is required for efficient training due to the discontinuity of spikes in these models. Note that for the Izhikevich model, surrogate gradients are not supported by default, but one can modify the Izhikevich model to include support. See the tutorial linked above.
