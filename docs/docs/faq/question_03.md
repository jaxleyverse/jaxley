# What kinds of models can be implemented in `Jaxley`?

`Jaxley` focuses on biophysical, Hodgkin-Huxley-type models. You can think of `Jaxley` like [the `NEURON` simulator](https://neuron.yale.edu/neuron/) written in `JAX`.

In short, `Jaxley` allows to simulate the following types of models:

- single-compartment (point neuron) Hodgkin-Huxley models
- multi-compartment Hodgkin-Huxley models
- rate-based neuron models

For all of these models, `Jaxley` is flexible and accurate. For example, it can flexibly [add new channel models](https://jaxleyverse.github.io/jaxley/tutorial/05_channel_and_synapse_models/), use [different kinds of synapses (conductance-based, tanh, ...)](https://github.com/jaxleyverse/jaxley/tree/main/jaxley/synapses), and it can [insert different kinds of channels in different branches](https://jaxleyverse.github.io/jaxley/tutorial/01_morph_neurons/) (or compartments) within single cells. Like `NEURON`, `Jaxley` implements a backward-Euler solver for stable numerical solution of multi-compartment neurons.

However, `Jaxley` does **not** implement the following types of models:

- leaky-integrate and fire neurons
- Ishikevich neuron models
- etc...
