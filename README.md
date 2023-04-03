# neurax
A differentiable neuron simulator in jax (by Michael)

### Units

`neurax` uses the same [units as `NEURON`](https://www.neuron.yale.edu/neuron/static/docs/units/unitchart.html).

### Limitations

In order to vectorize the elimination phase of the triangularization, we assume that each branch must the parent of exactly two (or zeros) other branches. This means that we do not allow threeway branches. See [this PR](https://github.com/mackelab/neurax/pull/10).
