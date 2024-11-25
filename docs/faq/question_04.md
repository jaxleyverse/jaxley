# How can I implement rate-based neuron models in Jaxley?

In this FAQ, we explain how one can implement rate-based neuron models of the form:

$$
\tau \frac{dV}{dt} = -V + \sum w_{\text{syn}} \phi(V_{\text{pre}})
$$

Here, $\phi$ is a nonlinearity such as a `tanh` or a `ReLU`.

To implement this in `Jaxley`, we first have to set up a network consisting of 
point-neurons:
```python
import jaxley as jx

num_cells = 100
cell = jx.Cell()  # Create a point-neuron.
net = jx.Network([cell for _ in range(num_cells)])
```

Next, we have to equip the neurons with a `Leak` so as to model:
$C \cdot dV/dt = -V$

```python
from jaxley.channels import Leak

net.insert(Leak())
net.set("Leak_eLeak", 0.0)  # Center the dynamics around zero.
net.set("Leak_gLeak", 1.0)  # We will deal with the time-constant later.
```

Next, we have to connect the cells with `Tanh` synapses:
```python
from jaxley.connect import fully_connect
from jaxley.synapses import TanhRateSynapse

fully_connect(net.cell("all"), net.cell("all"), TanhRateSynapse())
```

Lastly, what rate-based neuron models call the time constant is called the `capacitance`
in `Jaxley`:
```python
net.set("capacitance", 2.0)  # Default is 1.0.
```

That's it! As always, you can inspect your network by looking at `net.nodes` and
`net.edges`.

Equipped with this network, you can check out the 
[tutorial on how to simulate network models in Jaxley](https://jaxley.readthedocs.io/en/latest/tutorials/02_small_network.html).
You can also check out the
[API reference on different connect() methods](https://jaxley.readthedocs.io/en/latest/reference/jaxley.connect.html)
(e.g. `sparse_connect()`) or the
[tutorial on customizing synaptic parameters](https://jaxley.readthedocs.io/en/latest/tutorials/09_advanced_indexing.html).