# How can I save and load cells and networks?

All `module`s (i.e., compartments, branches, cells, and networks) in `Jaxley` can be saved and loaded with pickle:
```python
import jaxley as jx
import pickle

# ... define network, cell, etc.
network = jx.Network([cell1, cell2])

# Save.
with open("path/to/file.pkl", "wb") as handle:
    pickle.dump(network, handle)

# Load.
with open("path/to/file.pkl", "rb") as handle:
    network = pickle.dump(handle)
```