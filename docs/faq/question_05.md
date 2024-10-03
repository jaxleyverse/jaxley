# Can I implement complex intracellular dynamics (e.g., ion diffusion) in Jaxley?

Yes, you can (as of version `0.7.0`)! This is explained in [this tutorial](https://jaxley.readthedocs.io/en/latest/tutorials/11_ion_dynamics.html). `Jaxley` allows to:
- define ion pumps,  
- update the reversal potential via the Nernst equation, and  
- allows to diffuse ions within a cell.  

Here is a small code snippet that shows this (this snippet requires `pip install jaxley-mech`):
```python
from jaxley_mech.channels.l5pc import CaHVA, CaNernstPotential, CaPump


branch = jx.Branch()
cell = jx.Cell(branch, parents=[-1, 0, 0])

# Insert a voltage-gated calcium channel.
cell.insert(CaHVA())

# Insert an ion pump which modifies the intracellular calcium based on the calcium current.
cell.insert(CaPump())

# Insert a mechanism that updates the calcium reversal potential based on the intracellular calcium level.
cell.insert(CaNernstPotential())

# Let the intracellular calcium diffuse within the cell.
cell.diffuse("CaCon_i")
cell.set("axial_resistivity_CaCon_i", 1_000.0)
```

`Jaxley` does not yet provide functionality for diffusing ions extracellularly.