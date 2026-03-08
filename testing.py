import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
import geoutils as us 


test_indices = jnp.array([0, 4, 8])
grid_shape = (3, 3)

# 2. Run it
result = lin.grid(test_indices, grid_shape)

# 3. Visual Verification
print("Resulting Array:\n", result)
print("Shape:", result.shape)
print("Type:", type(result))