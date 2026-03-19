import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
 
from numba import njit, prange
import time


g = jnp.array([
    [1.0, 0.999999],
    [0.999999, 1.0]
])

basis = jnp.eye(2)

print(vct.nrm(g, basis))