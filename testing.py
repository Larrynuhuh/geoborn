
import jax
import jax.numpy as jnp

from manifolds import vectors as vct
from manifolds import metrics as mtc
from basis import linear as lin

square = jnp.array([
[0.,0.],
[1.,0.],
[1.,1.],
[0.,1.],
[0.,0.]
])

inside = jnp.array([0.5, 0.5])
outside = jnp.array([1.5, 0.5])

print(mtc.sdf(square, outside))
