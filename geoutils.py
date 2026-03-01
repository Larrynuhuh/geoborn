import jax
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

@jax.jit
def div(a, b):
    safe = b != 0
    den = jnp.where(safe, b, 1.0)

    return jnp.where(safe, a/den, 0.0)


eps = 1e-15


type Vector = Array # 1D [1, 2]
type Matrix = Array # 2D [[1, 2], [2, 1]]
type Scalar = Array # 0D [1]
type Tensor = Array # N-D