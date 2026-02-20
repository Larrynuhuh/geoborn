import jax
import jax.numpy as jnp
import utils as us

def normal(basis):

    u, s, vh = jnp.linalg.svd(basis, full_matrices = True)
    normal = vh[-1]
    return normal/ (jnp.linalg.norm(normal) + us.eps)