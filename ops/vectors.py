
import utils as us
import jax 
import jax.numpy as jnp

@jax.jit
def normal(basis):
    center = jnp.mean(basis, axis = 0)
    cb = basis - center

    u, s, vh = jnp.linalg.svd(cb, full_matrices = False)
    normal = vh[-1]

    check = jnp.dot(normal, center)
    nrm = jnp.where(check < 0, -normal, normal)
    return us.div(nrm, (jnp.linalg.norm(nrm) + us.eps))

vnormal = jax.vmap(normal)




