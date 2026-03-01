
import geoutils as us
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
    return us.div(nrm, (jnp.linalg.norm(nrm)))

vnormal = jax.jit(jax.vmap(normal, in_axes = (0)))

#dot product territory

def project_scalar(a, b): 
    
    norm = jnp.linalg.norm(b)
    prod = us.div(jnp.dot(a, b), norm)

    return prod

scalproj = jax.jit(jax.vmap(project_scalar, in_axes = (0, 0)))

def project_vector(a, b):

    norm = jnp.linalg.norm(b)
    prod = us.div(jnp.dot(a, b), norm)
    term = us.div(b, norm)
    proj = prod * term

    return proj

vectproj = jax.jit(jax.vmap(project_vector, in_axes = (0, 0)))


def reject_vector(a, b):

    proj = project_vector(a, b)
    reject = a - proj

    return reject

rejvect = jax.jit(jax.vmap(reject_vector, in_axes = (0, 0)))
