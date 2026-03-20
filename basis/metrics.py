import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

@jax.jit 
def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

def iprod(g: Matrix, u: Vector|Matrix, v: Vector|Matrix) -> Vector:
    return jnp.einsum('...i, ...ij, ...j -> ...', u, g, v)


@jax.jit
def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(jnp.maximum(iprod(g, u, u), 0.0))

def norm_sq(g: Matrix, u: Vector) -> Scalar: 
    return jnp.maximum(iprod(g, u, u), 0.0)


@jax.jit(static_argnums = (0,))
def fwdmet(f, v: Vector) -> Matrix:
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)

@jax.jit(static_argnums = (0,))
def revmet(f, v: Vector) -> Matrix:
    J = jax.jacrev(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return jnp.einsum('ai, aj -> ij', nJ, nJ)


