import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor

@jax.jit 
def euclid(x: Vector) -> Matrix:
    return jnp.eye(x.shape[-1])

@jax.jit
def iprod(g: Matrix, u: Vector, v: Vector) -> Scalar: 
    return jnp.dot(u, jnp.dot(g, v))

@jax.jit
def norm(g: Matrix, u: Vector) -> Scalar: 
    return jnp.sqrt(iprod(g, u, u))

@jax.jit
def det(g: Matrix, dg: Matrix) -> Scalar:
    d = jnp.linalg.det(g)
    s = jnp.abs(d) > us.eps

    inv = jnp.linalg.inv(jnp.where(s, g, jnp.eye(g.shape[-1])))
    grad = d * jnp.trace(inv @ dg)
    
    val = jnp.where(s, d, 0.0)
    grad = jnp.where(s, grad, 0.0)

    return val, grad

@jax.jit(static_argnums = (0,))
def metmap(f, v: Vector) -> Matrix:
    J = jax.jacfwd(f)(v)
    nJ = J.reshape(-1, v.shape[-1])
    return nJ.T @ nJ