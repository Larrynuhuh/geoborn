import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor

@jax.jit
def points(*dimens: int) -> Matrix: 

    state = jnp.meshgrid(*dimens, indexing = 'ij')
    p = jnp.stack(state, axis = -1)
    c = p.reshape(-1, len(dimens))

    return c

@jax.jit
def line(p1: Vector, p2: Vector, segs: int) -> Matrix:
    t = jnp.linspace(0, 1, segs)[:, jnp.newaxis]

    l = p1 + (t * (p2 - p1))

    return l

vline = jax.jit(jax.vmap(line, in_axes = (0, 0, None)))

@jax.jit
def polyline(pl: Matrix) -> Tensor:
    a = pl[:-1]
    b = pl[1:]
    c = jnp.stack([a, b], axis = 1)

    return c

vpolyline = jax.jit(jax.vmap(polyline, in_axes = 0))

