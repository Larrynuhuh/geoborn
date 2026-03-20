import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

def grid(idx: JAXArray, dimens: tuple):
    fg = jnp.unravel_index(idx, dimens)
    g = fg[::-1]
    ng = jnp.stack(g, axis=-1)

    return ng

static_argnums = (2,)
def line(p1: Vector, p2: Vector, segs: int) -> Matrix:
    t = jnp.linspace(0, 1, segs)[:, jnp.newaxis]

    l = p1 + (t * (p2 - p1))

    return l

static_argnums = (2,)
def xline(p1: Matrix, p2: Matrix, segs: int) -> Tensor:
    return jax.vmap(line, in_axes = (0, 0, None))(p1, p2, segs)

def polyline(pl: Matrix) -> Tensor:
    a = pl[:-1]
    b = pl[1:]
    c = jnp.stack([a, b], axis = 1)

    return c


def xpolyline(pl: Tensor) -> Tensor: 
    return jax.vmap(polyline, in_axes = (0,))
