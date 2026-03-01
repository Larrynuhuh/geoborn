import geoutils as us
import jax 
import jax.numpy as jnp
from ops import vectors as vct

@jax.jit
def linlen(l):
    
    diff = jnp.diff(l, axis = 0)
    lens = jnp.linalg.norm(diff, axis = 1)
    sums = jnp.sum(lens)

    return sums


def midp(p1, p2):
    mid = (p1 + p2) / 2.0
    return mid

vmidp = jax.jit(jax.vmap(midp))


# to check distance of point from line

@jax.jit
def segdist(f, g, pt):
    v = g-f
    w = pt-f
    t = us.div(jnp.dot(w,v),jnp.dot(v,v))
    tc = jnp.clip(t, 0, 1)

    cp = f + tc * v
    dist = jnp.linalg.norm(pt - cp)

    return dist

# USER CALLS PLDIST
@jax.jit
def pldist(l, pt):
    a = l[:-1]
    b = l[1:]

    curve_dist = jax.jit(jax.vmap(segdist, in_axes = (0, 0, None)))
    summed = curve_dist(a, b, pt)

    return jnp.min(summed)


vpldist = jax.jit(jax.vmap(pldist, in_axes = (None, 0)))

@jax.jit
def sdf(l, pt):
    
    a = l[:-1]
    b = l[1:]

    dist_func = jax.vmap(segdist, in_axes = (0, 0, None))
    dists = dist_func(a, b, pt)

    cidx = jnp.argmin(dists)
    cdist = dists[cidx]

    cseg = jnp.stack([a[cidx], b[cidx]])
    norm = vct.normal(cseg)

    mid = (a[cidx] + b[cidx])/2.0
    direction = pt - mid
    
    sign = jnp.sign(jnp.dot(direction, norm))

    return cdist * sign 


vsdf = jax.jit(jax.vmap(sdf, in_axes = (0, 0)))