import utils as us
import jax 
import jax.numpy as jnp

@jax.jit
def linlen(l):
    
    diff = jnp.diff(l, axis = 0)
    lens = jnp.linalg.norm(diff, axis = 1)
    sums = jnp.sum(lens)

    return sums


def midp(p1, p2):
    mid = (p1 + p2) / 2.0
    return mid

vmidp = jax.vmap(midp)


# to check distance of point from line
import utils as us

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

    curve_dist = jax.vmap(segdist, in_axes = (0, 0, None))
    summed = curve_dist(a, b, pt)

    return jnp.min(summed)


vpldist = jax.vmap(pldist)