import jax
import jax.numpy as jnp

def safe_div(a, b):
    eps = 1e-8
    return a/(b + eps)

@jax.jit
def pld(pts, pt):
    a = pts[:-1]
    b = pts[1:]

    def seg_dist(f, g, p):
        v = g-f
        w = p-f
        t = jnp.dot(w,v) / (jnp.dot(v, v) + 1e-8)
        t = jnp.clip(t, 0.0, 1.0)
        closest = f + t*v
        return jnp.linalg.norm(p-closest)
    
    dists = jax.vmap(seg_dist, in_axes = (0, 0, None))(a, b, pt)

    return jnp.min(dists)


wall = jnp.array([[10.0, 0.0], [10.0, 10.0]])
me = jnp.array([5.0, 5.0])
try: 
    d = pld(wall, me)
    print(f'distance: {d}')

except Exception as e: 
    print(f'fuck this shit  {e}.')