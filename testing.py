import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
 
from numba import njit, prange
import time


NUM_LINE_POINTS = 5000
NUM_QUERY_POINTS = 1000

# 1. 3D Spiral Path (x=t, y=sin(t), z=cos(t))
t = jnp.linspace(0, 10, NUM_LINE_POINTS)
line_3d = jnp.stack([t, jnp.sin(t), jnp.cos(t)], axis=-1)

# 2. 1,000 Random 3D Query Points
key = jax.random.PRNGKey(42)
points_3d = jax.random.uniform(key, (NUM_QUERY_POINTS, 3), minval=0, maxval=10)

# 3. Identity Metric for 3D Space
g_3d = jnp.eye(3)

# --- THE ENGINE PIPELINE ---
@jax.jit
def xagm_3d_engine(g, line, points):
    # Use your robust vmap logic
    fast_xpldist = jax.jit(jax.vmap(tpg.pldist, in_axes=(None, None, 0)))
    return fast_xpldist(g, line, points)

print(f"🚀 XAGM 3D FINAL BOSS: {NUM_QUERY_POINTS} pts vs {NUM_LINE_POINTS} 3D segments")

# Warmup (JIT compilation for 3D)
_ = xagm_3d_engine(g_3d, line_3d, points_3d[:5])

# Benchmark
start = time.time()
results = xagm_3d_engine(g_3d, line_3d, points_3d)
results.block_until_ready()
end = time.time()

# Stats
total_ops = NUM_QUERY_POINTS * (NUM_LINE_POINTS - 1)
throughput = total_ops / (end - start) / 1e6

print(f"\n📊 3D RESULTS:")
print(f"Total time: {end - start:.4f} seconds")
print(f"Throughput: {throughput:.2f} Million ops/sec")