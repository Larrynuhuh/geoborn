import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
 
from numba import njit, prange
import time


@jax.jit
def weird_map(v):
    x, y, z = v
    r = jnp.sqrt(x**2 + y**2 + 1e-6)
    theta = jnp.arctan2(y, x) + r # A twist based on distance
    return jnp.array([
        r * jnp.cos(theta), 
        r * jnp.sin(theta), 
        z + 0.1 * r**2 # Parabolic stretching
    ])

# 2. Generate the Metric using YOUR fwdmet
p = jnp.array([1.0, 0.5, -0.2])
g_induced = mtc.fwdmet(weird_map, p)

# 3. Define a local Tangent Basis (a 2D plane)
J = jnp.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.2, 0.1]
])

# 4. Generate the Basis (Tangents + Normals)
# This uses the QR-based 'analytical_basis' we built
t_basis, n_basis = vct.nrm(g_induced, J)

# --- THE VERIFICATION ---
# Is the normal perpendicular to the tangents in the induced space?
err_1 = mtc.iprod(g_induced, t_basis[:, 0], n_basis[:, 0])
err_2 = mtc.iprod(g_induced, t_basis[:, 1], n_basis[:, 0])

# Is the normal a unit vector in the induced metric?
n_norm = mtc.norm(g_induced, n_basis[:, 0])

print(f"📐 Orthogonality Errors (Should be ~0): {err_1:.5e}, {err_2:.5e}")
print(f"📏 Riemannian Norm (Should be 1.0): {n_norm:.6f}")