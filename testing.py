import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import calc as calc
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray
import geoutils as us
from numba import njit, prange
import time

def polar_transform(v):
    r, theta = v[0], v[1]
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

# Let's test at r = 3.0, theta = pi/4
r_val = 3.0
test_x = jnp.array([r_val, jnp.pi/4])

gamma = calc.christoffel(polar_transform, test_x)

# --- 3. Results Verification ---
print(f"--- Testing Polar Metrics at r = {r_val} ---")
print(f"Gamma^r_theta_theta (Expected {-r_val}): {gamma[0, 1, 1]:.4f}")
print(f"Gamma^theta_r_theta (Expected {1/r_val:.4f}): {gamma[1, 0, 1]:.4f}")

# Verification logic
is_correct = jnp.allclose(gamma[0, 1, 1], -r_val) and jnp.allclose(gamma[1, 0, 1], 1/r_val)
print(f"\nAccuracy Test Passed: {is_correct}")