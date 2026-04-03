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

def torus_mapping(params):
    u, v = params
    R, r = 2.0, 0.5
    return jnp.array([
        (R + r * jnp.cos(u)) * jnp.cos(v),
        (R + r * jnp.cos(u)) * jnp.sin(v),
        r * jnp.sin(u)
    ])


def sphere_mapping(params):
    u, v = params
    return jnp.array([jnp.sin(u) * jnp.cos(v), jnp.sin(u) * jnp.sin(v), jnp.cos(u)])


def run_final_comparison(p, v_true, mapping):
    # JIT the solvers
    fast_exp = jax.jit(calc.geoexp_solver, static_argnums=(2,))
    fast_log = jax.jit(calc.geolog_solver, static_argnums=(2, 3))

    # --- Warmup (Compilation) ---
    print("Compiling XLA Graphs...")
    q_target, _ = fast_exp(p, v_true, mapping)
    _ = fast_log(p, q_target, mapping, 5)
    
    # --- Benchmark Exponential Map (The Shot) ---
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        # We need the pos and vel for the drift check
        pos, vel = fast_exp(p, v_true, mapping)
        pos.block_until_ready()
    t_exp = (time.perf_counter() - start) / iters * 1000

    # Accuracy check for ExpMap (Drift)
    g_start = mtc.fwdmet(mapping, p)
    g_end = mtc.fwdmet(mapping, pos)
    len_0 = jnp.sqrt(jnp.dot(v_true, jnp.dot(g_start, v_true)))
    len_1 = jnp.sqrt(jnp.dot(vel, jnp.dot(g_end, vel)))
    drift = jnp.abs(len_0 - len_1)

    # --- Benchmark Logarithmic Map (The Find) ---
    start = time.perf_counter()
    for _ in range(iters):
        v_found = fast_log(p, q_target, mapping, 5)
        v_found.block_until_ready()
    t_log = (time.perf_counter() - start) / iters * 1000

    # Accuracy check for LogMap (Reconstruction)
    log_error = jnp.linalg.norm(v_found - v_true)

    # --- 2. THE RESULTS ---
    print(f"\n--- XAGM PERFORMANCE AUDIT ---")
    print(f"Surface: Torus (Variable Curvature)")
    
    print(f"\n[EXPONENTIAL MAP]")
    print(f"Runtime:  {t_exp:.4f} ms")
    print(f"Accuracy: {drift:.2e} (Metric Drift)")

    print(f"\n[LOGARITHMIC MAP]")
    print(f"Runtime:  {t_log:.4f} ms")
    print(f"Accuracy: {log_error:.2e} (Vector Error)")

    print(f"\n[EFFICIENCY RATIO]")
    # How many ExpMaps fit into one LogMap?
    ratio = t_log / t_exp
    print(f"1 LogMap ≈ {ratio:.1f} ExpMaps")
    print(f"Theoretical Cost: ~15x (5 Newton steps * (1 shoot + 2 partials))")
    print(f"Observed geodesic distance: {calc.geodist(p, q_target, mapping, 5):.4f}")

# Run it
p_test = jnp.array([0.1, 0.1])
v_test = jnp.array([0.4, 0.2])
run_final_comparison(p_test, v_test, torus_mapping)