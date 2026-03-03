import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from basis import metrics as mtc
import geoutils as us # Assuming us.eps is defined here (e.g., 1e-15)



# --- RUNNING THE TEST ---
# Point doesn't matter since the scale is constant
def fold_map(x):
    u, v = x[0], x[1]
    # At u=0, the entire 'v' dimension vanishes from the output
    return jnp.array([u, u*v, u*v**2])

pos_fold = jnp.array([0.0, 1.0]) # The 'Singular' point
g_fold = mtc.metmap(fold_map, pos_fold)
val_fold, grad_fold = mtc.det(g_fold, jnp.eye(2)) # Pass a real dg to test grad

print(f"\nFolded Singularity Metric (at u=0):\n{g_fold}")
print(f"Folded Value: {val_fold} (Should be 0.0)")
print(f"Folded Grad:  {grad_fold} (Should be 0.0 due to safety)")
