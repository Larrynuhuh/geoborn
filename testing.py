import geoutils as us
import jax
import jax.numpy as jnp

from ops import vectors as vct

triangle = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])


result_normal = vct.normal(triangle)


print(f"Input Shape: {triangle.shape}")   # Should be (3, 3)
print(f"Output Normal: {result_normal}")  # Should be [0, 0, 1] or [0, 0, -1]
print(f"Output Shape: {result_normal.shape}")


batch_triangles = jnp.stack([triangle, triangle]) 
batch_normals = vct.vnormal(batch_triangles)

print(f"Batch Output Shape: {batch_normals.shape}")