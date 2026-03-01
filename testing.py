import geoutils as us
import jax
import jax.numpy as jnp

# Check if your 64-bit utils flag actually kicked in
x = jnp.array([1.0])
print(f"Dtype: {x.dtype}") 
print(f"JAX Config x64: {jax.config.read('jax_enable_x64')}")

print(us.div(8, 0))