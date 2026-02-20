import jax
import jax.numpy as jnp

@jax.jit
def div(a, b):
    safe = b != 0
    den = jnp.where(safe, b, 1.0)

    return jnp.where(safe, a/den, 0.0)


eps = 1e-15