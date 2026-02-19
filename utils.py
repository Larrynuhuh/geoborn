import jax
import jax.numpy as jnp

def div(a, b):
    safe = denominator != 0
    den = jnp.where(safe, b, 1.0)

    return jnp.where(safe, a/den, 0.0)