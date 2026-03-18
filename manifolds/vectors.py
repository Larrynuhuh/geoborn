
import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

from basis import metrics as mtc

@jax.jit
def nrm(g: Matrix, jacobian: Matrix) -> Matrix:
    
    n, k = jacobian.shape
    # 1. Euclidean QR (The "Guess")
    q, r = jnp.linalg.qr(jacobian, mode='complete')
    tangents = q[:, :k]
    raw_normals = q[:, k:] # This is what failed the test!

    # 2. THE RIEMANNIAN FIX (Metric-Projection)
    # We find the part of raw_normals that 'leaks' into the tangent space
    # in the warped metric g, and subtract it.
    gram = tangents.T @ g @ tangents
    # Solve: tangents.T @ g @ (raw_normals - tangents @ proj) = 0
    proj = jnp.linalg.solve(gram, tangents.T @ g @ raw_normals)
    
    # 3. The True Normal (Orthogonal in metric g)
    true_normals = raw_normals - (tangents @ proj)
    
    # 4. Final Unitize using your mtc.norm
    # This ensures the Riemannian Norm is exactly 1.0
    unit_n = true_normals / mtc.norm(g, true_normals)
    
    return tangents, unit_n

#dot product territory
@jax.jit
def scalproj(g: Matrix, a: Vector, b: Vector) -> Scalar: 
    
    norm = mtc.norm(g, b)
    prod = us.div(mtc.iprod(g, a, b), norm)

    return prod

@jax.jit
def xscalproj(g: Matrix, a: Matrix, b: Matrix) -> Vector:
    return jax.vmap(scalproj, in_axes = (None, 0, 0))(g, a, b)

@jax.jit
def vectproj(g: Matrix, a: Vector, b: Vector) -> Vector:

    term = mtc.iprod(g, b, b)
    prod = us.div(mtc.iprod(g, a, b), term)
    proj = prod * b

    return proj

@jax.jit
def xvectproj(g: Matrix, a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(vectproj, in_axes = (None, 0, 0))(g, a, b)

@jax.jit
def rejvect(g: Matrix, a: Vector, b: Vector) -> Vector:

    proj = vectproj(g, a, b)
    reject = a - proj

    return reject

@jax.jit
def xrejvect(g: Matrix, a: Matrix, b: Matrix) -> Matrix:
    return jax.vmap(rejvect, in_axes = (None, 0, 0))(g, a, b)

@jax.jit
def unitize(g: Matrix, u: Vector) -> Vector: 
    return us.div(u, mtc.norm(g, u))

@jax.jit
def xunitize(g: Matrix, u: Matrix) -> Matrix: 
    return jax.vmap(unitize, in_axes=(None, 0))(g, u)