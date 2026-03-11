import jax
import jax.numpy as jnp
import numpy as np

from basis import metrics as mtc
from basis import linear as lin
from manifolds import vectors as vct
from manifolds import topology as tpg
 
from numba import njit, prange
import time


def randomized_svd(A, rank, oversample=5):
    """
    Computes a truncated SVD using a randomized algorithm.
    """
    m, n = A.shape
    # 1. Generate a random Gaussian matrix
    P = np.random.randn(n, rank + oversample)
    
    # 2. Form the sample matrix Z and orthonormalize it
    Z = A @ P
    Q, _ = np.linalg.qr(Z)
    
    # 3. Project A into the smaller subspace
    Y = Q.T @ A
    
    # 4. Perform standard SVD on the much smaller matrix Y
    U_tilde, Sigma, Vt = np.linalg.svd(Y, full_matrices=False)
    
    # 5. Project the left singular vectors back to the original space
    U = Q @ U_tilde
    
    return U[:, :rank], Sigma[:rank], Vt[:rank, :]

# Setup: Large noisy dataset (e.g., 5000 points in 500 dimensions)
rows, cols = 5000, 500
target_rank = 10 
data = np.random.randn(rows, cols)

# Benchmark Standard SVD
start = time.time()
U_full, S_full, V_full = np.linalg.svd(data, full_matrices=False)
t_full = time.time() - start

# Benchmark Randomized SVD
start = time.time()
U_rand, S_rand, V_rand = randomized_svd(data, target_rank)
t_rand = time.time() - start

# Accuracy check: Comparing the first singular value
error = np.abs(S_full[0] - S_rand[0]) / S_full[0]

print(f"Standard SVD Time: {t_full:.4f}s")
print(f"Randomized SVD Time: {t_rand:.4f}s")
print(f"Speedup: {t_full / t_rand:.2f}x")
print(f"Relative Error (Top Singular Value): {error:.2e}")