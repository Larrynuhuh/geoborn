import geoutils as us
import jax 
import jax.numpy as jnp
from geoutils import Vector, Matrix, Scalar, Tensor, JAXArray

from basis import metrics as mtc

def christoffel(func, x: Vector) -> Matrix:
    
    g = mtc.fwdmet(func, x)
    ginv = mtc.metinv(g)
    mtc_func = lambda v: mtc.fwdmet(func, v)

    __,dg_raw = jax.vmap(lambda v: jax.jvp(mtc_func, (x,), (v,)))(jnp.eye(x.shape[0]))

    dg = jnp.moveaxis(dg_raw, 0, -1)

    term1 = jnp.transpose(dg, axes=[1, 2, 0])
    term2 = jnp.transpose(dg, axes=[0, 1, 2])
    term3 = jnp.transpose(dg, axes=[2, 0, 1])

    contract1 = 0.5 * ginv
    contract2 = term1 + term2 - term3
    gamma = jnp.einsum('kl, lij -> kij', contract1, contract2)

    return gamma

import diffrax


def geoexp_term(t, state, args) -> Vector:
    dim = state.shape[0] // 2
    x = state[:dim] 
    v = state[dim:]
    func = args['func']
    gamma = christoffel(func, x)

    v_dot = -jnp.einsum('kij, i, j -> k', gamma, v, v)

    return jnp.concatenate([v, v_dot])


def geoexp_solver(p: Vector, v: Vector, mapped_func) -> Vector:
    state = jnp.concatenate([p, v])

    solution = diffrax.diffeqsolve(
        terms = diffrax.ODETerm(geoexp_term),
        solver = diffrax.Tsit5(),
        t0=0,
        t1=1,
        dt0=1e-3,
        y0=state,
        args = {'func': mapped_func},
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True)
    )

    result = solution.ys[0]

    dim = p.shape[0]
    final_pos = result[:dim]
    final_vel = result[dim:]

    return final_pos, final_vel

