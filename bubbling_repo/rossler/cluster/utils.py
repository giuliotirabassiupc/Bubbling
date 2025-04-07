import numba
import numpy as np
import tqdm


@numba.njit()
def rossler(t, x, a, b, c):
    """Rössler system equations"""
    x1, y1, z1 = x
    return np.array([-y1 - z1, x1 + a * y1, b + z1 * (x1 - c)])


@numba.njit()
def rossler_transverse(t, x, a, b, c, s):
    """Rössler system equations"""
    xp, yp, zp, xt, yt, zt = x
    return np.array(
        [
            -yp - zp,
            xp + a * yp,
            b + zp * (xp - c),
            -yt - zt - s * xt,
            xt + a * yt - s * yt,
            zt * (xp - c - s) + zp * xt,
        ]
    )


@numba.njit()
def rossler_jac(x, a, b, c, s):
    xp, yp, zp = x
    return np.array(
        [
            [-s, -1.0, -1.0],
            [1.0, a - s, 0.0],
            [zp, 0.0, xp - c - s],
        ]
    )


@numba.njit()
def rossler_tangent(t, x, a, b, c, s):
    """Rössler system equations"""
    X = x[0:3]
    base = x[3:].reshape((3, 3))
    jac = np.dot(rossler_jac(X, a, b, c, s), base).flatten()
    dxdt = rossler(t, X, a, b, c)

    return np.concatenate((dxdt, jac))


@numba.njit()
def step_RK4(x, f, f_args, dt):
    k1 = f(None, x, *f_args)
    k2 = f(None, x + 0.5 * dt * k1, *f_args)
    k3 = f(None, x + 0.5 * dt * k2, *f_args)
    k4 = f(None, x + dt * k3, *f_args)
    dxdt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x + dxdt * dt


@numba.njit()
def integrate(f, x0, dt, N, args=None):
    for _ in range(N):
        x0 = step_RK4(x0, f, args, dt)
    return x0


def compute_lyapunov_spectrum(tangent_system, u0, N, dt, args):
    orth_base = np.identity(u0.size)
    les = np.zeros((u0.size, N))
    traj = np.zeros((u0.size, N))
    x0 = u0
    for i in tqdm.tqdm(range(N)):
        xnew = step_RK4(
            np.concatenate((x0, orth_base.flatten())),
            tangent_system,
            args,
            dt,
        )
        new_base = xnew[x0.size :].reshape((x0.size, x0.size))
        orth_base, R = np.linalg.qr(new_base)
        norms = np.abs(np.diag(R))
        if np.any(norms == 0):
            break
        les[:, i] = np.log(norms)
        traj[:, i] = x0
        x0 = xnew[: x0.size]
    return les / dt, traj
