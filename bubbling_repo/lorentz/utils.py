import numba
import numpy as np
import tqdm


@numba.njit()
def lorentz(t, x, b, r, s):
    """Rössler system equations"""
    x1, y1, z1 = x
    return np.array([s * (y1 - x1), x1 * (r - z1) - y1, x1 * y1 - b * z1])


@numba.njit()
def lorentz_jac(x, b, r, s, k):
    xp, yp, zp = x
    return np.array(
        [
            [-s - k, s, 0],
            [r - zp, -1 - k, -xp],
            [yp, xp, -b - k],
        ]
    )


@numba.njit()
def lorentz_tangent(t, x, a, b, c, s):
    """Rössler system equations"""
    X = x[0:3]
    base = x[3:].reshape((3, 3))
    jac = np.dot(lorentz_jac(X, a, b, c, s), base).flatten()
    dxdt = lorentz(t, X, a, b, c)

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
    for i in range(N):
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
