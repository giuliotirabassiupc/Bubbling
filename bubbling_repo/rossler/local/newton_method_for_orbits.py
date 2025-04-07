import numpy as np
import numba
from scipy.integrate import odeint

# Practical numerical algorithms for chaotic systems parker and chua 2012


@numba.njit()
def step_RK4_function_jacobian(x, base, f, jac, f_args, j_args, dt):
    identity = np.eye(x.size)
    k1 = f(x, *f_args)
    k2 = f(x + 0.5 * dt * k1, *f_args)
    k3 = f(x + 0.5 * dt * k2, *f_args)
    k4 = f(x + dt * k3, *f_args)
    dxdt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    Jk1 = jac(x, *j_args)
    Jk2 = jac(x + 0.5 * dt * k1, *j_args).dot(identity + Jk1 * dt * 0.5)
    Jk3 = jac(x + 0.5 * dt * k2, *j_args).dot(identity + Jk2 * dt * 0.5)
    Jk4 = jac(x + dt * k3, *j_args).dot(identity + Jk3 * dt)
    J = (Jk1 + 2 * Jk2 + 2 * Jk3 + Jk4) / 6
    return x + dxdt * dt, base + J @ base * dt


def step_run(x0, map_period, f, f_jac, dt=0.01, args=None, j_args=None, damping=2):
    manifold = x0[0]
    n = len(x0)
    basenew = np.eye(n)
    xnew = x0
    windings = 0
    i = 0
    traj = [xnew]
    while windings < map_period:
        x = xnew
        base = basenew
        xnew, basenew = step_RK4_function_jacobian(x, base, f, f_jac, args, j_args, dt)
        traj.append(xnew)
        i += 1
        if xnew[0] > manifold and x[0] < manifold:
            windings += 1
        if i * dt > 1000:
            print("Simulation diverges")
            return x, 1, None, np.inf
    traj = np.array(traj)
    tstar = -dt * x[0] / (xnew[0] - x[0])
    period = (i - 1) * dt + tstar
    y = x + tstar * (xnew - x) / dt
    phiT = base + tstar * (basenew - base) / dt
    fy = f(y, *args).reshape((n, 1))
    h = np.array([[1], [0], [0]])
    DP = (np.eye(n) - fy.dot(h.T) / fy.T.dot(h)).dot(phiT)
    H = y - x0
    DH = DP - np.eye(n)
    err = np.linalg.norm(H)
    return x0 - (2**-damping) * np.linalg.inv(DH).dot(H), period, traj, err


def poincare_method(
    x0,
    map_period,
    f,
    f_jac,
    dt=0.01,
    args=None,
    j_args=None,
    damping=2,
    atol=1e-6,
    maxiter=1000,
):
    i = 0
    err = np.inf
    xk = x0
    message = None
    while err > atol:
        xkplusone, period, traj, err = step_run(
            xk, map_period, f, f_jac, dt, args, j_args, damping
        )
        err2 = np.linalg.norm(xk - xkplusone)
        # print(xkplusone, err, err2)
        xk = xkplusone
        i += 1
        if i > maxiter:
            message = "Failure"
            break
    if err > atol:
        message = "Failure"
    if not message:
        message = "Success"
    return xkplusone, period, traj, err, err2, i, message


def newton_method(
    x0,
    T0,
    f,
    f_jac,
    args=None,
    j_args=None,
    atol=1e-6,
    maxiter=1000,
):

    def F(t, x, n):
        x0 = x[0:n]
        base = x[n:].reshape((n, n))
        dxdt = f(x0, *args)
        dbdt = f_jac(x0, *j_args) @ base
        return np.concatenate((dxdt, dbdt.flatten()))

    n = x0.size

    niter = 0
    while niter < maxiter:
        X0 = np.concatenate((x0, np.eye(n).flatten()))
        XT = odeint(
            F,
            y0=X0,
            t=np.linspace(0, T0, int(T0 / 0.001)),
            rtol=1e-10,
            atol=1e-10,
            tfirst=True,
            args=(n,),
        )
        traj = XT[:, :n]
        xnew = XT[-1, 0:n]
        phiT = XT[-1, n:].reshape((n, n))
        H = x0 - xnew
        DH = np.vstack(
            [
                np.hstack([phiT - np.eye(n), f(xnew, *args).reshape(-1, 1)]),
                np.hstack([f(x0, *args), [0]]),
            ]
        )
        dX = np.linalg.inv(DH).dot(np.concatenate((H, [0])))
        x0 = x0 + dX[0:n]
        T0 = T0 + dX[-1]
        if np.linalg.norm(H) < atol:
            break
        niter += 1
        if np.linalg.norm(dX) > 100 or T0 < 0:
            print("The algorithm has not converged")
            break
        # print(np.linalg.norm(H), np.linalg.norm(dX))
    return x0, T0, np.linalg.norm(dX), traj, niter
