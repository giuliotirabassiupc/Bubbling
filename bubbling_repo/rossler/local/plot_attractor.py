import numpy as np
from scipy.integrate import odeint
import tqdm
import numba
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]

gc.enable()


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


if __name__ == "__main__":
    a = 0.2
    b = 0.2
    c = 7.0

    system = rossler
    tangent_system = rossler_tangent

    u0 = odeint(
        system,
        y0=[1.0, 1.0, 0.0],
        t=(0, 1000),
        args=(a, b, c),
        tfirst=True,
        rtol=1e-10,
        atol=1e-10,
    )[-1, :]
    N = 1000000
    dt = 0.005
    Ntrans = 50000

    s = 0.075
    les, traj = compute_lyapunov_spectrum(rossler_tangent, u0, N, dt, (a, b, c, s))
    les = les[:, Ntrans:]
    traj = traj[:, Ntrans:]
    print(s, les.mean(axis=1))

    # plot all attractor
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=20, azim=-80, roll=0)

    x, y, z = traj

    mpbl = ax.scatter(
        x,
        y,
        z,
        c=les[0, :],
        s=1,
        cmap="coolwarm",
        norm=mpl.colors.SymLogNorm(linthresh=0.1),
    )
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$y$", fontsize=15)
    ax.set_zlabel(r"$z$", fontsize=15)
    clb = plt.colorbar(mpbl, ax=ax, location="top", fraction=0.03, shrink=1.5, pad=0.0)
    clb.set_label(size=13, label=r"$\lambda^\perp_i$", labelpad=10)
    fig.tight_layout(w_pad=0)
    plt.savefig("figures/attractor_symlog.png", dpi=600)
    plt.show()

    # plot closeup

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=20, azim=-80, roll=0)

    x, y, z = traj
    idx = (x > -15) & (x < 15) & (y > -5) & (y < 5) & (z > 20)

    mpbl = ax.scatter(
        x[idx],
        y[idx],
        z[idx],
        c=les[0, idx],
        s=1,
        cmap="coolwarm",
    )
    ax.set_xlabel(r"$x$", fontsize=15)
    ax.set_ylabel(r"$y$", fontsize=15)
    ax.set_zlabel(r"$z$", fontsize=15)
    plt.colorbar(
        mpbl,
        ax=ax,
        location="left",
        fraction=0.02,
        pad=0,
        label=r"$\lambda^\perp_i$",
    )
    ax.set_xlim([1, 13])
    ax.set_zlim([20, 32])

    fig.tight_layout()
    plt.savefig("figures/attractor_closeup.png", dpi=600)
    plt.show()
