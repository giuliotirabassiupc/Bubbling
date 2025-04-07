import numpy as np
from scipy.integrate import odeint
import tqdm
import numba
import gc
import matplotlib.pyplot as plt

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
    N = 200000
    dt = 0.005
    Ntrans = 50000

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

    for i, s in enumerate([0, 0.074, 0.2]):
        les, traj = compute_lyapunov_spectrum(rossler_tangent, u0, N, dt, (a, b, c, s))

        if s == 0:
            bins = np.logspace(-3, 1.6, 100)
            bins = np.sort(np.concatenate((bins, -bins)))
            axes[0].hist(
                les[0, :], bins=bins, log=True, density=True, color="tab:purple"
            )
            axes[0].set_xlabel(r"$\lambda^{\perp}_i$", fontsize=14)
            axes[0].set_xscale("symlog", linthresh=1e-2)
            axes[0].set_xticks([-100, -1, -0.01, 0.01, 1, 100])
            axes[0].grid(ls=":", c="grey")
            lam = les[0, Ntrans:].mean()
            axes[0].set_title(
                r"$\sigma="
                + f"{s:0.3f}$   "
                + r"$\lambda^{\perp}_{max}="
                + f"{lam:0.3f}$",
                fontsize=10,
            )

        i += 1
        ax = axes[i]
        les = les[:, Ntrans:]
        traj = traj[:, Ntrans:]
        print(s, les.mean(axis=1))

        lmin = -10
        lmax = 10

        counts, bins = np.histogram(les[0, :], bins=np.linspace(lmin / 10, 0, 100))
        dbin = bins[1] - bins[0]
        counts = counts / (les[0, :].size * dbin)
        ax.hist(bins[:-1], bins, weights=counts, color="tab:blue")

        counts, bins = np.histogram(les[0, :], bins=np.linspace(0, lmax / 10, 100))
        dbin = bins[1] - bins[0]
        counts = counts / (les[0, :].size * dbin)
        ax.hist(bins[:-1], bins, weights=counts, color="tab:red")

        ax.grid(ls=":", c="grey")
        ax.set_yscale("log")
        ax.set_xlim([lmin / 10, lmax / 10])
        lam = les[0, :].mean()
        if round(lam, ndigits=3) == 0:
            lam = 0
        ax.set_xlabel(r"$\lambda^{\perp}_i$", fontsize=14)
        ax.set_title(
            r"$\sigma=" + f"{s:0.3f}$   " + r"$\lambda^{\perp}_{max}=" + f"{lam:0.3f}$",
            fontsize=10,
        )
    for i, ax in enumerate(axes):
        ax.text(
            x=-0.15,
            y=1.1,
            s="abcd"[i] + ")",
            fontsize=16,
            horizontalalignment="left",
            transform=ax.transAxes,
        )
    axes[0].set_ylabel("Density", fontsize=14)
    fig.tight_layout()
    plt.savefig("figures/local_lyapunov_exponent.png", dpi=300)
    plt.savefig("figures/local_lyapunov_exponent.pdf")
    plt.savefig("figures/local_lyapunov_exponent.eps")
    plt.show()
