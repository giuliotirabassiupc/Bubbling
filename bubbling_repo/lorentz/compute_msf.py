from utils import lorentz, lorentz_tangent, compute_lyapunov_spectrum
from scipy.integrate import odeint
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import tqdm
import gc
import os
import pandas as pd

gc.enable()


if __name__ == "__main__":
    b = 8 / 3
    r = 28.0
    s = 10.0

    system = lorentz
    tangent_system = lorentz_tangent

    u0 = odeint(
        system,
        y0=[1.0, 1.0, 0.0],
        t=(0, 1000),
        args=(b, r, s),
        tfirst=True,
        rtol=1e-10,
        atol=1e-10,
    )[-1, :]
    dt = 0.005
    Ntrans = 50000
    N = int(1500 / dt) + Ntrans

    sigmas = np.linspace(0, 15, 100)

    with ProcessPoolExecutor(os.cpu_count()) as executor:
        futs = {}
        for k in sigmas:
            futs[k] = executor.submit(
                compute_lyapunov_spectrum, lorentz_tangent, u0, N, dt, (b, r, s, k)
            )
        less = []
        for k in sigmas:
            les, _ = futs[k].result()
            les = les[:, Ntrans:]
            less.append(les)
        less = np.stack(less)

    mean_les = less.mean(axis=-1)

    critical_k = sigmas[np.argmin(np.abs(mean_les[:, 0]))]
    les, traj = futs[critical_k].result()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    mpbl = ax.scatter(
        traj[0, Ntrans:].T,
        traj[1, Ntrans:].T,
        traj[2, Ntrans:].T,
        c=les[0, Ntrans:],
        vmin=-11,
        vmax=11,
        cmap="RdBu_r",
        s=1
    )
    clb = plt.colorbar(mpbl, ax=ax, location="top", fraction=0.03, shrink=1.5, pad=0.0)
    clb.set_label(size=13, label=r"$\lambda^\perp_i$", labelpad=10)
    fig.tight_layout(w_pad=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    plt.savefig("attractor_dt=0.005.png", dpi=600)
    plt.clf()

    plt.plot(sigmas, mean_les[:, 0])
    plt.grid()
    plt.savefig("MSF_dt=0.005.png")
    plt.clf()

    def fast_moving_sum(a, n=3):
        ret = np.cumsum(a, dtype=float, axis=1)
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        return ret[:, n - 1 :]

    loggrowths = []
    windows = np.logspace(0, 5.12, 500)
    for w in tqdm.tqdm(windows):
        loggrowth = (fast_moving_sum(less[:, 0], int(w)) * dt).max(axis=1)
        loggrowths.append(loggrowth)
    loggrowths = np.stack(loggrowths)

    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    Amax = loggrowths.max(axis=0)
    idx = Amax < 10
    axes[0].plot(sigmas[idx], np.exp(Amax[idx]), "o", c="k", zorder=100)
    axes[0].set_ylabel(r"$\mathcal{A}^{max}$", fontsize=14)
    axes[0].set_yscale("log")
    timescales = windows[np.argmax(loggrowths, axis=0)] * dt
    axes[1].plot(sigmas, timescales, "o")
    axes[1].set_xlabel(r"$\sigma$", fontsize=14)
    axes[1].set_ylabel(r"$\tau_{\mathcal{A}^{max}}$", fontsize=14)
    axes[1].set_yscale("log")
    axes[1].set_ylim([1e-4, 1e4])

    fig.tight_layout()
    plt.savefig("gain_factor_dt=0.005.png")
    plt.clf()
    res = pd.DataFrame(dict(sigmas=sigmas, logAmax=loggrowths.max(axis=0), tau=timescales))
    res.to_csv("gain_factor_dt=0.005.csv", index=False)