import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numba
from local_lyapunov_exponents import rossler
import pickle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]

main_path = "/Users/giuliotirabassi/Documents/kalman_filter/rossler/"
adjacency_matrix_path = main_path + "Structure/Net_7.dat"


def err_synch(data):
    return np.mean(np.abs(data - data.mean(axis=0)), axis=0)


def max_single_err_synch(data):
    return np.abs(data - data.mean(axis=0)).max()


@numba.njit()
def rossler_rhs(t, x, a, b, c, Kx, Ky, Kz, L):
    N = L.shape[0]
    N2 = 2 * N
    N3 = 3 * N
    xxdot = np.zeros(x.size)
    z = x[N2:N3]
    y = x[N:N2]
    x = x[:N]
    xdot = -(y + z) - Kx * np.dot(L, x)
    ydot = x + a * y - Ky * np.dot(L, y)
    zdot = b + z * (x - c) - Kz * np.dot(L, z)
    xxdot[:N] = xdot
    xxdot[N:N2] = ydot
    xxdot[N2:N3] = zdot
    return xxdot


def simulate(K, adj, Ttrans=2000, Ttrue=10000, Nsample=50000):
    random = np.random.RandomState(0)
    N = adj.shape[0]
    Kx, Ky, Kz = K
    a, b, c = 0.2, 0.2, 7  # 0.42, 2.0, 4.0
    n = int(Nsample * Ttrue / (Ttrans + Ttrue))
    rel_inhomogeneity = 0.01
    aa = a * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    bb = b * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    cc = c * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    L = np.diag(adj.sum(axis=0)) - adj
    initial_spread = 0.1
    sol = odeint(
        rossler_rhs,
        y0=np.array([1, 1, 0] * N)
        + 2 * initial_spread * random.rand(N * 3)
        - initial_spread,
        t=np.linspace(0, Ttrans + Ttrue, Nsample),
        rtol=1e-9,
        atol=1e-9,
        args=(aa, bb, cc, Kx, Ky, Kz, L.astype(float)),
        tfirst=True,
    ).T[:, -n:]

    x = sol[:N, :]
    y = sol[28 : 28 * 2, :]
    z = sol[28 * 2 : 28 * 3, :]

    return x, y, z


if __name__ == "__main__":
    a, b, c = 0.2, 0.2, 7.0
    with open("results/results_for_storyboard.pkl", "rb") as f:
        x, y, z, err, imin, imax, eslong = pickle.load(f)

    orbits = np.loadtxt(f"../Rossler_orbits_({a}, {b}, {c}).csv")

    _, u0x, u0y, u0z, T = orbits[0]
    dt = 100 * T / 100000

    tlong = np.arange(eslong[0][1].size) * dt
    tmin = tlong[imin]

    print("Computing orbits...")
    orbit_paths = [
        odeint(
            rossler,
            y0=orbit[1:4],
            t=np.linspace(0, orbit[4], 1000),
            atol=1e-10,
            rtol=1e-10,
            tfirst=True,
            args=(a, b, c),
        )
        for orbit in orbits
    ]

    xavg = x.mean(axis=0)
    yavg = y.mean(axis=0)

    orbits_ids = [9, 10]

    fig = plt.figure(figsize=(5, 6))

    ois = [9, 10]
    i2, i12, i1 = 34000, 39000, 45000
    es = err_synch(x)
    xavg = x.mean(axis=0)
    yavg = y.mean(axis=0)

    ax = fig.add_subplot(2, 2, (1, 2))
    t = tmin + np.arange(i2, i1) * dt
    ax.plot(t, es[i2:i1])
    plt.plot([t[i2 - i2], t[i12 - i2 - 1]], [es[i2:i12].max() * 2] * 2, lw=4, c="k")
    plt.plot([t[i12 - i2], t[i1 - i2 - 1]], [es[i12:i1].max() * 1.1] * 2, lw=4, c="k")
    ax.text(
        t[: i12 - i2].mean(),
        es[i2:i12].max() * 2 + 0.1,
        "b)",
        fontsize=14,
    )
    ax.text(
        t[i12 - i2 :].mean(),
        es[i12:i1].max() * 1.1 + 0.1,
        "c)",
        fontsize=14,
    )
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], 1.2 * ylim[1]])
    ax.set_ylabel("SE", fontsize=12)
    ax.text(-0.095, 1.1, "a)", fontsize=16, transform=ax.transAxes)
    ax.set_xlabel(r"Time", fontsize=12)
    ax.grid(ls=":")

    for jj, oi in enumerate(ois):
        ax = fig.add_subplot(2, 2, 3 + jj)
        I2, I1 = [i2, i12, i1][jj : jj + 2]
        xsec, ysec = xavg[I2:I1], yavg[I2:I1]
        ax.plot(xsec, ysec, c="k", lw=2)
        arrows_idx = np.linspace(1, I1 - I2 - 1, 5).astype(int)
        dx = xsec[arrows_idx] - xsec[arrows_idx - 1]
        dy = ysec[arrows_idx] - ysec[arrows_idx - 1]
        for j, aidx in enumerate(arrows_idx):
            ax.arrow(
                xsec[aidx],
                ysec[aidx],
                dx[j],
                dy[j],
                shape="full",
                lw=0,
                length_includes_head=True,
                head_width=0.75,
                color="k",
            )
        ax.plot(
            orbit_paths[oi][:, 0], orbit_paths[oi][:, 1], ls="--", lw=1, c="tab:red"
        )
        ax.set_xlabel(r"$\bar{x}$", fontsize=12)
        ax.set_ylabel(r"$\bar{y}$", fontsize=12, labelpad=-5)
        ax.grid(ls=":")
        ax.text(
            -0.2,
            1.05,
            "bc"[jj] + ")",
            fontsize=16,
            transform=ax.transAxes,
        )
    fig.tight_layout()
    # plt.savefig("figures/transient_growths_2.png", dpi=300)
    plt.savefig("figures/transient_growths_3.eps")
    plt.show()
