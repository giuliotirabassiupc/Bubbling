import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import numba
import gc
from matplotlib import animation
import pickle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]


gc.enable()


@numba.njit()
def rossler(t, x, a, b, c):
    x, y, z = x
    return np.array(
        [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ]
    )


def err_synch(data):
    return np.mean(np.abs(data - data.mean(axis=0)), axis=0)


main_path = "/Users/giuliotirabassi/Documents/kalman_filter/rossler/"
adjacency_matrix_path = main_path + "Structure/Net_7.dat"

A = 0.2
B = 0.2
C = 7.0
SIGMA = 0.1
Nrepl = 100
TTRUE = 100000
NSAMPLE = 5 * TTRUE


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


def simulate(K, adj, y0, rel_inhomogeneity, Ttrans=2000, Ttrue=TTRUE, Nsample=NSAMPLE):
    random = np.random.RandomState(0)
    N = adj.shape[0]
    Kx, Ky, Kz = K
    a, b, c = A, B, C
    n = int(Nsample * Ttrue / (Ttrans + Ttrue))
    aa = a * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    bb = b * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    cc = c * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    if N == 2:
        aa = a * np.array([1 + rel_inhomogeneity, 1 - rel_inhomogeneity])
        bb = b * np.array([1 + rel_inhomogeneity, 1 - rel_inhomogeneity])
        cc = c * np.array([1 + rel_inhomogeneity, 1 - rel_inhomogeneity])
    L = np.diag(adj.sum(axis=0)) - adj
    initial_spread = 0.01
    if y0.size != 3 * N:
        y0 = (
            np.array([y0[0]] * N + [y0[1]] * N + [y0[2]] * N)
            + 2 * initial_spread * random.rand(N * 3)
            - initial_spread
        )

    sol = odeint(
        rossler_rhs,
        y0=y0,
        t=np.linspace(0, Ttrans + Ttrue, Nsample),
        rtol=1e-9,
        atol=1e-9,
        args=(aa, bb, cc, Kx, Ky, Kz, L.astype(float)),
        tfirst=True,
    ).T[:, -n:]

    x = sol[:N, :]
    y = sol[N : N * 2, :]
    z = sol[N * 2 : N * 3, :]

    return x, y, z


if __name__ == "__main__":

    structure = pd.read_csv(adjacency_matrix_path, header=None, sep="\s+")

    adj = np.zeros((28, 28))
    for i, j in structure.values - 1:
        adj[i, j] = adj[j, i] = 1

    N = adj.shape[0]

    L = np.diag(adj.sum(axis=0)) - adj
    evals = np.linalg.eigvals(L)
    evals.sort()
    lam = np.real(evals[1])

    K = SIGMA / lam

    orbits = np.loadtxt(f"../Rossler_orbits_({A}, {B}, {C}).csv")
    _, u0x, u0y, u0z, T = orbits[0]
    orbit_phase = np.linspace(0, T, 100)
    u0 = np.array([u0x, u0y, u0z])
    orbit = odeint(
        rossler,
        y0=u0,
        t=orbit_phase,
        atol=1e-10,
        rtol=1e-10,
        tfirst=True,
        args=(A, B, C),
    )

    for o in orbit:
        NN = 1000000
        x, y, z = simulate([K, K, K], adj, o, 0.01, 0, 1000 * T, NN)
        dt = 100 * T / 100000
        es = err_synch(x)
        print(es.max())
        if es.max() > 0.4:
            break
    eslong = [(0, es)]
    y0 = np.concatenate([x[:, -1], y[:, -1], z[:, -1]])
    for i in range(30):
        xlong, ylong, zlong = simulate([K, K, K], adj, y0, 0.01, 0, 1000 * T, NN)
        newes = err_synch(xlong)
        if newes.max() > 0.4:
            print("bubbling found...")
            eslong.append((i, newes[1:]))
        y0 = np.concatenate([xlong[:, -1], ylong[:, -1], zlong[:, -1]])

    imin = 72000
    imax = 89000

    imin = 490000
    imax = 520000

    with open("results/results_for_storyboard.pkl", "wb") as f:
        pickle.dump(
            [
                x[:, imin - 30000 : imax],
                y[:, imin - 30000 : imax],
                z[:, imin - 30000 : imax],
                es[imin - 30000 : imax],
                imin - 30000,
                imax,
                eslong,
            ],
            f,
        )

    raise RuntimeError("Thanks but stop here")

    x = x[:, imin:imax]
    y = y[:, imin:imax]
    z = z[:, imin:imax]
    es = es[imin:imax]

    plt.plot(x.mean(axis=0) / 10)
    plt.plot(es)
    plt.show()

    def update_lines(ii, x, y, z, window, lines, esline, es, t):
        for i, line in enumerate(lines):
            data = np.stack(
                (x[i, ii : ii + window], y[i, ii : ii + window], z[i, ii : ii + window])
            )
            line.set_data_3d(data)
        esline.set_data(np.stack((t[: ii + window], es[: ii + window])))
        return lines

    window = 1000
    idx = np.arange(0, x.shape[1] - window, 5)

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 3, (1, 2), projection="3d")
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.grid(ls=":", c="grey")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Synchronization Error")
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], ls="--", c="k")
    # Create lines initially without data
    lines = [
        ax.plot(
            [],
            [],
            [],
            c="tab:red",
            alpha=0.5,
        )[0]
        for _ in range(x.shape[0])
    ]
    esline = ax2.plot([], [], c="tab:blue")[0]
    t = np.arange(x.shape[1]) * dt
    ax2.set_xlim([-10 * dt, t.max() + 10 * dt])
    ax2.set_ylim([-0.1, 1.1])

    # Setting the Axes properties
    ax.set(xlim3d=(-12, 12), xlabel="x")
    ax.set(ylim3d=(-12, 12), ylabel="y")
    ax.set(zlim3d=(-1, 25), zlabel="z")

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig,
        update_lines,
        idx[::4],
        fargs=(x, y, z, window, lines, esline, es, t),
        interval=1,
    )
    fig.tight_layout()
    ani.save(
        filename="figures/pillow_example.gif",
        writer="pillow",
        progress_callback=lambda i, n: print(i),
        fps=24,
    )
    plt.show()
