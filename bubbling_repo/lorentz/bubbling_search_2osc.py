import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import numba

main_path = "/Users/giuliotirabassi/Documents/kalman_filter/rossler/"
adjacency_matrix_path = main_path + "Structure/Net_7.dat"


HET = 0.001


def err_synch(data):
    return np.mean(np.abs(data - data.mean(axis=0)), axis=0)


def max_single_err_synch(data):
    return np.abs(data - data.mean(axis=0)).max()


@numba.njit()
def lorentz_rhs(t, x, a, b, c, Kx, Ky, Kz, L):
    # a = sigma
    # b = beta
    # c = rho
    N = L.shape[0]
    N2 = 2 * N
    N3 = 3 * N
    xxdot = np.zeros(x.size)
    z = x[N2:N3]
    y = x[N:N2]
    x = x[:N]
    xdot = a * (y - x) - Kx * np.dot(L, x)
    ydot = x * (c - z) - y - Ky * np.dot(L, y)
    zdot = x * y - b * z - Kz * np.dot(L, z)
    xxdot[:N] = xdot
    xxdot[N:N2] = ydot
    xxdot[N2:N3] = zdot
    return xxdot


def simulate(K, adj, Ttrans=200, Ttrue=10000, Nsample=50000, rel_het=HET):
    N = adj.shape[0]
    Kx, Ky, Kz = K
    a, b, c = 10.0, 8 / 3, 28.0

    n = int(round(Nsample * Ttrans / Ttrue))
    hhet = np.linspace(1 - rel_het / 2, 1 + rel_het / 2, N)
    aa = a * hhet
    bb = b * hhet
    cc = c * hhet
    L = np.diag(adj.sum(axis=0)) - adj
    initial_spread = 0.1
    sol = odeint(
        lorentz_rhs,
        y0=np.array([1, 1, 1] * N)
        + 2 * initial_spread * np.random.rand(N * 3)
        - initial_spread,
        t=np.linspace(0, Ttrans + Ttrue, Nsample + n),
        rtol=1e-9,
        atol=1e-9,
        args=(aa, bb, cc, Kx, Ky, Kz, L.astype(float)),
        tfirst=True,
    ).T[:, -n:]

    x = sol[:N, :]
    # y = sol[28 : 28 * 2, :]
    # z = sol[28 * 2 : 28 * 3, :]

    return x


def compute(K, adj):
    res = []
    for _ in range(100):
        x = simulate(K, adj)
        errsynch = err_synch(x)
        res.append([np.median(errsynch), errsynch.max(), max_single_err_synch(x)])
    res = np.array(res)
    return res.mean(axis=0), res.std(axis=0)


if __name__ == "__main__":

    adj = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )

    adj = np.array([[0, 1], [1, 0]])

    L = np.diag(adj.sum(axis=0)) - adj
    es = np.linalg.eigvals(L)
    lam = sorted(es)[1]

    sigmas = np.linspace(0.1, 8, 40)
    Ks = sigmas / lam

    futures = {}
    esynchs = {}
    rs = {}
    conditions = {}
    with ProcessPoolExecutor(os.cpu_count()) as executor:
        for K in Ks:
            futures[K] = executor.submit(compute, [K, K, K], adj)
        print("jobs submitted...")
        for k, f in futures.items():
            es = f.result()
            esynchs[k] = es
            print(len(esynchs), "/", len(futures))

    med_esynch = [esynchs[k][0][0] for k in Ks]
    max_esynch = [esynchs[k][0][1] for k in Ks]
    max_single_esynch = [esynchs[k][0][2] for k in Ks]

    e_med_esynch = [esynchs[k][1][0] for k in Ks]
    e_max_esynch = [esynchs[k][1][1] for k in Ks]
    e_max_single_esynch = [esynchs[k][1][2] for k in Ks]

    plt.errorbar(
        Ks * lam,
        max_esynch,
        yerr=e_max_esynch,
        label="Max",
        fmt="o",
        elinewidth=2,
        capsize=2,
    )
    plt.errorbar(
        Ks * lam,
        med_esynch,
        yerr=e_med_esynch,
        label="Median",
        fmt="o",
        elinewidth=2,
        capsize=2,
    )
    plt.yscale("log")
    plt.legend()
    plt.ylabel("Err. Synch.")
    plt.xlabel("K")
    plt.grid(ls=":", c="grey")
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(
        {
            "sigma": Ks * lam,
            "K": Ks,
            "MaxSynchErr": max_esynch,
            "MaxSynchErrSTD": e_max_esynch,
            "MedianSynchErr": med_esynch,
            "MedianSynchErrSTD": e_med_esynch,
        }
    ).to_csv(f"lorentz_err_synch_het={HET}_N=2.csv")
