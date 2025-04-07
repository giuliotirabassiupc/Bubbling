import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import numba
from utils import lorentz


N = 2
TTRUE = 1500
TTRANS = 1000
NSAMPLE = 500000
HET = 0.001
NREPL = 500



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


def random_adjacency_matrix(size, edge_prob):
    if size == 2:
        return np.array([[0, 1], [1, 0]])
    random = np.random.RandomState(0)
    matrix = random.rand(size, size) < edge_prob
    np.fill_diagonal(matrix, 0)
    return np.triu(matrix) + np.triu(matrix, 1).T


def err_synch(data):
    return np.mean(np.abs(data - data.mean(axis=0)), axis=0)


def max_single_err_synch(data):
    return np.abs(data - data.mean(axis=0)).max()



def simulate(K, adj, rel_inhomogeneity=HET, Ttrans=TTRANS, Ttrue=TTRUE, Nsample=NSAMPLE):
    random = np.random
    N = adj.shape[0]
    L = np.diag(adj.sum(axis=0)) - adj
    Kx, Ky, Kz = K
    a, b, c = 10.0, 8 / 3, 28.0


    u0 = odeint(
        lorentz,
        y0=np.array([1.0, 1.0, 0.0]) + random.rand(3),
        t=(0, 1000),
        args=(b, c, a),
        tfirst=True,
        rtol=1e-10,
        atol=1e-10,
    )[-1, :]


    n = int(Nsample * Ttrue / (Ttrans + Ttrue))
    aa = a * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    bb = b * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    cc = c * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)

    if N == 2:
        aa = a * np.array([1 + rel_inhomogeneity/2, 1 - rel_inhomogeneity/2])
        bb = b * np.array([1 + rel_inhomogeneity/2, 1 - rel_inhomogeneity/2])
        cc = c * np.array([1 + rel_inhomogeneity/2, 1 - rel_inhomogeneity/2])

    initial_spread = 0
    sol = odeint(
        lorentz_rhs,
        y0=np.array(u0.tolist() * N)
        + 2 * initial_spread * np.random.rand(N * 3)
        - initial_spread,
        t=np.linspace(0, Ttrans + Ttrue, Nsample),
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
    for _ in range(NREPL):
        x = simulate(K, adj)
        errsynch = err_synch(x)
        res.append([np.median(errsynch), errsynch.max(), max_single_err_synch(x)])
    res = np.array(res)
    return res.mean(axis=0), res.std(axis=0)


if __name__ == "__main__":

    b = 8 / 3
    r = 28.0
    s = 10.0


    adj = random_adjacency_matrix(N, 0.1)

    L = np.diag(adj.sum(axis=0)) - adj
    es = np.linalg.eigvals(L)
    lam = sorted(es)[1]

    sigmas = np.linspace(0.1, 15, 40)
    Ks = sigmas / lam

    futures = {}
    esynchs = {}
    rs = {}
    conditions = {}
    with ProcessPoolExecutor(len(sigmas)) as executor:
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
        sigmas, max_esynch, yerr=e_max_esynch, label="Max", fmt="o", elinewidth=2, capsize=2
    )
    plt.errorbar(
        sigmas,
        med_esynch,
        yerr=e_med_esynch,
        label="Median",
        fmt="o",
        elinewidth=2,
        capsize=2,
    )

    plt.legend()
    plt.ylabel("Err. Synch.")
    plt.xlabel("K")
    # plt.yscale("log")
    plt.grid(ls=":", c="grey")
    plt.tight_layout()
    plt.savefig(f"max_se_{N}_{HET}.png")
    plt.clf()

    df = pd.DataFrame(dict(sigma=sigmas, K=Ks, masSE=max_esynch, masSEstd=e_max_esynch))
    df.to_csv(f"max_se_{N}_{HET}.csv", index=False)
