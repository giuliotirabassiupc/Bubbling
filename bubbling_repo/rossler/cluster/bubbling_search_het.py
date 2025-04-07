import numpy as np
from scipy.integrate import odeint
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import numba
import gc
import tqdm

gc.enable()


main_path = "/home/users/giulio.tirabassi/rosslers/"
adjacency_matrix_path = main_path + "Structure/Net_7.dat"

A = 0.2
B = 0.2
C = 7.0
SIGMA = 0.1
Nrepl = 100
TTRUE = 100000
NSAMPLE = 5 * TTRUE


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


def simulate(K, adj, rel_inhomogeneity, Ttrans=2000, Ttrue=TTRUE, Nsample=NSAMPLE):
    N = adj.shape[0]
    Kx, Ky, Kz = K
    a, b, c = A, B, C
    n = int(Nsample * Ttrue / (Ttrans + Ttrue))
    aa = a * (1 + rel_inhomogeneity * np.random.rand(N) - rel_inhomogeneity / 2)
    bb = b * (1 + rel_inhomogeneity * np.random.rand(N) - rel_inhomogeneity / 2)
    cc = c * (1 + rel_inhomogeneity * np.random.rand(N) - rel_inhomogeneity / 2)
    if N == 2:
        aa = a * np.array([1 + rel_inhomogeneity, 1 - rel_inhomogeneity])
        bb = b * np.array([1 + rel_inhomogeneity, 1 - rel_inhomogeneity])
        cc = c * np.array([1 + rel_inhomogeneity, 1 - rel_inhomogeneity])
    L = np.diag(adj.sum(axis=0)) - adj
    initial_spread = 0.01
    sol = odeint(
        rossler_rhs,
        y0=np.array([1, 1, 0] * N)
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


def compute(K, adj, het):
    res = []
    for _ in tqdm.tqdm(range(Nrepl)):
        x = simulate(K, adj, het)
        errsynch = err_synch(x)

        res.append(
            [
                np.quantile(errsynch, 0.5),
                np.quantile(errsynch, 0.90),
                np.quantile(errsynch, 0.99),
                errsynch.max(),
                errsynch.mean(),
                errsynch.std(),
                skew(errsynch),
            ]
        )
    res = np.array(res)
    return res.mean(axis=0), res.std(axis=0)


if __name__ == "__main__":
    structure = pd.read_csv(adjacency_matrix_path, header=None, sep="\s+")

    adj = np.zeros((28, 28))
    for i, j in structure.values - 1:
        adj[i, j] = adj[j, i] = 1
    p = adj.sum() / (adj.size-adj.shape[0])
    # adj = np.array([[0, 1], [1, 0]])

    #adj = random_adjacency_matrix(50, p)  # use 2p for N=14

    N = adj.shape[0]

    L = np.diag(adj.sum(axis=0)) - adj
    evals = np.linalg.eigvals(L)
    evals.sort()
    lam = np.real(evals[1])

    K = SIGMA / lam

    hets = np.linspace(0, 0.1, 40)

    futures = {}
    esynchs = {}
    rs = {}
    conditions = {}
    with ProcessPoolExecutor(40) as executor:
        for het in hets:
            futures[het] = executor.submit(compute, [K, K, K], adj, het)
        print("jobs submitted...")
        for k, f in futures.items():
            es = f.result()
            esynchs[k] = es
            print(len(esynchs), "/", len(futures))

    med_esynch = [esynchs[k][0][0] for k in hets]
    med_esynch_std = [esynchs[k][1][0] for k in hets]
    q90_esynch = [esynchs[k][0][1] for k in hets]
    q99_esynch = [esynchs[k][0][2] for k in hets]
    max_esynch = [esynchs[k][0][3] for k in hets]
    max_esynch_std = [esynchs[k][1][3] for k in hets]
    meanplusstd_esynch = [esynchs[k][0][4] + 3 * esynchs[k][0][5] for k in hets]
    mean_esynch = [esynchs[k][0][4] for k in hets]
    std_esynch = [esynchs[k][0][5] for k in hets]
    sk_esynch = [esynchs[k][0][6] for k in hets]

    plt.plot(hets, max_esynch, "o", label="Max")
    plt.plot(hets, q90_esynch, "o", label="Q90")
    plt.plot(hets, q99_esynch, "o", label="Q99")
    plt.plot(hets, med_esynch, "o", label="Med")
    plt.plot(hets, meanplusstd_esynch, "o", label="AVG+3STD")
    plt.plot(hets, sk_esynch, "o", label="SKW")
    plt.legend()
    plt.ylabel("Err. Synch.")
    plt.xlabel("Heterogeneity")
    plt.grid(ls=":", c="grey")
    plt.tight_layout()
    plt.savefig(f"err_synch_diff_het_a={A}_b={B}_c={C}_{N}osc.png")
    plt.show()
    plt.clf()

    data = np.array(
        [hets, med_esynch, med_esynch_std, q90_esynch, q99_esynch, max_esynch, max_esynch_std, mean_esynch, std_esynch, sk_esynch]
    ).T
    columns = [
        "het",
        "MedianSynchErr",
        "MedianSynchErrSTD",
        "Q90SynchErr",
        "Q99SynchErr",
        "MaxSynchErr",
        "MaxSynchErrSTD",
        "AVGSynchErr",
        "STDSynchErr",
        "Skewness"
    ]
    pd.DataFrame(data=data, columns=columns).to_csv(
        f"results/err_synch_diff_het_a={A}_b={B}_c={C}_{N}osc.csv", index=False
    )
