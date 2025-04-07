import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import numba
import gc
from bubbling_search_het import random_adjacency_matrix

gc.enable()


main_path = "/home/users/giulio.tirabassi/rosslers/"
adjacency_matrix_path = main_path + "Structure/Net_7.dat"

A = 0.2
B = 0.2
C = 7.0
HET = 0.01 #/ np.sqrt(8)# 0.01 for N=28
Nrepl = 100
TTRUE = 100000
NSAMPLE = 5 * TTRUE
NOSC=28

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


def simulate(K, adj, rel_inhomogeneity=HET, Ttrans=2000, Ttrue=TTRUE, Nsample=NSAMPLE):
    random = np.random
    N = adj.shape[0]
    L = np.diag(adj.sum(axis=0)) - adj
    Kx, Ky, Kz = K
    a, b, c = A, B, C
    n = int(Nsample * Ttrue / (Ttrans + Ttrue))
    aa = a * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    bb = b * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)
    cc = c * (1 + rel_inhomogeneity * random.rand(N) - rel_inhomogeneity / 2)

    # tie inhomogeneities to most unstable mode
    #evals, evecs = np.linalg.eigh(L)
    #delta = evecs[:, 1]/evecs[:, 1].max()
    #aa = a * (1 + rel_inhomogeneity * delta)
    #bb = b * (1 + rel_inhomogeneity * delta)
    #cc = c * (1 + rel_inhomogeneity * delta)

    if N == 2:
        aa = a * np.array([1 + rel_inhomogeneity/2, 1 - rel_inhomogeneity/2])
        bb = b * np.array([1 + rel_inhomogeneity/2, 1 - rel_inhomogeneity/2])
        cc = c * np.array([1 + rel_inhomogeneity/2, 1 - rel_inhomogeneity/2])

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

def rmse(data):
    xi = data - data.mean(axis=0)
    xinorm = np.sum(xi ** 2, axis=0)
    xiavg = np.mean(xinorm)
    return xiavg**0.5

def compute(K, adj):
    res = []
    for _ in range(Nrepl):
        x = simulate(K, adj)
        errsynch = err_synch(x)
        rmse_mean = rmse(x)
        median = np.median(errsynch)
        maxe = errsynch.max()
        maxse = max_single_err_synch(x)
        errsynch = np.lib.stride_tricks.sliding_window_view(errsynch, 50).mean(axis=1)
        th = 0.02 #np.quantile(errsynch, q=0.95)
        spike_times = []
        for i in range(errsynch.size - 1):
            if errsynch[i] < th and errsynch[i+1] >= th:
                spike_times.append(i)
        interspike_time = np.diff(spike_times).mean()
        res.append([median, maxe, maxse, interspike_time, rmse_mean])
    res = np.array(res)
    return res.mean(axis=0), res.std(axis=0)


if __name__ == "__main__":
    structure = pd.read_csv(adjacency_matrix_path, header=None, sep="\s+")

    adj = np.zeros((28, 28))
    for i, j in structure.values - 1:
        adj[i, j] = adj[j, i] = 1
    p = adj.sum() / (adj.size-adj.shape[0])

    #adj = random_adjacency_matrix(NOSC, p)  # use 2p for N=14


    N = adj.shape[0]

    L = np.diag(adj.sum(axis=0)) - adj
    evals = np.linalg.eigvals(L)
    evals.sort()
    lam = np.real(evals[1])

    sigmas = np.linspace(0.0, 0.2, 40)
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

    is_time = [esynchs[k][0][3] for k in Ks]
    e_is_time = [esynchs[k][1][3] for k in Ks]

    rmse_mean = [esynchs[k][0][4] for k in Ks]
    rmse_std = [esynchs[k][1][4] for k in Ks]

    plt.errorbar(
        Ks, max_esynch, yerr=e_max_esynch, label="Max", fmt="o", elinewidth=2, capsize=2
    )
    plt.errorbar(
        Ks,
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
    plt.yscale("log")
    plt.grid(ls=":", c="grey")
    plt.tight_layout()
    plt.savefig(f"err_synch_het={HET}_a={A}_b={B}_c={C}_{N}osc.png")
    plt.show()
    plt.clf()

    data = np.array([Ks, sigmas, med_esynch, e_med_esynch, max_esynch, e_max_esynch, is_time, e_is_time, rmse_mean, rmse_std]).T
    columns = ["K", "sigma", "MedianSynchErr", "MedianSynchErrSTD", "MaxSynchErr", "MaxSynchErrSTD", "InterspikeTime", "InterpikeTimeSTD", "RMSE", "RMSE_STD"]
    pd.DataFrame(data=data, columns=columns).to_csv(f"results/err_synch_het={HET}_a={A}_b={B}_c={C}_{N}osc.csv", index=False)