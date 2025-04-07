import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
from bubbling_search import simulate, err_synch, rmse
import tqdm
from concurrent.futures import ProcessPoolExecutor
import os


def rmse(data):
    xi = data - data.mean(axis=0)
    xinorm = np.sum(xi ** 2, axis=0)
    xiavg = np.mean(xinorm)
    return xiavg**0.5

N=10
clusters = [
        (6.000000000000001, ((2, 3),)),
        (3.999999999999999, ((0, 1, 2, 3), (6, 8))),
        (2.9999999999999987, ((0, 1, 2, 3), (6, 8), (7, 9))),
        (0.8977526693768172, (tuple(range(10)),)),
    ]

sigmas = np.linspace(0, 0.2, 100)
Ks = sigmas / 0.8977526693768172

all_clusters = list(set(c for _, cs in clusters for c in cs))

def compute(K, rel_inhomogeneity, adj):
    ess = [[] for _ in all_clusters]
    for _ in range(100):
        x = simulate([K, K, K], adj, rel_inhomogeneity=rel_inhomogeneity)
        for i, c in enumerate(all_clusters):
            errsynch = err_synch(x[c, :])
            ess[i].append([
                np.median(errsynch),
                np.max(errsynch),
                rmse(x[c, :]).mean(),
            ])
    return ess


if __name__ == "__main__":

    # Create a new graph
    G = nx.Graph()

    # Add 10 nodes
    nodes = range(N)
    G.add_nodes_from(nodes)

    # Define external equitable partition
    # Partition: Group 1 = {0, 1}, Group 2 = {2, 3, 4}, Group 3 = {5, 6, 7}, Group 4 = {8, 9}

    # Define inter-group connections (quotient graph)
    # Example: Group 1 connects to Group 2, Group 2 connects to Group 3, and so on
    edges = []
    for node in (0, 1, 2, 3):
        for other_node in (4, 5, 6, 8):
            edges.append((node, other_node))
    G.add_edges_from(edges + [(2, 3), (4, 5), (5, 9), (5, 7), (7, 9)])
    a = np.array(nx.adjacency_matrix(G).todense())
    degree = a.sum(axis=1)
    laplacian = np.eye(a.shape[0]) * degree - a
    evals, evecs = eigh(laplacian)
    e_i = [np.subtract.outer(evecs[:, i], evecs[:, i]) ** 2 for i in range(evecs.shape[1])]
    s_i = []
    for e in e_i[::-1]:  # from highest to lowest eigenvalue
        if not s_i:
            s_i.append(e)
        else:
            s_i.append(s_i[-1] + e)

    for e, s in zip(evals[::-1], s_i):
        print(e)
        print(np.isclose(s, 2) * 1)

    with ProcessPoolExecutor(os.cpu_count()) as executor:
        futures = [executor.submit(compute, K, 0.01, a) for K in Ks]
        results = []
        for f in tqdm.tqdm(futures):
            results.append(f.result())

    results = np.array(results)
    print(results.shape)

    ess = results.mean(axis=-2)
    assert ess.shape == (Ks.size, len(all_clusters), 3)
    np.save("results/boccaletti_bubbling_0.2_0.2_7.0_0.01.npy", ess)


    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    transitions = 0.075 / np.array([c for c, _ in clusters])
    transitions = 0.138 / np.array([c for c, _ in clusters])
    for i, c in enumerate(all_clusters):
        label = "whole network" if c == tuple(range(10)) else c
        axes[0].plot(Ks, ess[:, i, 0], "o-", label=label)
    for t in transitions:
        axes[0].vlines(t, 0, 2, ls="--", color="k")
    axes[0].set_ylabel("Median Synchronization Error")
    axes[0].legend()

    transitions = 0.12 / np.array([c for c, _ in clusters])
    for i, es in enumerate(all_clusters):
        label = "whole network" if all_clusters[i] == tuple(range(10)) else all_clusters[i]
        axes[1].plot(Ks, ess[:, i, 2], "o-", label=label)
    for t in transitions[:-1]:
        axes[1].vlines(t, 0, 7, ls="--", color="k")
    axes[1].legend()
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("K")

    transitions = 0.12 / np.array([c for c, _ in clusters])
    for i, es in enumerate(all_clusters):
        label = "whole network" if all_clusters[i] == tuple(range(10)) else all_clusters[i]
        axes[2].plot(Ks, ess[:, i, 1], "o-", label=label)
    for t in transitions[:-1]:
        axes[2].vlines(t, 0, 7, ls="--", color="k")
    axes[2].legend()
    axes[2].set_ylabel("Max. Synchronization Error")
    axes[2].set_xlabel("K")
    plt.savefig("boccaletti_0.2_0.2_0.7_0.01.png", dpi=300)
