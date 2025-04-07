import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import gridspec

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["hatch.linewidth"] = 4

clusters = [
    (6.000000000000001, ((2, 3),)),
    (3.999999999999999, ((0, 1, 2, 3), (6, 8))),
    (2.9999999999999987, ((0, 1, 2, 3), (6, 8), (7, 9))),
    (0.8977526693768172, (tuple(range(10)),)),
]

colors = {
    (0, 1, 2, 3): "tab:orange",
    (2, 3): "tab:red",
    (6, 8): "tab:purple",
    (7, 9): "tab:green",
}

Ks_allcoupling = np.linspace(0.005, 0.15, 80)  # th 0.06
Ks_only_x = np.linspace(0.005, 0.30, 80)  # th 0.12


transitions_allc = 0.075 / np.array([c for c, _ in clusters])
transitions_xonly = 0.138 / np.array([c for c, _ in clusters])

transitions = transitions_allc
Ks = Ks_allcoupling
TH = 100

sigmas = np.linspace(0, 0.2, 100)
Ks = sigmas / 0.8977526693768172

all_clusters = list(set(c for _, cs in clusters for c in cs))

ess = np.load("clutser_results/boccaletti_bubbling_0.2_0.2_7.0_0.01_new.npy")

idx = (Ks <= TH) & (Ks > 0)
Ks = Ks[idx]
ess = ess[idx, :, :]
# Create a new graph
G = nx.Graph()

# Add 10 nodes
nodes = range(10)
G.add_nodes_from(nodes)

edges = []
for node in (0, 1, 2, 3):
    for other_node in (4, 5, 6, 8):
        edges.append((node, other_node))

G.add_edges_from(edges + [(2, 3), (4, 5), (5, 9), (5, 7), (7, 9)])

fig = plt.figure(figsize=(10.5, 5))
gs = gridspec.GridSpec(2, 3)
ax0 = plt.subplot(gs[:, 0])
ax1 = plt.subplot(gs[0, 1])
ax2 = plt.subplot(gs[0, 2], sharex=ax1)
ax3 = plt.subplot(gs[1, 1], sharex=ax1)
ax4 = plt.subplot(gs[1, 2], sharex=ax1)
axes = [ax0, ax1, ax2, ax3, ax4]

ax = axes[0]
# Draw the graph
pos = {
    0: (3, 2),
    1: (0, 2),
    2: (2, 2),
    3: (1, 2),
    4: (2, 1),
    5: (1, 1),
    6: (2, 3),
    7: (1, 0),
    8: (1, 3),
    9: (2, 0),
}
for color, node in {
    "tab:orange": (2, 3, 0, 1),
    "tab:green": (7, 9),
    "grey": (4, 5),
    "tab:purple": (6, 8),
}.items():
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=color,
        nodelist=node,
        node_size=500,
        ax=ax,
    )
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
nx.draw_networkx_labels(
    G,
    {k: (v[0], v[1] - 0.01) for k, v in pos.items()},
    dict(zip(nodes, nodes)),
    font_size=14,
    font_color="whitesmoke",
    ax=ax,
    horizontalalignment="center",
    verticalalignment="center",
)
ax.set_ylim([-0.35, 3.35])
ax.plot((2, 1), (2, 2), "o", c="tab:red", ms=16)
ax.axis("off")

ax = axes[1]
cluster = (2, 3)
scale_factor = [i for i, j in clusters if cluster in j][0]
i = all_clusters.index(cluster)
# transitions = 0.12 / np.array([c for c, _ in clusters])
label = cluster
medianes = ess[:, i, 0]
maxes = ess[:, i, 1]
ax.plot(Ks * scale_factor, medianes, "d", c=colors[cluster], mfc="white")
ax.plot(Ks * scale_factor, maxes, "o", c=colors[cluster])
ylim = ax.get_ylim()
ax.set_ylabel("SE", fontsize=12)
ax.set_xlabel(r"$\sigma_8$", fontsize=12)
ax.grid(ls=":", c="grey")
ax.plot([], [], "d", label="Median", c="k", mfc="white")
ax.plot([], [], "o", label="Max", c="k")
ax.set_ylim(ylim)
ax.set_xlim([0.01, 0.2])
ax.legend(loc="upper right")

ax = axes[2]
for cluster in ((0, 1, 2, 3), (6, 8)):
    scale_factor = [i for i, j in clusters if cluster in j][0]
    i = all_clusters.index(cluster)
    # transitions = 0.12 / np.array([c for c, _ in clusters])
    label = cluster
    medianes = ess[:, i, 0]
    maxes = ess[:, i, 1]
    ax.plot(Ks * scale_factor, medianes, "d", c=colors[cluster], mfc="white")
    ax.plot(Ks * scale_factor, maxes, "o", c=colors[cluster])
ylim = ax.get_ylim()
ax.set_xlabel(r"$\sigma_4$", fontsize=12)
ax.grid(ls=":", c="grey")
ax.plot([], [], "d", label="Median", c="k", mfc="white")
ax.plot([], [], "o", label="Max", c="k")
ax.set_ylim(ylim)
ax.set_xlim([0.01, 0.2])
ax.legend(loc="upper right")

ax = axes[3]
cluster = (7, 9)
scale_factor = [i for i, j in clusters if cluster in j][0]
i = all_clusters.index(cluster)
# transitions = 0.12 / np.array([c for c, _ in clusters])
label = cluster
medianes = ess[:, i, 0]
maxes = ess[:, i, 1]
ax.plot(Ks * scale_factor, medianes, "d", c=colors[cluster], mfc="white")
ax.plot(Ks * scale_factor, maxes, "o", c=colors[cluster])
ax.set_xlabel(r"$\sigma_3$", fontsize=12)
ax.set_ylabel("SE", fontsize=12)
ylim = ax.get_ylim()
ax.grid(ls=":", c="grey")
ax.plot([], [], "d", label="Median", c="k", mfc="white")
ax.plot([], [], "o", label="Max", c="k")
ax.set_ylim(ylim)
ax.set_xlim([0.01, 0.2])
ax.legend(loc="upper right")

ax = axes[4]
cluster = tuple(range(10))
scale_factor = [i for i, j in clusters if cluster in j][0]
i = all_clusters.index(cluster)
# transitions = 0.12 / np.array([c for c, _ in clusters])
label = cluster
medianes = ess[:, i, 0]
maxes = ess[:, i, 1]
ax.plot(Ks * scale_factor, medianes, "d", c="tab:blue", mfc="white")
ax.plot(Ks * scale_factor, maxes, "o", c="tab:blue")
ylim = ax.get_ylim()
ax.set_xlabel(r"$\sigma$", fontsize=12)
ax.grid(ls=":", c="grey")
ax.plot([], [], "d", label="Median", c="k", mfc="white")
ax.plot([], [], "o", label="Max", c="k")
ax.set_ylim(ylim)
ax.set_xlim([0.01, 0.2])
ax.legend(loc="upper right")

for i, letter in enumerate("abcde"):
    axes[i].text(-0.15, 1.02, letter + ")", fontsize=16, transform=axes[i].transAxes)
    if i == 0:
        continue
    axes[i].add_patch(
        plt.Rectangle(
            (0.073, -10),
            0.125 - 0.073,
            30,
            facecolor="#FFAB9F",
            edgecolor="white",
            hatch=r"//",
        )
    )


# axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
fig.tight_layout()
plt.savefig("figures/boccaletti_bubbling.pdf")

plt.show()
