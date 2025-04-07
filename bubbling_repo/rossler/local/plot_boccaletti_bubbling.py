import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]
plt.rcParams["pdf.fonttype"] = 42

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
TH = 0.15

all_clusters = list(set(c for _, cs in clusters for c in cs))

ess = np.load("clutser_results/boccaletti_bubbling_0.2_0.2_7.0_0.01.npy")

idx = Ks <= TH
Ks = Ks[idx]
ess = ess[idx, :, :]


fig = plt.figure(figsize=(5, 6))
gs = gridspec.GridSpec(2, 1)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)
axes = [ax1, ax2]  # , ax3]

ax = axes[0]
# transitions = 0.12 / np.array([c for c, _ in clusters])
for i, es in enumerate(all_clusters):
    label = "Network" if all_clusters[i] == tuple(range(10)) else all_clusters[i]
    median = ess[:, i, 0] / np.sqrt(len(all_clusters[i]))
    ax.plot(Ks, median, "o-", label=label, c=colors.get(all_clusters[i], "tab:blue"))
ylim = ax.get_ylim()
ax.vlines(transitions[0], *ylim, ls="--", color="tab:red")
ax.vlines(transitions[1], *ylim, ls="-", color="tab:orange")
ax.vlines(transitions[1], *ylim, ls="--", color="tab:purple")
ax.vlines(transitions[2], *ylim, ls="--", color="tab:green")
ax.set_ylim(ylim)
ax.set_xlabel("K", fontsize=12)
ax.set_ylabel("Median SE", fontsize=12)
ax.grid(ls=":", c="grey")
ax.legend(loc="best", ncol=2)
ax.set_xlim([0.005, 0.05])

ax = axes[1]
for i, es in enumerate(all_clusters):
    label = "Network" if all_clusters[i] == tuple(range(10)) else all_clusters[i]
    ax.plot(
        Ks, ess[:, i, 1], "o-", label=label, c=colors.get(all_clusters[i], "tab:blue")
    )
ylim = ax.get_ylim()
ax.vlines(transitions[0], *ylim, ls="--", color="tab:red")
ax.vlines(transitions[1], *ylim, ls="-", color="tab:orange")
ax.vlines(transitions[1], *ylim, ls="--", color="tab:purple")
ax.vlines(transitions[2], *ylim, ls="--", color="tab:green")
ax.set_ylim(ylim)
ax.set_ylabel("Max SE", fontsize=12)
ax.set_xlabel("K", fontsize=12)
ax.grid(ls=":", c="grey")
ax.set_xlim([0.005, 0.05])

for i, letter in enumerate("ab"):
    axes[i].text(-0.15, 1.02, letter + ")", fontsize=16, transform=axes[i].transAxes)
fig.tight_layout(h_pad=-2)
axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
fig.tight_layout(h_pad=0.05)
plt.savefig("figures/boccaletti_bubbling_old.pdf")
plt.show()
