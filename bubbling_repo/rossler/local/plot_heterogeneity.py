import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]

data = pd.read_csv("clutser_results/err_synch_diff_het_a=0.2_b=0.2_c=7.0_28osc.csv")

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.errorbar(
    100 * data.het / 2,
    data.MedianSynchErr,
    data.MedianSynchErrSTD,
    fmt="o",
    elinewidth=2,
    capsize=2,
    c="tab:blue",
    zorder=100,
)
ax.grid(ls=":", c="grey")
ax.set_ylim([-0.1 / 15, 1.75 / 15])
ax.set_xlabel(r"Heterogeneity [%]", fontsize=14)

axtw = ax.twinx()
axtw.errorbar(
    100 * data.het / 2,
    data.MaxSynchErr,
    data.MaxSynchErrSTD,
    fmt="o",
    elinewidth=2,
    capsize=2,
    c="k",
    zorder=100,
)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylabel("Median SE", fontsize=14, color="tab:blue")
axtw.yaxis.tick_left()
axtw.yaxis.set_label_position("left")
axtw.set_ylabel("Max SE", fontsize=14)
axtw.set_ylim([-0.1, 1.75])
fig.tight_layout()
plt.savefig("figures/heterogeneity.eps")
plt.show()
