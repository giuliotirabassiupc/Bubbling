import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]
plt.rcParams["hatch.linewidth"] = 4

N = 50

if __name__ == "__main__":
    A = 0.2
    B = 0.2
    C = 7.0
    df = pd.read_csv(f"clutser_results/err_synch_het=0.01_a={A}_b={B}_c={C}_{N}osc.csv")
    df2 = pd.read_csv(
        f"clutser_results/err_synch_het=0.0040_a={A}_b={B}_c={C}_2osc.csv"
    )
    sigmas1 = df.sigma  # 0.1625
    sigmas2 = df2.sigma
    idx1 = (sigmas1 >= 0.05) & (sigmas1 <= 0.16)
    idx2 = (sigmas2 >= 0.05) & (sigmas2 <= 0.16)
    df = df[idx1]
    df2 = df2[idx2]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(4.5, 5.5))
    ax[0].errorbar(
        df.sigma,
        df.MedianSynchErr,
        yerr=df.MedianSynchErrSTD,
        fmt="o",
        elinewidth=2,
        capsize=2,
        c="tab:blue",
        zorder=100,
    )
    axtw = ax[0].twinx()
    axtw.errorbar(
        df.sigma,
        df.MaxSynchErr,
        yerr=df.MaxSynchErrSTD,
        fmt="o",
        elinewidth=2,
        capsize=2,
        c="k",
    )
    ax[0].tick_params(axis="y", labelcolor="tab:blue")
    ax[0].yaxis.tick_right()
    ax[0].yaxis.set_label_position("right")
    ax[0].set_ylabel("Median SE", color="tab:blue", fontsize=14)
    axtw.yaxis.tick_left()
    axtw.yaxis.set_label_position("left")
    axtw.set_ylabel("Max SE", fontsize=14)
    axtw.set_yticks([0, 1, 2, 3])
    axtw.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axtw.set_ylim([-0.1, 3])
    ax[0].set_yticks([0, 0.25, 0.5, 0.75])
    ax[0].set_ylim([-0.025, 0.75])
    ax[0].grid(ls=":", c="grey")

    ax[1].errorbar(
        df2.sigma,
        df2.MedianSynchErr,
        yerr=df2.MedianSynchErrSTD,
        fmt="o",
        elinewidth=2,
        capsize=2,
        c="tab:blue",
        zorder=100,
    )
    axtw = ax[1].twinx()
    axtw.errorbar(
        df2.sigma,
        df2.MaxSynchErr,
        yerr=df2.MaxSynchErrSTD,
        fmt="o",
        elinewidth=2,
        capsize=2,
        c="k",
    )
    ax[1].tick_params(axis="y", labelcolor="tab:blue")
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].set_ylabel("Median SE", color="tab:blue", fontsize=14)
    axtw.yaxis.tick_left()
    axtw.yaxis.set_label_position("left")
    axtw.set_ylabel("Max SE", fontsize=14)
    axtw.set_yticks([0, 1.5, 3, 4.5])
    axtw.set_ylim([-0.3, 4.5])
    ax[1].set_yticks([0, 0.15, 0.3, 0.45])
    ax[1].set_ylim([-0.03, 0.45])
    ax[1].grid(ls=":", c="grey")

    ax[1].set_xlabel(r"$\sigma$", fontsize=14)
    ax[0].text(-0.2, 1.05, "a)", fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.05, "b)", fontsize=18, transform=ax[1].transAxes)
    ax[0].text(0.7, 0.8, r"$N=50$", fontsize=10, transform=ax[0].transAxes)
    ax[1].text(0.7, 0.8, r"$N=2$", fontsize=10, transform=ax[1].transAxes)
    for aa in ax:
        ylim = aa.get_ylim()
        aa.add_patch(
            plt.Rectangle(
                (0.073, -10),
                0.125 - 0.073,
                30,
                facecolor="#FFAB9F",
                edgecolor="white",
                hatch=r"//",
            )
        )
        aa.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(f"figures/max_synch_err_{A}_{B}_{C}_2osc.png", dpi=300)
    plt.savefig(f"figures/max_synch_err_{A}_{B}_{C}_2osc.eps")
    plt.show()
    plt.clf()
