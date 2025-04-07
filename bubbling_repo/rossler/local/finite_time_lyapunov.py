import numpy as np
from scipy.integrate import odeint
import tqdm
import gc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from utils import rossler, rossler_tangent, compute_lyapunov_spectrum

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]
plt.rcParams["hatch.linewidth"] = 4

gc.enable()


if __name__ == "__main__":
    a = 0.2
    b = 0.2
    c = 7.0

    system = rossler
    tangent_system = rossler_tangent

    u0 = odeint(
        system,
        y0=[1.0, 1.0, 0.0],
        t=(0, 1000),
        args=(a, b, c),
        tfirst=True,
        rtol=1e-10,
        atol=1e-10,
    )[-1, :]
    N = 200000
    dt = 0.005
    Ntrans = 50000

    sigmas = np.linspace(0, 0.2, 40)
    less = []
    for s in sigmas:
        les, traj = compute_lyapunov_spectrum(rossler_tangent, u0, N, dt, (a, b, c, s))
        les = les[:, Ntrans:]
        less.append(les)
    less = np.stack(less)

    less = less[:, 0, :]

    def fast_moving_sum(a, n=3):
        ret = np.cumsum(a, dtype=float, axis=1)
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        return ret[:, n - 1 :]

    loggrowths = []
    windows = np.logspace(0, 5.17, 300)
    for w in tqdm.tqdm(windows):
        loggrowth = (fast_moving_sum(less, int(w)) * dt).max(axis=1)
        loggrowths.append(loggrowth)
    loggrowths = np.stack(loggrowths)

    floquets = np.loadtxt(f"results/Floquets_rossler_(0.2, 0.2, 7.0).csv")
    floquets_interp = [
        interp1d(floquets[:, 0], floquets[:, i]) for i in range(1, floquets.shape[1])
    ]
    odds = []
    for sigma in sigmas:
        lambdas_prime = [f(sigma) / f(0) for f in floquets_interp]
        odds.append(sum(ll if ll > 0 else 0 for ll in lambdas_prime))

    df1 = pd.read_csv(
        "/Users/giuliotirabassi/Documents/kalman_filter/rossler/final/clutser_results/err_synch_het=0.01_a=0.2_b=0.2_c=7.0_28osc.csv"
    )
    me1 = interp1d(df1.sigma, df1.MaxSynchErr, bounds_error=False)
    df2 = pd.read_csv(
        "/Users/giuliotirabassi/Documents/kalman_filter/rossler/final/clutser_results/err_synch_het=0.01_a=0.2_b=0.2_c=7.0_50osc.csv"
    )
    me2 = interp1d(df2.sigma, df2.MaxSynchErr)
    df3 = pd.read_csv(
        "/Users/giuliotirabassi/Documents/kalman_filter/rossler/final/clutser_results/err_synch_het=0.007_a=0.2_b=0.2_c=7.0_2osc.csv"
    )
    me3 = interp1d(df3.sigma, df3.MaxSynchErr)

    news = 0.8977526693768172 * np.linspace(0.005, 0.15, 80)
    boc = np.load("clutser_results/boccaletti_bubbling_0.2_0.2_7.0_0.01.npy")
    me4 = interp1d(news, boc[:, 0, 1], bounds_error=False)

    df5 = pd.read_csv(
        "/Users/giuliotirabassi/Documents/kalman_filter/rossler/final/clutser_results/err_synch_het=0.01_a=0.2_b=0.2_c=7.0_30osc.csv"
    )
    me5 = interp1d(df5.sigma, df5.MaxSynchErr)

    df6 = pd.read_csv(
        "/Users/giuliotirabassi/Documents/kalman_filter/rossler/final/clutser_results/err_synch_het=0.01_a=0.2_b=0.2_c=7.0_100osc.csv"
    )
    me6 = interp1d(df6.sigma, df6.MaxSynchErr)

    for i in range(len(windows)):
        plt.plot(sigmas, np.exp(loggrowths[i]), c="grey")
    plt.plot(
        sigmas,
        np.exp(loggrowths[np.argmin([np.abs(w - (1 / 0.075) / dt) for w in windows])]),
        c="k",
        lw=2,
        label=r"window$=1/\lambda_{max}$",
    )
    plt.plot(
        sigmas, np.exp(loggrowths.max(axis=0)), c="fuchsia", lw=2, label="envelope"
    )

    plt.plot(
        sigmas,
        10 * np.exp(odds),
        c="cyan",
        lw=2,
        label=r"$10*\exp\left\{\sum_i\tau_i\lambda_i\right\}$",
    )

    plt.plot(df3.sigma, df3.MaxSynchErr / (0.005 / np.sqrt(8)), "o", label="N=2")
    plt.plot(news, me4(news) / 0.005, "o", label="N=10")
    plt.plot(df1.sigma, df1.MaxSynchErr / 0.005, "o", label="N=28")
    plt.plot(df5.sigma, df5.MaxSynchErr / (0.005), "o", label="N=30")
    plt.plot(df2.sigma, df2.MaxSynchErr / 0.005, "o", label="N=50")
    plt.plot(df6.sigma, df6.MaxSynchErr / 0.005, "o", label="N=100")
    plt.yscale("log")
    plt.xlabel(r"$\sigma$")
    plt.legend()
    plt.show()

    plt.plot(np.exp(loggrowths.max(axis=0)), me3(sigmas), "o", label="N=2", mfc="none")
    plt.plot(np.exp(loggrowths.max(axis=0)), me4(sigmas), "o", label="N=10", mfc="none")
    plt.plot(np.exp(loggrowths.max(axis=0)), me1(sigmas), "o", label="N=28", mfc="none")
    plt.plot(np.exp(loggrowths.max(axis=0)), me5(sigmas), "o", label="N=30", mfc="none")
    plt.plot(np.exp(loggrowths.max(axis=0)), me2(sigmas), "o", label="N=50", mfc="none")
    idx = np.argmin(np.abs(sigmas - 0.13))
    plt.plot(np.exp(loggrowths.max(axis=0)[idx]), me3(sigmas[idx]), "o", c="tab:blue")
    plt.plot(np.exp(loggrowths.max(axis=0)[idx]), me4(sigmas[idx]), "o", c="tab:orange")
    plt.plot(np.exp(loggrowths.max(axis=0)[idx]), me1(sigmas[idx]), "o", c="tab:green")
    plt.plot(np.exp(loggrowths.max(axis=0)[idx]), me5(sigmas[idx]), "o", c="tab:red")
    plt.plot(np.exp(loggrowths.max(axis=0)[idx]), me2(sigmas[idx]), "o", c="tab:purple")

    plt.plot(np.logspace(1, 2), 0.003 * np.logspace(1, 2), "--", c="k")
    plt.xlabel("Max SE")
    plt.ylabel("envelope")
    plt.xlim([5, 300])
    plt.ylim([0.02, 5])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(ls=":")
    plt.gca().set_aspect(1)
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    axes[0].plot(sigmas, np.exp(loggrowths.max(axis=0)), "o", c="k", zorder=100)
    axes[0].set_ylabel(r"$\mathcal{A}^{max}$", fontsize=14)
    axes[0].set_yscale("log")
    axes[0].set_xlim([0.045, 0.205])
    axes[0].set_ylim([1, 1e8])
    timescales = windows[np.argmax(loggrowths, axis=0)] * dt
    idx = timescales < 300
    axes[1].plot(sigmas, timescales, "o")
    axes[1].plot(sigmas[idx], timescales[idx], "o-", c="k", zorder=100)
    axes[1].arrow(
        sigmas[idx][0],
        timescales[idx][0],
        0,
        550 - timescales[idx][0],
        color="k",
        width=0.0002,
        length_includes_head=True,
        head_length=60,
        head_width=0.003,
    )
    T = 5.9  # average of 100 pseudo-periods
    xlim = axes[1].get_xlim()
    axes[1].plot(xlim, [T, T], c="tab:blue", lw=2)
    axes[1].set_xlim(xlim)
    axes[1].text(0.17, T + 2, r"$T_{ave}$", fontsize=14, color="tab:blue")
    axes[1].set_xlabel(r"$\sigma$", fontsize=14)
    axes[1].set_ylabel(r"$\tau_{\mathcal{A}^{max}}$", fontsize=14)
    axes[1].set_yscale("log")
    axes[1].set_ylim([0.3, 580])
    for i, ax in enumerate(axes):
        ax.grid(ls=":")
        ax.text(
            x=-0.2,
            y=1,
            s="abcd"[i] + ")",
            fontsize=18,
            horizontalalignment="left",
            transform=ax.transAxes,
        )
    for ax in axes:
        ylim = ax.get_ylim()
        ax.add_patch(
            plt.Rectangle(
                (0.073, -10),
                0.125 - 0.073,
                1e8,
                facecolor="#FFAB9F",
                edgecolor="white",
                hatch=r"//",
            )
        )
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig("figures/finite_time_gain.pdf")
    plt.show()
