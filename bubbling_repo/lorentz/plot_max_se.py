import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]

df = pd.read_csv("max_se_2_0.001.csv")
gf = pd.read_csv("gain_factor_dt=0.005.csv")

fig, ax = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
ax[0].plot(
    gf.sigmas,
    0.005 * np.exp(gf.logAmax),
    "o-",
    label=r"$\mathcal{A}^{max}$ / 200",
    c="k",
)
ax[0].plot(df.sigma, df.masSE, "o", mfc="none", label="Max SE", c="tab:red")
ax[0].set_yscale("log")
ax[0].legend()
ax[0].set_ylim([1e-3, 1e2])

ax[1].plot(gf.sigmas, gf.tau, "o-", c="k")
ax[1].set_yscale("log")
ax[1].set_ylabel(r"$\tau_{\mathcal{A}^{max}}$", fontsize=14)
ax[1].set_xlabel(r"$\sigma$", fontsize=14)
xlim = ax[1].get_xlim()
# ax[1].plot(xlim, [0.752, 0.752], c="tab:blue")
ax[1].plot(xlim, [0.0065, 0.0065], c="tab:blue")
# ax[1].plot(xlim, [0.752 + 0.11, 0.752 + 0.11], c="tab:blue", ls="--")
# ax[1].plot(xlim, [0.752 - 0.11, 0.752 - 0.11], c="tab:blue", ls="--")
ax[1].set_xlim(xlim)
ax[1].text(
    1.25,
    0.009,
    r"$T_{th}$",
    fontsize=14,
    color="tab:blue",
    horizontalalignment="center",
)
ax[1].set_ylim([2e-3, 1e1])
for i, a in enumerate(ax):
    a.grid(ls=":", c="grey")
    a.text(-0.2, 1.02, "ab"[i] + ")", fontsize=16, transform=a.transAxes)

plt.tight_layout()
plt.savefig("lorentz.pdf")
plt.savefig("lorentz.png")
plt.clf()
