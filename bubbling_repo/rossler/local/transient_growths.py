import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from scipy.stats import linregress

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial"]
plt.rcParams["hatch.linewidth"] = 4

N = 2

a, b, c = (0.2, 0.2, 7.0)

floquets = np.loadtxt(f"results/Floquets_rossler_({a}, {b}, {c}).csv")
floquets_interp = [
    interp1d(floquets[:, 0], floquets[:, i]) for i in range(1, floquets.shape[1])
]
lyaps = floquets[0, 1:]
orbits = np.loadtxt(f"../Rossler_orbits_({a}, {b}, {c}).csv")
df = pd.read_csv(f"clutser_results/err_synch_het=0.01_a={a}_b={b}_c={c}_{N}osc.csv")
sigmas = df.sigma
sigmas = sigmas[sigmas <= floquets[:, 0].max()]
odds = []
for sigma in sigmas:
    lambdas_prime = [f(sigma) / f(0) for f in floquets_interp]
    odds.append(sum(ll if ll > 0 else 0 for ll in lambdas_prime))
odds = np.array(odds)

subset = sigmas <= 0.1
fit = linregress(sigmas[subset], odds[subset])

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
ax.plot((0.05, 0.2), (1, 1), c="tab:blue", lw=2)
ax.plot(sigmas, np.exp(odds), "o", c="k")
ax.set_yscale("log")
ax.grid(ls=":")
ax.add_patch(
    plt.Rectangle(
        (0.0735, -10),
        0.125 - 0.0735,
        3e6,
        facecolor="#FFAB9F",
        edgecolor="white",
        hatch=r"//",
    )
)
ax.set_ylim((0.4, 1e6))
ax.set_xlim((0.05, 0.2))
ax.set_ylabel(r"$C$", fontsize=14)
ax.set_xlabel(r"$\sigma$", fontsize=14)

plt.tight_layout()
plt.savefig("figures/transient_growth_scaling.eps")
plt.savefig("figures/transient_growth_scaling.png", dpi=300)
plt.show()
