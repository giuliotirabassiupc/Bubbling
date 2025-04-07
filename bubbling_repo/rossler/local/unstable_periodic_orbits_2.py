from scipy.integrate import solve_ivp, odeint
import numba
import numpy as np
import matplotlib.pyplot as plt
import collections
import gc
from newton_method_for_orbits import poincare_method, newton_method
from msf_orbits3 import rossler, rossler_jacobian
from sklearn.cluster import AgglomerativeClustering

gc.enable()


@numba.njit()
def extended_rossler_rhs_msf(t, x, a, b, c):
    x, y, z = x
    return np.array(
        [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ]
    )


def compute_poincare_section(trajectories, poincare_idx):
    others = [i for i in range(trajectories.shape[0]) if i != poincare_idx]
    x = trajectories[poincare_idx, :]
    change_sign = x[0:-1] * x[1:] < 0
    change_sign = np.append(change_sign, [False])
    crossing = change_sign & (x < 0)
    points = trajectories[others, :][:, crossing].T
    return points


def find_periodic_orbits(
    model, y0, max_time, n_points, poincare_idx=0, radius=0.01, args=None, max_period=5
):
    print(f"Integrating system for {max_time} time units")
    dim = len(y0)
    t = np.linspace(0, max_time, n_points)
    traj = odeint(
        model,
        y0=y0,
        t=t,
        rtol=1e-9,
        atol=1e-9,
        args=args,
        tfirst=True,
    ).T
    points = compute_poincare_section(traj, poincare_idx)
    del traj

    orbits = collections.defaultdict(list)

    found_orbits = []
    for i in range(1, max_period + 1):
        print(f"Period-{i}")
        candidates_idx = np.linalg.norm(points[i:, :] - points[:-i, :], axis=1) < radius
        candidates = points[i:, :][candidates_idx, :]
        if not candidates.size:
            print("NO candidates found for this period")
            continue

        clusters = AgglomerativeClustering(
            distance_threshold=radius, n_clusters=None
        ).fit(candidates)
        centroids = []
        for label in set(clusters.labels_):
            idx = clusters.labels_ == label
            centroids.append(candidates[idx, :].mean(axis=0))
        candidates = centroids
        print("Candidates :", len(candidates))

        while candidates:
            candidate = candidates.pop(0)
            for o in found_orbits:
                for p in o:
                    if np.linalg.norm(p - candidate) < radius:
                        continue
            try:
                sol = poincare_method(
                    np.concatenate(([0], candidate)),
                    i,
                    rossler,
                    rossler_jacobian,
                    args=args,
                    j_args=(*args, 0.0),
                    damping=0,
                    dt=0.01,
                    maxiter=25,
                    atol=1e-3,
                )
                x0, T, err, _, _ = newton_method(
                    sol[0],
                    sol[1],
                    rossler,
                    rossler_jacobian,
                    args=args,
                    j_args=(*args, 0.0),
                    maxiter=10,
                    atol=1e-6,
                )
            except Exception as e:
                print(e)
                continue
            if err > 1e-5:
                print("not converging")
                continue
            y = odeint(
                model,
                t=np.linspace(0, 1.1 * T, 1000000),
                y0=x0,
                rtol=1e-9,
                atol=1e-9,
                args=args,
                tfirst=True,
            ).T
            ps = compute_poincare_section(y, poincare_idx)
            del y
            add_orbit = True
            for o in found_orbits:
                dists = [
                    np.subtract.outer(ps[:, i], o[:, i]) ** 2 for i in range(dim - 1)
                ]
                dists = np.stack(dists).sum(axis=0) ** 0.5
                if dists.min() < 1e-4:
                    add_orbit = False
                    break
            if add_orbit:
                found_orbits.append(ps)
                orbits[i].append((x0, T))
                print(x0, T)

        print(f"Orbits found: {len(orbits[i])}\n")
    return dict(orbits)


if __name__ == "__main__":
    args = (0.42, 2.0, 4.0)  # (0.2, 0.2, 7)
    y0 = solve_ivp(extended_rossler_rhs_msf, (0, 1000), y0=[1, 1, 1], args=args).y[
        :, -1
    ]
    orbits = find_periodic_orbits(
        extended_rossler_rhs_msf,
        y0,
        20000,
        10000000,
        radius=0.1,
        args=args,
        max_period=8,
    )

    fig, ax = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
    ax = ax.ravel()
    for key, os in orbits.items():
        for i, (x0, t) in enumerate(os):
            y = solve_ivp(
                extended_rossler_rhs_msf,
                (0, t),
                t_eval=np.linspace(0, t, 500),
                y0=x0,
                args=args,
                rtol=1e-9,
                atol=1e-9,
            ).y
            ax[0].plot(y[0, :], y[1, :], c="k")
            ax[key].plot(y[0, :], y[1, :], c=plt.color_sequences["tab10"][i])
            ax[key].set_title(f"Period-{key}")

    for a in ax:
        a.grid(ls=":", c="grey")
    plt.show()

    with open(f"Rossler_orbits_{args}.csv", "w") as f:
        for key, o in orbits.items():
            for x0, period in o:
                f.write(f"{key}\t" + "\t".join([str(i) for i in x0]) + f"\t{period}\n")
