# 05_cluster_sweep.py
"""
Sweep candidate k values and report cluster-validity diagnostics, so the
final choice of k is justified rather than assumed.

Reports four things, because no single index is decisive:
  - inertia          (elbow plot)
  - silhouette       (higher is better)
  - Calinski-Harabasz (higher is better)
  - Davies-Bouldin   (LOWER is better)

Plus cluster sizes, because a k that produces a 5-day cluster is useless
for Phase 3 no matter how good its silhouette is.

Usage:
    python 05_cluster_sweep.py --all
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import config
import utils


def sweep(city):

    utils.banner(f"CLUSTER SWEEP: {city}")

    in_path = (
        config.processed_dir(city) / f"{city}_daily_features.csv"
    )

    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found; run 04 first."
        )

    out = config.output_dir(city, "clustering")

    daily = pd.read_csv(in_path, parse_dates=["date"])

    X = daily[config.FEATURES]

    Z = StandardScaler().fit_transform(X)

    results = []

    for k in config.K_RANGE:

        model = KMeans(
            n_clusters=k,
            n_init=config.KMEANS_N_INIT,
            random_state=config.RANDOM_SEED,
        )

        labels = model.fit_predict(Z)

        counts = pd.Series(labels).value_counts()

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(Z, labels),
                "calinski_harabasz": calinski_harabasz_score(Z, labels),
                "davies_bouldin": davies_bouldin_score(Z, labels),
                "smallest_cluster": int(counts.min()),
                "largest_cluster": int(counts.max()),
                "smallest_frac": float(counts.min() / len(labels)),
            }
        )

        candidate = daily.copy()
        candidate["cluster"] = labels
        candidate.to_csv(out / f"candidate_k{k}.csv", index=False)

    results = pd.DataFrame(results)

    results.to_csv(out / "cluster_sweep.csv", index=False)

    print("\n" + results.round(3).to_string(index=False))

    # -----------------------------------------------------
    # DIAGNOSTIC FIGURE
    # -----------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    panels = [
        ("inertia", "Inertia (elbow)", axes[0][0]),
        ("silhouette", "Silhouette (higher better)", axes[0][1]),
        ("calinski_harabasz", "Calinski-Harabasz (higher better)",
         axes[1][0]),
        ("davies_bouldin", "Davies-Bouldin (LOWER better)", axes[1][1]),
    ]

    for column, title, ax in panels:
        ax.plot(results["k"], results[column], marker="o")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(config.K_RANGE))
        ax.grid(alpha=0.3)

    fig.suptitle(f"{city.title()}: cluster validity by k")
    fig.tight_layout()
    fig.savefig(out / "cluster_validity.png", dpi=200)
    plt.close(fig)

    print(f"\n  Figures and candidate labels saved to {out}")

    # -----------------------------------------------------
    # GUIDANCE
    # -----------------------------------------------------

    print(
        "\n  Choose k by BOTH the metrics above and physical "
        "interpretability."
    )
    print(
        "  Inspect candidate_k*.csv, then run 07 on a trial k to see "
        "whether the"
    )
    print(
        "  regimes map onto real meteorology (monsoon phases for "
        "Singapore;"
    )
    print(
        "  frontal / northerly / onshore types for Melbourne). A cluster "
        "holding"
    )
    print(
        "  under ~5% of days is usually too rare to sample for Phase 3."
    )
    print(
        f"\n  Then set CITIES['{city}']['final_k'] in config.py and run 06."
    )


def main():

    cities = utils.parse_cities("Sweep candidate k values.")

    for city in cities:
        sweep(city)


if __name__ == "__main__":
    main()
