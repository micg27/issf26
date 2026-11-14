"""
decides how many weather regimes to use
runs k-means clustering for a range of k (2 to 8) and reports four scores (elbow, silhouette, Calinski–Harabasz, Davies–Bouldin),
plus how many days fall in each cluster
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


def sweep(city):              # runs k-means clustering for k=2 to 8, records 4 best scores for each k and size 
                              # of smallest and largest cluster, then plots them

    utils.banner(f"CLUSTER SWEEP: {city}")

    inpath = (
        config.processed_dir(city) / f"{city}_daily_features.csv"
    )

    if not inpath.exists():
        raise FileNotFoundError(
            f"{inpath} not found; run 04 first."
        )

    out = config.output_dir(city, "clustering")

    daily = pd.read_csv(inpath, parse_dates=["date"])

    X = daily[config.FEATURES]

    Z = StandardScaler().fit_transform(X)

    results = []

    for k in config.K_RANGE:

        model = KMeans(
            n_clusters=k,
            n_init=config.KMEANS_INIT,
            random_state=config.SEED,
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    panels = [
        ("inertia", "Inertia", axes[0][0]),
        ("silhouette", "Silhouette", axes[0][1]),
        ("calinski_harabasz", "Calinski-Harabasz",
         axes[1][0]),
        ("davies_bouldin", "Davies-Bouldin", axes[1][1]),
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

def main():

    cities = utils.parse_cities("Sweep candidate k values.")

    for city in cities:
        sweep(city)


if __name__ == "__main__":
    main()
