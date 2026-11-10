# test_networks.py
"""
Checks for networks.py against direct (slow) calculations.

Usage:
    python test_networks.py
"""

import numpy as np

import networks as N


def setup(city="singapore"):
    maps = N.load_maps(city)
    tr = maps["train"]
    Sigma = N.second_moment(tr["X"], tr["w"])
    vg = N.fit_variogram(maps["cells"], Sigma)
    K = N.covariance(maps["cells"], vg)
    return maps, tr, Sigma, vg, K


def direct_mse(tr, S, K, vg, sigma2):
    W = N.ok_weights(K, S, N.sensor_variance(vg, sigma2))
    return float(np.sum(tr["w"] * N.map_mse(tr["X"], S, W, sigma2)))


def test_uniform_background_cancels():
    maps, tr, Sigma, vg, K = setup()
    S = [0, 10, 50, 100]
    W = N.ok_weights(K, S, N.sensor_variance(vg, 0.0))
    assert np.allclose(W.sum(axis=1), 1.0)
    X = tr["X"][:5]
    shifted = X + 7.3                                       # add a uniform background
    assert np.allclose(N.map_mse(X, S, W, 0.0), N.map_mse(shifted, S, W, 0.0))


def test_moment_form_matches_map_by_map():
    maps, tr, Sigma, vg, K = setup()
    S, sigma2 = [3, 40, 90, 150, 170], 0.01
    W = N.ok_weights(K, S, N.sensor_variance(vg, sigma2))
    C = K.shape[0]
    moment = (np.trace(Sigma) - 2 * np.sum(W * Sigma[:, S])
              + np.sum((W @ Sigma[np.ix_(S, S)]) * W) + sigma2 * np.sum(W**2)) / C
    assert abs(moment / direct_mse(tr, S, K, vg, sigma2) - 1) < 1e-9


def test_greedy_matches_direct_and_picks_the_best():
    maps, tr, Sigma, vg, K = setup()
    sigma2 = (0.25 * N.spatial_sd(tr["X"], tr["w"])) ** 2
    S, hist = N.greedy_rmse(K, Sigma, vg, sigma2, 12)
    C = K.shape[0]
    for n in (1, 2, 5, 12):
        assert abs(hist[n - 1] / direct_mse(tr, S[:n], K, vg, sigma2) - 1) < 1e-8
    # At step 6, the greedy choice must be the true argmin over candidates.
    base = S[:5]
    errs = [direct_mse(tr, base + [c], K, vg, sigma2) if c not in base else np.inf
            for c in range(C)]
    assert int(np.argmin(errs)) == S[5]
    assert all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1))


def test_qr_and_kmeans_return_distinct_cells():
    maps, tr, Sigma, vg, K = setup()
    modes, _ = N.pod_modes(tr["X"], tr["w"])
    for n in (5, 22, 40):
        assert len(set(N.qr_network(modes, n))) == n
        assert len(set(N.kmeans_network(maps["cells"], n, 42))) == n


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
