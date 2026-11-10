# networks.py
"""
Shared machinery for Phase 4-5: the daily-mean maps, the kriging
reconstruction every strategy is scored with, and the network-selection
algorithms.

Notation
    C        number of candidate cells
    X        (maps, C) daily-mean PM2.5 maps, one per (day, emission member)
    w        (maps,) map weights: each regime contributes its frequency w_k,
             split equally among its maps, so sum(w) = 1
    S        list of selected candidate indices (columns of X)
    sigma2   sensor noise variance

Reconstruction is ORDINARY KRIGING with an exponential covariance fitted to
the training maps. It is linear: x_hat = W x_S, where W is (C, |S|) and
each row sums to 1. Because of that row sum, a spatially uniform
background (the daily b_d) cancels out of the error exactly. So the
error depends only on each map's deviation from its own spatial mean.

Expected error under sensor noise eps ~ N(0, sigma2 I) is exact, no
sampling needed:
    E ||x - W (x_S + eps)||^2 = ||x - W x_S||^2 + sigma2 ||W||_F^2
and over the weighted ensemble, with Sigma = sum_m w_m x~_m x~_m^T
(x~ = map minus its spatial mean),
    MSE(S) = [ tr Sigma - 2 tr(W Sigma[S,:]) + tr(W Sigma[S,S] W^T)
               + sigma2 ||W||_F^2 ] / C.
RMSE(S) = sqrt(MSE(S)): the root of the weighted mean squared error over
all maps and cells.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import qr
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

import config


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

def _meta(dates, regimes, splits, w_k, M):
    D = len(dates)
    return pd.DataFrame({
        "day": np.repeat(np.arange(D), M),
        "member": np.tile(np.arange(M), D),
        "date": np.repeat(dates, M),
        "regime": np.repeat(regimes.astype(str), M),
        "split": np.repeat(splits.astype(str), M),
        "w_k": np.repeat(w_k, M),
    })


def _pack(X, m):
    n_in_regime = m.groupby("regime")["regime"].transform("size").to_numpy()
    w = m["w_k"].to_numpy() / n_in_regime
    return {"X": X, "w": w / w.sum(), "meta": m.reset_index(drop=True)}


def _load_extension(city, name, C, cell_ids, k_tag=""):
    ext = xr.open_dataset(config.PHASE3_DIR / city / f"{city}_scenarios_{name}{k_tag}.nc")
    assert np.array_equal(ext["cell_id"].values, cell_ids)
    dd = ext["pm25_daily"].values
    De, Me, _ = dd.shape
    m = _meta(ext["date"].values, ext["regime"].values, ext["split"].values,
              ext["w_k"].values, Me)
    return _pack(dd.reshape(De * Me, C).astype(float), m)


def load_maps(city, test_set="selected", train_set="selected", k_tag=""):
    """Daily-mean maps and regime weights, split into train and test.

    test_set="selected" (pre-registered): the 18-19 test days Phase 2
    sampled. test_set="all2025" (post-results extension): every 2025 day.
    train_set="selected" (pre-registered): the 44-45 sampled training days.
    train_set="all" (post-results extension): every 2021-2024 day.
    Extensions come from Phase 3's 05_extend_test_days.py.
    k_tag (Phase 8, e.g. "_K200"): read the extensions simulated at another K;
    needs test_set="all2025" and train_set="all".
    """
    if k_tag and (test_set != "all2025" or train_set != "all"):
        raise ValueError("Phase 8 K runs exist only for all training and all 2025 days")
    ds = xr.open_dataset(config.PHASE3_DIR / city / f"{city}_scenarios.nc")
    daily = ds["pm25"].mean("hour").values                 # (day, member, cell)
    D, M, C = daily.shape
    meta = _meta(ds["date"].values, ds["regime"].values, ds["split"].values, ds["w_k"].values, M)
    X = daily.reshape(D * M, C).astype(float)
    cells = pd.DataFrame({"cell_id": ds["cell_id"].values,
                          "x": ds["cell_x"].values, "y": ds["cell_y"].values})
    out = {"cells": cells}
    for split in ("train", "test"):
        sel = (meta["split"] == split).to_numpy()
        out[split] = _pack(X[sel], meta[sel])
    if test_set == "all2025":
        out["test"] = _load_extension(city, "2025_all", C, ds["cell_id"].values, k_tag)
    elif test_set != "selected":
        raise ValueError(test_set)
    if train_set == "all":
        out["train"] = _load_extension(city, "train_all", C, ds["cell_id"].values, k_tag)
    elif train_set != "selected":
        raise ValueError(train_set)
    return out


def centred(X):
    return X - X.mean(axis=1, keepdims=True)


def second_moment(X, w):
    Xc = centred(X)
    return (Xc * w[:, None]).T @ Xc


def spatial_sd(X, w):
    """Weighted mean (over maps) of each map's spatial standard deviation."""
    return float(np.sum(w * X.std(axis=1)))


# ---------------------------------------------------------
# KRIGING
# ---------------------------------------------------------

def exponential(h, nugget, sill, rng):
    return nugget + sill * (1 - np.exp(-h / rng))


def fit_variogram(cells, Sigma):
    """Exponential variogram fitted to the weighted mean semivariogram.

    The mean semivariance between cells i and j over the ensemble is
    (Sigma_ii + Sigma_jj - 2 Sigma_ij) / 2, so no per-map loop is needed.
    """
    xy = cells[["x", "y"]].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    g = 0.5 * (np.diag(Sigma)[:, None] + np.diag(Sigma)[None, :] - 2 * Sigma)
    iu = np.triu_indices(len(xy), 1)
    d, g = d[iu], g[iu]
    keep = d <= config.VARIOGRAM_MAX_DIST_FRACTION * d.max()
    bins = np.floor(d[keep] / config.VARIOGRAM_BIN_M).astype(int)
    n = np.bincount(bins)
    ok = n > 0
    h = np.bincount(bins, d[keep])[ok] / n[ok]
    gam = np.bincount(bins, g[keep])[ok] / n[ok]
    n = n[ok]
    ok = np.ones(len(n), bool)
    p0 = [0.0, gam[ok].max(), h[ok].max() / 3]
    (nug, sill, rng), _ = curve_fit(
        exponential, h[ok], gam[ok], p0=p0, sigma=1 / np.sqrt(n[ok]),
        bounds=([0, 1e-12, 100.0], [np.inf, np.inf, 1e6]), maxfev=20000)
    return {"nugget": float(nug), "sill": float(sill), "range_m": float(rng),
            "bins_h": h[ok].tolist(), "bins_gamma": gam[ok].tolist(),
            "bins_n": n[ok].tolist()}


def covariance(cells, vg):
    """Covariance between candidate cells (structured part only)."""
    xy = cells[["x", "y"]].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    return vg["sill"] * np.exp(-d / vg["range_m"])


def sensor_variance(vg, sigma2):
    """Extra variance on the sensor block: fitted nugget + sensor noise."""
    return vg["nugget"] + sigma2 + 1e-9 * vg["sill"]      # tiny jitter for f = 0


def ok_weights(K, S, extra_var):
    """Ordinary kriging weights W (C, |S|), computed directly."""
    S = list(S)
    n = len(S)
    A = np.zeros((n + 1, n + 1))
    A[0, 1:] = A[1:, 0] = 1.0
    A[1:, 1:] = K[np.ix_(S, S)] + extra_var * np.eye(n)
    R = np.column_stack([np.ones(K.shape[0]), K[:, S]])
    G = np.linalg.solve(A, R.T).T                       # A symmetric
    return G[:, 1:]


def map_mse(X, S, W, sigma2):
    """Expected MSE per map (rows of X) for network S with weights W."""
    resid = X - X[:, S] @ W.T
    return np.mean(resid**2, axis=1) + sigma2 * np.sum(W**2) / X.shape[1]


def ensemble_rmse(X, w, S, K, vg, sigma2):
    W = ok_weights(K, S, sensor_variance(vg, sigma2))
    return float(np.sqrt(np.sum(w * map_mse(X, S, W, sigma2))))


# ---------------------------------------------------------
# STRATEGY 1: GREEDY RMSE MINIMISATION (S_met)
# ---------------------------------------------------------

def greedy_rmse(K, Sigma, vg, sigma2, n_max):
    """Forward greedy selection minimising the ensemble kriging MSE.

    Exact, but evaluates every candidate at every step with rank-one
    updates of the bordered kriging matrix, O(C n) per candidate instead
    of O(n^3 + C n^2). Returns the selection order and MSE after each step.

    State for current network S (n sensors), index 0 = the OK constraint:
        Minv = inverse of [[0, 1^T], [1, K_SS + v I]]   ((n+1) x (n+1))
        G    = [1, K[:, S]] @ Minv                       (C x (n+1))
        W    = G[:, 1:]                                  kriging weights
    Adding candidate c with b = [1; K[S, c]], d = K_cc + v:
        u = Minv b,  s = d - b.u,  e = K[:, c] - G b,  a = u[1:] / s
        W' = [W - e a^T, e / s]
    """
    C = K.shape[0]
    v = sensor_variance(vg, sigma2)
    trS = np.trace(Sigma)
    diagK = np.diag(K) + v

    # Step 1, direct: with one sensor every prediction equals that reading.
    mse1 = (trS - 2 * Sigma.sum(axis=0) + C * np.diag(Sigma) + sigma2 * C) / C
    first = int(np.argmin(mse1))
    S = [first]
    Minv = np.linalg.inv(np.array([[0.0, 1.0], [1.0, diagK[first]]]))
    G = np.column_stack([np.ones(C), K[:, first]]) @ Minv
    history = [float(mse1[first])]

    available = np.ones(C, bool)
    available[first] = False
    while len(S) < n_max:
        W = G[:, 1:]
        SigS = Sigma[:, S]                                   # (C, n)
        SigSS = Sigma[np.ix_(S, S)]
        H = W @ SigSS                                        # (C, n)
        L0 = np.sum(W * SigS)
        Q0 = np.sum(H * W)
        F0 = np.sum(W**2)

        cand = np.nonzero(available)[0]
        B = np.vstack([np.ones(len(cand)), K[np.ix_(S, cand)]])   # (n+1, Nc)
        U = Minv @ B
        s = diagK[cand] - np.sum(B * U, axis=0)
        E = K[:, cand] - G @ B                                     # (C, Nc)
        A = U[1:] / s                                              # (n, Nc)
        Qm = Sigma[np.ix_(S, cand)]                                # (n, Nc)
        ee = np.sum(E**2, axis=0)

        L = L0 - np.sum(E * (SigS @ A), axis=0) + np.sum(E * Sigma[:, cand], axis=0) / s
        Q = (Q0 - 2 * np.sum(E * (H @ A), axis=0) + np.sum(A * (SigSS @ A), axis=0) * ee
             + 2 / s * (np.sum(E * (W @ Qm), axis=0) - np.sum(A * Qm, axis=0) * ee)
             + np.diag(Sigma)[cand] * ee / s**2)
        F = F0 - 2 * np.sum(E * (W @ A), axis=0) + ee * np.sum(A**2, axis=0) + ee / s**2
        mse = (trS - 2 * L + Q + sigma2 * F) / C

        k = int(np.argmin(mse))
        c, u, sk, e = cand[k], U[:, k], s[k], E[:, k]
        n1 = Minv.shape[0]
        Mnew = np.empty((n1 + 1, n1 + 1))
        Mnew[:n1, :n1] = Minv + np.outer(u, u) / sk
        Mnew[:n1, n1] = Mnew[n1, :n1] = -u / sk
        Mnew[n1, n1] = 1 / sk
        Minv = Mnew
        G = np.column_stack([G - np.outer(e, u) / sk, e / sk])
        S.append(int(c))
        available[c] = False
        history.append(float(mse[k]))
    return S, history


# ---------------------------------------------------------
# STRATEGY 2: POD + QR PIVOTING (S_met, literature check)
# ---------------------------------------------------------

def pod_modes(X, w):
    """Spatial POD modes (C, rank) of the weighted, per-map-centred maps."""
    _, sv, Vt = np.linalg.svd(np.sqrt(w)[:, None] * centred(X), full_matrices=False)
    return Vt.T, sv


def qr_network(modes, n):
    """n sensors from r = n // QR_OVERSAMPLING modes (Manohar et al. 2018).

    The first r sensors are QR column pivots of Psi_r^T. QR pivoting alone
    gives at most r useful pivots, so the remaining n - r are added
    greedily, maximising det(Theta^T Theta) (Theta = Psi_r rows at the
    sensors). Each step adds the candidate with the largest leverage
    psi_c^T (Theta^T Theta)^-1 psi_c, which is the same volume-maximising
    criterion QR pivoting approximates.
    """
    r = max(1, n // config.QR_OVERSAMPLING)
    Psi = modes[:, :r]
    _, _, piv = qr(Psi.T, pivoting=True, mode="economic")
    S = [int(i) for i in piv[:r]]
    available = np.ones(Psi.shape[0], bool)
    available[S] = False
    while len(S) < n:
        info_inv = np.linalg.inv(Psi[S].T @ Psi[S])
        lev = np.einsum("ij,jk,ik->i", Psi, info_inv, Psi)
        lev[~available] = -np.inf
        c = int(np.argmax(lev))
        S.append(c)
        available[c] = False
    return S


def pod_rmse(X, w, S, modes, sigma2):
    """QR's native reconstruction (Manohar et al. 2018): least squares on a
    constant plus r = N // QR_OVERSAMPLING POD modes. Used only as a check
    that QR isn't being judged unfairly by the shared kriging."""
    C = X.shape[1]
    r = max(1, len(S) // config.QR_OVERSAMPLING)
    B = np.column_stack([np.ones(C), modes[:, :r]])
    W = B @ np.linalg.pinv(B[S])
    mse = np.mean((X - X[:, S] @ W.T) ** 2, axis=1) + sigma2 * np.sum(W**2) / C
    return float(np.sqrt(np.sum(w * mse)))


# ---------------------------------------------------------
# BASELINES
# ---------------------------------------------------------

def kmeans_network(cells, n, seed):
    """Spatial coverage (Walvoort et al. 2010): k-means on cell coordinates,
    then the candidate cell nearest each cluster centre (no repeats)."""
    xy = cells[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=n, n_init=config.KMEANS_N_INIT, random_state=seed).fit(xy)
    S = []
    for centre in km.cluster_centers_:
        order = np.argsort(np.hypot(*(xy - centre).T))
        S.append(int(next(i for i in order if i not in S)))
    return S


def random_networks(C, n, k, seed):
    rng = np.random.default_rng([seed, n])
    return [sorted(rng.choice(C, n, replace=False).tolist()) for _ in range(k)]
