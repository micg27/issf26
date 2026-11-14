"""
shared maps and code for Phase 4-5. the daily-mean maps, the kriging
reconstruction every strategy is scored with, and the network-selection
algorithms.

notation
    C        number of candidate cells
    X        (maps, C) daily-mean PM2.5 maps, one per (day, emission member)
    w        (maps,) map weights: each regime contributes its frequency w_k,
             split equally among its maps, so sum(w) = 1
    S        list of selected candidate indices (columns of X)
    sigma2   sensor noise variance

"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import qr
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

import config



# data


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
    ninregime = m.groupby("regime")["regime"].transform("size").to_numpy()
    w = m["w_k"].to_numpy() / ninregime
    return {"X": X, "w": w / w.sum(), "meta": m.reset_index(drop=True)}


def _load_extension(city, name, C, cellids, k_tag=""):
    ext = xr.open_dataset(config.P3_DIR / city / f"{city}_scenarios_{name}{k_tag}.nc")
    assert np.array_equal(ext["cell_id"].values, cellids)
    dd = ext["pm25_daily"].values
    De, Me, _ = dd.shape
    m = _meta(ext["date"].values, ext["regime"].values, ext["split"].values,
              ext["w_k"].values, Me)
    return _pack(dd.reshape(De * Me, C).astype(float), m)


def load_maps(city, testset="selected", trainset="selected", k_tag=""):    # loads daily average pollution maps and splits them into train/test sets
    if k_tag and (testset != "all2025" or trainset != "all"):
        raise ValueError("Phase 8 K runs exist only for all training and all 2025 days")
    ds = xr.open_dataset(config.P3_DIR / city / f"{city}_scenarios.nc")
    daily = ds["pm25"].mean("hour").values                 
    D, M, C = daily.shape
    meta = _meta(ds["date"].values, ds["regime"].values, ds["split"].values, ds["w_k"].values, M)
    X = daily.reshape(D * M, C).astype(float)
    cells = pd.DataFrame({"cell_id": ds["cell_id"].values,
                          "x": ds["cell_x"].values, "y": ds["cell_y"].values})
    out = {"cells": cells}
    for split in ("train", "test"):
        sel = (meta["split"] == split).to_numpy()
        out[split] = _pack(X[sel], meta[sel])
    if testset == "all2025":
        out["test"] = _load_extension(city, "2025_all", C, ds["cell_id"].values, k_tag)
    elif testset != "selected":
        raise ValueError(testset)
    if trainset == "all":
        out["train"] = _load_extension(city, "train_all", C, ds["cell_id"].values, k_tag)
    elif trainset != "selected":
        raise ValueError(trainset)
    return out


def centred(X):
    return X - X.mean(axis=1, keepdims=True)


def second_moment(X, w):
    Xc = centred(X)
    return (Xc * w[:, None]).T @ Xc


def spatial_sd(X, w):
    return float(np.sum(w * X.std(axis=1)))



# kriging


def exponential(h, nugget, sill, rng):
    return nugget + sill * (1 - np.exp(-h / rng))


def fit_variogram(cells, Sigma):
    xy = cells[["x", "y"]].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    g = 0.5 * (np.diag(Sigma)[:, None] + np.diag(Sigma)[None, :] - 2 * Sigma)
    iu = np.triu_indices(len(xy), 1)
    d, g = d[iu], g[iu]
    keep = d <= config.VG_MAX_FRAC * d.max()
    bins = np.floor(d[keep] / config.VG_BIN_M).astype(int)
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
    xy = cells[["x", "y"]].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    return vg["sill"] * np.exp(-d / vg["range_m"])


def sensor_variance(vg, sigma2):
    return vg["nugget"] + sigma2 + 1e-9 * vg["sill"]      

def ok_weights(K, S, extravar):    # ordinary kriging 
    S = list(S)
    n = len(S)
    A = np.zeros((n + 1, n + 1))
    A[0, 1:] = A[1:, 0] = 1.0
    A[1:, 1:] = K[np.ix_(S, S)] + extravar * np.eye(n)
    R = np.column_stack([np.ones(K.shape[0]), K[:, S]])
    G = np.linalg.solve(A, R.T).T                      
    return G[:, 1:]


def map_mse(X, S, W, sigma2):
    resid = X - X[:, S] @ W.T
    return np.mean(resid**2, axis=1) + sigma2 * np.sum(W**2) / X.shape[1]


def ensemble_rmse(X, w, S, K, vg, sigma2):        # rmse across all maps weighted by weather type
    W = ok_weights(K, S, sensor_variance(vg, sigma2))
    return float(np.sqrt(np.sum(w * map_mse(X, S, W, sigma2))))



# strategy 1: greedy rmse minimisation (S_met)


def greedy_rmse(K, Sigma, vg, sigma2, nmax):
    C = K.shape[0]
    v = sensor_variance(vg, sigma2)
    trS = np.trace(Sigma)
    diagK = np.diag(K) + v

    
    mse1 = (trS - 2 * Sigma.sum(axis=0) + C * np.diag(Sigma) + sigma2 * C) / C
    first = int(np.argmin(mse1))
    S = [first]
    Minv = np.linalg.inv(np.array([[0.0, 1.0], [1.0, diagK[first]]]))
    G = np.column_stack([np.ones(C), K[:, first]]) @ Minv
    history = [float(mse1[first])]

    avail = np.ones(C, bool)
    avail[first] = False
    while len(S) < nmax:
        W = G[:, 1:]
        SigS = Sigma[:, S]                                   
        SigSS = Sigma[np.ix_(S, S)]
        H = W @ SigSS                                       
        L0 = np.sum(W * SigS)
        Q0 = np.sum(H * W)
        F0 = np.sum(W**2)

        cand = np.nonzero(avail)[0]
        B = np.vstack([np.ones(len(cand)), K[np.ix_(S, cand)]])   
        U = Minv @ B
        s = diagK[cand] - np.sum(B * U, axis=0)
        E = K[:, cand] - G @ B                                   
        A = U[1:] / s                                            
        Qm = Sigma[np.ix_(S, cand)]                               
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
        avail[c] = False
        history.append(float(mse[k]))
    return S, history



# strategy 2: pod + qr pivoting (S_met, literature check)

def pod_modes(X, w):
    _, sv, Vt = np.linalg.svd(np.sqrt(w)[:, None] * centred(X), full_matrices=False)
    return Vt.T, sv


def qr_network(modes, n):
    r = max(1, n // config.QR_OVERSAMPLE)
    Psi = modes[:, :r]
    _, _, piv = qr(Psi.T, pivoting=True, mode="economic")
    S = [int(i) for i in piv[:r]]
    avail = np.ones(Psi.shape[0], bool)
    avail[S] = False
    while len(S) < n:
        infoinv = np.linalg.inv(Psi[S].T @ Psi[S])
        lev = np.einsum("ij,jk,ik->i", Psi, infoinv, Psi)
        lev[~avail] = -np.inf
        c = int(np.argmax(lev))
        S.append(c)
        avail[c] = False
    return S


def pod_rmse(X, w, S, modes, sigma2):
    C = X.shape[1]
    r = max(1, len(S) // config.QR_OVERSAMPLE)
    B = np.column_stack([np.ones(C), modes[:, :r]])
    W = B @ np.linalg.pinv(B[S])
    mse = np.mean((X - X[:, S] @ W.T) ** 2, axis=1) + sigma2 * np.sum(W**2) / C
    return float(np.sqrt(np.sum(w * mse)))



# baselines

def kmeans_network(cells, n, seed):
    xy = cells[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=n, n_init=config.KMEANS_INIT, random_state=seed).fit(xy)
    S = []
    for centre in km.cluster_centers_:
        order = np.argsort(np.hypot(*(xy - centre).T))
        S.append(int(next(i for i in order if i not in S)))
    return S


def random_networks(C, n, k, seed):
    rng = np.random.default_rng([seed, n])
    return [sorted(rng.choice(C, n, replace=False).tolist()) for _ in range(k)]
