import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

HMIN = 50.0        
CFL = 0.8


# meteorology in domain

def load_met(city, phase2dir, utcstart, utcend):
    years = range(utcstart.year, utcend.year + 1)
    files = [phase2dir / city / f"{city}_hourly_{y}.nc" for y in years]
    ds = xr.open_mfdataset(files, combine="by_coords")[["u10", "v10", "blh"]]
    ds = ds.sel(time=slice(utcstart, utcend)).load()
    nexpected = int((utcend - utcstart) / pd.Timedelta(hours=1)) + 1
    assert ds.sizes["time"] == nexpected, (
        f"{city}: met has {ds.sizes['time']} hours in {utcstart}..{utcend}, "
        f"expected {nexpected}"
    )
    return ds


def met_to_domain(met, domain):
    toll = Transformer.from_crs(domain.attrs["crs"], "EPSG:4326", always_xy=True)
    X, Y = np.meshgrid(domain.x.values, domain.y.values)
    lon, lat = toll.transform(X, Y)

    lats = met.latitude.values
    lons = met.longitude.values
    order = np.argsort(lats)
    lats = lats[order]
    pts = np.column_stack([
        np.clip(lat.ravel(), lats.min(), lats.max()),
        np.clip(lon.ravel(), lons.min(), lons.max()),
    ])

    out = {}
    for var in ("u10", "v10", "blh"):
        a = met[var].values[:, order, :]
        fields = np.empty((a.shape[0],) + X.shape)
        for t in range(a.shape[0]):
            f = RegularGridInterpolator((lats, lons), a[t], method="linear")
            fields[t] = f(pts).reshape(X.shape)
        out[var] = fields
    out["blh"] = np.maximum(out["blh"], HMIN)
    return out



def _mc_limited_slope(c, axis):
    dfwd = np.diff(c, axis=axis, append=np.take(c, [-1], axis=axis))
    dbwd = np.diff(c, axis=axis, prepend=np.take(c, [0], axis=axis))
    dcen = 0.5 * (dfwd + dbwd)
    same = np.sign(dfwd) == np.sign(dbwd)
    lim = np.minimum(np.minimum(2 * np.abs(dfwd), 2 * np.abs(dbwd)), np.abs(dcen))
    return np.where(same, np.sign(dcen) * lim, 0.0)


def _advect_1d(c, velface, dt, dx, bg, axis):
    pad = [(0, 0)] * c.ndim
    pad[axis] = (2, 2)
    g = np.pad(c, pad, mode="constant")
    idxlo = [slice(None)] * c.ndim
    idxhi = [slice(None)] * c.ndim
    idxlo[axis] = slice(0, 2)
    idxhi[axis] = slice(-2, None)
    g[tuple(idxlo)] = bg[(...,) + (None,) * (c.ndim - 1)] if np.ndim(bg) else bg
    g[tuple(idxhi)] = bg[(...,) + (None,) * (c.ndim - 1)] if np.ndim(bg) else bg

    s = _mc_limited_slope(g, axis)
    n = c.shape[axis]
    left = [slice(None)] * c.ndim
    right = [slice(None)] * c.ndim
    left[axis] = slice(1, n + 2)     
    right[axis] = slice(2, n + 3)    
    cl, sl = g[tuple(left)], s[tuple(left)]
    cr, sr = g[tuple(right)], s[tuple(right)]

    nu = velface * dt / dx
    faceval = np.where(
        velface >= 0,
        cl + 0.5 * (1 - nu) * sl,
        cr - 0.5 * (1 + nu) * sr,
    )
    flux = velface * faceval
    lo = [slice(None)] * c.ndim
    hi = [slice(None)] * c.ndim
    lo[axis] = slice(0, n)
    hi[axis] = slice(1, n + 1)
    veldiv = velface[tuple(hi)] - velface[tuple(lo)]
    return c - dt / dx * (flux[tuple(hi)] - flux[tuple(lo)] - c * veldiv)


def _diffuse(c, K, dt, dx, bg):
    g = np.pad(c, [(0, 0)] * (c.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    lap = (g[..., 2:, 1:-1] + g[..., :-2, 1:-1] + g[..., 1:-1, 2:] + g[..., 1:-1, :-2]
           - 4 * c) / dx**2
    return c + dt * K * lap


def _faces(field, axis):
    inner = 0.5 * (np.take(field, range(field.shape[axis] - 1), axis=axis)
                   + np.take(field, range(1, field.shape[axis]), axis=axis))
    first = np.take(field, [0], axis=axis)
    last = np.take(field, [-1], axis=axis)
    return np.concatenate([first, inner, last], axis=axis)



def run(fields, dx, emis, lam, K, bg, c0, samplecells=None, keephourly=False):
    uall, vall, hall = fields["u10"], fields["v10"], fields["blh"]
    nhours = uall.shape[0] - 1
    lam = np.asarray(lam, dtype=float)[:, None, None]
    bg = np.asarray(bg, dtype=float)
    c = np.array(c0, dtype=float)

    samples = [] if samplecells is not None else None
    hourly = [] if keephourly else None

    for i in range(nhours):
        speedmax = max(np.abs(uall[i:i + 2]).max(), np.abs(vall[i:i + 2]).max(), 1e-6)
        dtadv = CFL * dx / speedmax
        dtdif = 0.2 * dx**2 / max(K, 1e-9)
        nsub = int(np.ceil(3600.0 / min(dtadv, dtdif)))
        dt = 3600.0 / nsub

        dhdt = (hall[i + 1] - hall[i]) / 3600.0
        acc = np.zeros_like(c)
        for k in range(nsub):
            w = (k + 0.5) / nsub
            u = (1 - w) * uall[i] + w * uall[i + 1]
            v = (1 - w) * vall[i] + w * vall[i + 1]
            h = (1 - w) * hall[i] + w * hall[i + 1]
            c = _advect_1d(c, _faces(u, axis=1)[None], dt, dx, bg, axis=2)
            c = _advect_1d(c, _faces(-v, axis=0)[None], dt, dx, bg, axis=1)
            if K > 0:
                c = _diffuse(c, K, dt, dx, bg)
            we_h = np.maximum(dhdt, 0.0) / h
            krate = lam + we_h[None]
            src = emis / h[None] + we_h[None] * bg[:, None, None]
            decay = np.exp(-krate * dt)
            c_eq = np.where(krate > 0, src / np.where(krate > 0, krate, 1), 0.0)
            c = np.where(krate > 0, c_eq + (c - c_eq) * decay, c + src * dt)

            acc += c
        meanc = acc / nsub
        if samples is not None:
            samples.append(meanc[:, samplecells[0], samplecells[1]])
        if hourly is not None:
            hourly.append(meanc.astype(np.float32))

    result = {"final": c}
    if samples is not None:
        result["samples"] = np.stack(samples)
    if hourly is not None:
        result["hourly"] = np.stack(hourly)
    return result
