# transport.py
"""
Reduced-order 2D advection-diffusion model for PM2.5 on a city domain.

The state is the column-mean concentration C (ug/m3) of a well-mixed
boundary layer of depth h(x, y, t):

    dC/dt + u . grad C = K lap C + E / h - lam C + (w_e / h) (C_bg - C)

Advection is in ADVECTIVE form (u . grad C), not flux form (div(u C)).
The two differ by C div(u). The interpolated ERA5 10 m wind is divergent
(up to ~1e-4 s-1), and in flux form that term spuriously concentrates or
dilutes even a uniform field, by tens of percent per hour. Physically,
low-level convergence is balanced by vertical motion that carries air of
the same well-mixed concentration, so a uniform field must stay uniform.
The scheme computes TVD fluxes and then adds back C div(u) per axis.
For a uniform field the two cancel exactly.

    u, v   10 m wind from ERA5 (hourly, bilinear in space, linear in time)
    h      ERA5 boundary-layer height, floored at H_MIN_M
    E      surface emission flux (ug m-2 s-1)
    K      horizontal eddy diffusivity (m2 s-1)          -- calibrated
    lam    first-order loss rate (deposition etc., s-1)  -- calibrated
    w_e    entrainment velocity max(dh/dt, 0): a growing boundary layer
           mixes in background air from above
    C_bg   background (regional) concentration, also used at inflow edges

The equation is linear in (E, C_bg) with the initial state set to C_bg,
so a run with several "tracers" (unit emissions, unit background, one per
loss rate) gives every combination calibration needs from one pass.

Numerics: finite volumes on the 2 km domain raster. Advection is
dimensionally split with a second-order TVD scheme (MC limiter), so that
numerical diffusion stays well below the physical K being calibrated. A
first-order upwind scheme would add ~u*dx/2 ~ 3000 m2/s of numerical
diffusion at 3 m/s, swamping K. Diffusion is explicit. Sources, loss and
entrainment are integrated exactly over each step (exponential update).
Time step: the largest dt with Courant number <= CFL and diffusion
number <= 0.2, so that each hour divides into a whole number of steps.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

HMIN = 50.0        # floor on boundary-layer height (ERA5 nights reach ~10 m)
CFL = 0.8


# ---------------------------------------------------------
# METEOROLOGY ON THE DOMAIN
# ---------------------------------------------------------

def load_met(city, phase2dir, utcstart, utcend):
    """Hourly u10/v10/blh (UTC) from Phase 2's processed yearly files."""
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
    """Interpolate ERA5 fields onto domain cell centres.

    Bilinear in (lat, lon). Cells outside the ERA5 point grid take the
    nearest edge value (Singapore's ERA5 box has only 2 x 2 points, so its
    western third is extrapolated this way).
    Returns arrays shaped (n_hours, ny, nx).
    """
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


# ---------------------------------------------------------
# NUMERICS
# ---------------------------------------------------------

def _mc_limited_slope(c, axis):
    """Monotonized-central limited slope along `axis` (zero at the edges)."""
    dfwd = np.diff(c, axis=axis, append=np.take(c, [-1], axis=axis))
    dbwd = np.diff(c, axis=axis, prepend=np.take(c, [0], axis=axis))
    dcen = 0.5 * (dfwd + dbwd)
    same = np.sign(dfwd) == np.sign(dbwd)
    lim = np.minimum(np.minimum(2 * np.abs(dfwd), 2 * np.abs(dbwd)), np.abs(dcen))
    return np.where(same, np.sign(dcen) * lim, 0.0)


def _advect_1d(c, velface, dt, dx, bg, axis):
    """One TVD advection step along `axis` (last two axes are y, x).

    vel_face holds velocities on the n+1 faces along `axis` (positive
    toward increasing index). Ghost cells beyond the edges hold the
    background value, so inflow brings in background air and outflow
    leaves freely.
    """
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
    left[axis] = slice(1, n + 2)      # cell to the left of each face
    right[axis] = slice(2, n + 3)     # cell to the right of each face
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
    # Flux divergence plus the C div(u) correction -> advective form.
    veldiv = velface[tuple(hi)] - velface[tuple(lo)]
    return c - dt / dx * (flux[tuple(hi)] - flux[tuple(lo)] - c * veldiv)


def _diffuse(c, K, dt, dx, bg):
    g = np.pad(c, [(0, 0)] * (c.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    lap = (g[..., 2:, 1:-1] + g[..., :-2, 1:-1] + g[..., 1:-1, 2:] + g[..., 1:-1, :-2]
           - 4 * c) / dx**2
    return c + dt * K * lap


def _faces(field, axis):
    """Cell-centred field -> face values along axis (edges copied)."""
    inner = 0.5 * (np.take(field, range(field.shape[axis] - 1), axis=axis)
                   + np.take(field, range(1, field.shape[axis]), axis=axis))
    first = np.take(field, [0], axis=axis)
    last = np.take(field, [-1], axis=axis)
    return np.concatenate([first, inner, last], axis=axis)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

def run(fields, dx, emis, lam, K, bg, c0, samplecells=None, keephourly=False):
    """Integrate the model over all hours in `fields`.

    emis   (ntr, ny, nx) emission flux per tracer, ug m-2 s-1
    lam    (ntr,) loss rate per tracer, s-1
    K      scalar diffusivity, m2 s-1
    bg     (ntr,) background concentration per tracer, ug m-3
    c0     (ntr, ny, nx) initial state
    sample_cells (rows, cols): if given, return hour-mean concentrations at
           those cells, shape (n_hours - 1, ntr, n_cells); hour i is the
           mean over [t_i, t_i + 1 h).
    store_hourly: also return the hour-mean full fields
           (n_hours - 1, ntr, ny, nx).

    The number of hours integrated is len(fields["u10"]) - 1 (fields are
    instants at the start and end of each hour).
    """
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

            # x: columns, positive east. y: rows run north -> south, so
            # the "positive index" velocity along rows is -v.
            c = _advect_1d(c, _faces(u, axis=1)[None], dt, dx, bg, axis=2)
            c = _advect_1d(c, _faces(-v, axis=0)[None], dt, dx, bg, axis=1)
            if K > 0:
                c = _diffuse(c, K, dt, dx, bg)

            # Sources, loss and entrainment, exact over dt:
            #   dC/dt = S - k C,  S = E/h + w_e C_bg / h,  k = lam + w_e / h
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
