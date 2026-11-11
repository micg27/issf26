# test_transport.py
"""
Physics checks for transport.py: each case has an exact answer.

Usage:
    python test_transport.py
"""

import numpy as np

import transport as T

NY, NX, DX = 60, 80, 2000.0
YY, XX = np.mgrid[:NY, :NX]


def fields(u, v, h, ninstants):
    shape = (ninstants, NY, NX)
    return {"u10": np.full(shape, u), "v10": np.full(shape, v), "blh": np.full(shape, h)}


def blob(row, col, sigmacells=3.0):
    return np.exp(-((XX - col) ** 2 + (YY - row) ** 2) / (2 * sigmacells**2))[None]


def zeros():
    return np.zeros((1, NY, NX))


def test_background_stays_uniform():
    f = fields(3.0, -2.0, 500.0, 7)
    f["blh"] = np.linspace(100, 1000, 7)[:, None, None] * np.ones((7, NY, NX))  # entrainment on
    c = T.run(f, DX, zeros(), [0.0], 300.0, [10.0], np.full((1, NY, NX), 10.0))["final"]
    assert np.abs(c - 10).max() < 1e-9


def test_uniform_field_stays_uniform_in_divergent_wind():
    # Wind speeding up eastward and turning: strongly divergent. A uniform
    # field must stay uniform (no spurious concentration/dilution).
    f = fields(0.0, 0.0, 500.0, 7)
    f["u10"] = 1.0 + 6.0 * XX[None] / NX * np.ones((7, 1, 1))
    f["v10"] = -3.0 + 4.0 * YY[None] / NY * np.ones((7, 1, 1))
    c = T.run(f, DX, zeros(), [0.0], 300.0, [10.0], np.full((1, NY, NX), 10.0))["final"]
    assert np.abs(c - 10).max() < 1e-9


def test_advection_east_conserves_mass_and_moves_correctly():
    c0 = blob(30, 15)
    c = T.run(fields(4.0, 0.0, 500.0, 7), DX, zeros(), [0.0], 0.0, [0.0], c0)["final"][0]
    assert abs(c.sum() / c0.sum() - 1) < 1e-6
    shift = (c * XX).sum() / c.sum() - 15
    assert abs(shift - 4.0 * 6 * 3600 / DX) < 0.05          # 43.2 cells
    assert c.max() > 0.9                                      # low numerical diffusion


def test_northward_wind_moves_toward_row_zero():
    c0 = blob(30, 40)
    c = T.run(fields(0.0, 4.0, 500.0, 2), DX, zeros(), [0.0], 0.0, [0.0], c0)["final"][0]
    assert abs((c * YY).sum() / c.sum() - (30 - 4.0 * 3600 / DX)) < 0.05   # row 22.8


def test_emission_loss_steady_state():
    E, lam, h = 1e-3, 1 / 3600.0, 500.0
    c = T.run(fields(0.0, 0.0, h, 25), DX, np.full((1, NY, NX), E), [lam], 0.0, [0.0], zeros())["final"]
    assert abs(c[0, 30, 40] / (E / (h * lam)) - 1) < 1e-6


def test_diffusion_variance_growth():
    K, hours = 500.0, 3
    c0 = blob(30, 40)
    c = T.run(fields(0.0, 0.0, 500.0, hours + 1), DX, zeros(), [0.0], K, [0.0], c0)["final"][0]
    cx = (c * XX).sum() / c.sum()
    var = (c * (XX - cx) ** 2).sum() / c.sum() * DX**2
    growth = var - 9 * DX**2
    assert abs(growth / (2 * K * hours * 3600) - 1) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
