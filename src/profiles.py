"""Input profiles: turn real-world numbers into the 24-hour curves the model needs.

Every parameter comes from a public source (see the data table in docs/report.md).
They are *calibrations* for modelling, not a precise set of accounts -- conclusions hold
within +/-30% of these values (see the sensitivity and robustness sweeps).
"""
import numpy as np

# ---------------------------------------------------------------- calibrated inputs
# Industrial time-of-use tariff, peak-to-valley ratio about 3x
# (valley 0.34 / flat 0.66 / peak 1.08 CNY per kWh)
PRICE_24 = (
    [0.34] * 8          # 00-07 valley
    + [0.66] * 3        # 08-10 flat
    + [1.08] * 3        # 11-13 peak (air-conditioning + industry)
    + [0.66] * 4        # 14-17 flat
    + [1.08] * 4        # 18-21 peak (evening)
    + [0.66] * 2        # 22-23 flat
)

# Scenario B: Shandong-style duck curve -- PV oversupply makes midday a valley
PRICE_24_SHANDONG = (
    [0.34] * 8 + [0.66] * 3 + [0.26] * 3 + [0.66] * 4 + [1.08] * 4 + [0.50] * 2
)

# Grid carbon intensity kgCO2/MWh: coal baseload at night, PV pushes noon down,
# coal ramps back up for the evening peak (the dirtiest hours of the day)
CARBON_24 = (
    [740.0] * 8
    + [600.0] * 3
    + [450.0] * 3
    + [520.0] * 4
    + [760.0] * 4
    + [700.0] * 2
)

# Default parameters (MW / MWh / CNY)
PARAMS = dict(
    T=96,              # 15-minute slots, 96 per day
    dt=0.25,           # hours
    PUE=1.15,          # liquid-cooled AI hall, PUE 1.1-1.2
    Xmax=8.0,          # deferrable compute peak power (IT side)
    W=100.0,           # deferrable energy demand per day (IT side, MWh)
    Gmax=15.0,         # transformer capacity (grid side, MW)
    Emax=20.0,         # storage capacity MWh
    Bp=5.0,            # storage power MW
    eta=0.95,          # one-way efficiency, ~90% round trip
    E0=10.0,           # initial / final state of charge (MWh)
    d_deg=50.0,        # battery wear CNY/MWh throughput
    S_peak=8.0,        # on-site PV peak MW
    deadline_hour=20.0,
    scenario="A",
)


def _hour_lookup(vals24, T, dt, start=0.0):
    """Map 24 hourly values onto T slots (step hold, no interpolation, so peaks stay sharp)."""
    hours = start + np.arange(T) * dt
    idx = np.clip(np.floor(hours).astype(int) % 24, 0, 23)
    return np.asarray(vals24, dtype=float)[idx]


def solar_curve(T, dt, peak=8.0, sunrise=6.2, sunset=18.8, start=0.0):
    """Clear-sky PV: a sine bell between sunrise and sunset."""
    h = start + np.arange(T) * dt
    x = (h - sunrise) / (sunset - sunrise)
    shape = np.clip(np.sin(np.pi * x), 0.0, None) ** 1.5
    return peak * shape


def rigid_load(T, dt, start=0.0):
    """Rigid load (online inference): follows user traffic, highest in the evening."""
    h = start + np.arange(T) * dt
    g = lambda mu, sig: np.exp(-0.5 * ((h - mu) / sig) ** 2)
    return 2.0 + 0.9 * g(9.5, 2.0) + 1.1 * g(13.5, 2.5) + 2.6 * g(20.0, 2.0)


def build_instance(**overrides):
    p = dict(PARAMS)
    p.update(overrides)
    T, dt, start = int(p["T"]), float(p["dt"]), float(p.get("start_hour", 0.0))
    price24 = PRICE_24_SHANDONG if p["scenario"] == "B" else PRICE_24
    carbon24 = p.get("carbon24", CARBON_24)
    inst = dict(
        T=T, dt=dt,
        # CNY/kWh -> CNY/MWh (the model works in MW and MWh)
        prices=_hour_lookup(price24, T, dt, start) * 1000.0,
        carbon=_hour_lookup(carbon24, T, dt, start),
        solar=solar_curve(T, dt, p["S_peak"], start=start),
        L_rigid=rigid_load(T, dt, start),
        PUE=p["PUE"], Xmax=p["Xmax"], W=p["W"], Gmax=p["Gmax"],
        Emax=p["Emax"], Bp=p["Bp"], eta=p["eta"], E0=p["E0"],
        d_deg=p["d_deg"], S_peak=p["S_peak"],
        deadline_idx=int(round(p["deadline_hour"] / dt)),
        params=p,
    )
    return inst


def summaries(inst):
    """A few profile numbers for the report."""
    dt = inst["dt"]
    return dict(
        rigid_MWh=float(inst["L_rigid"].sum() * dt),
        batch_MWh=float(inst["W"]),
        solar_MWh=float(inst["solar"].sum() * dt),
        solar_peak_MW=float(inst["solar"].max()),
        price_min=float(inst["prices"].min()),
        price_max=float(inst["prices"].max()),
        carbon_min=float(inst["carbon"].min()),
        carbon_max=float(inst["carbon"].max()),
    )
