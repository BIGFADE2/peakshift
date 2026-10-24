"""Core model: one day of compute-power co-scheduling, written as a linear program.

Decisions (one set per 15-minute slot):
  g[t]     grid import                 (MW)
  x[t]     deferrable compute power    (MW, IT side)
  bc[t]    battery charging power      (MW)
  bd[t]    battery discharging power   (MW)
  s[t]     PV actually used            (MW)
  E[t]     state of charge at end of slot (MWh)

Objective: min  electricity + carbon cost + battery wear (+ a small smoothing term)
Constraints: power balance / work total and deadline / storage dynamics and capacity /
             transformer limit / PV cannot be over-drawn
"""
import _deps
_deps.ensure()

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

IDX = dict(g=0, x=1, bc=2, bd=3, s=4, E=5, u=6, uc=7)
NAMES = ["g", "x", "bc", "bd", "s", "E", "u", "uc"]


def _blocks(T):
    return {k: i * T for i, k in enumerate(NAMES)}


def solve(inst, objective="cost", lam=0.0, carbon_cap=None,
          shift=True, battery=True, smoothing=20.0, method="highs",
          demand_charge=None):
    """Solve the scheduling LP once.

    objective     : 'cost' (minimise money) or 'carbon' (minimise emissions)
    lam           : carbon price in CNY/kgCO2, folded into the objective
    carbon_cap    : emission cap in kg, for an epsilon-constraint frontier
    demand_charge : proxy demand charge in CNY/MW/day; adds a daily peak variable
    """
    T, dt = inst["T"], inst["dt"]
    p, e = inst["prices"], inst["carbon"]
    L, sol = inst["L_rigid"], inst["solar"]
    PUE, Gmax = inst["PUE"], inst["Gmax"]
    W, Xmax = inst["W"], inst["Xmax"]
    Emax, Bp, eta, E0 = inst["Emax"], inst["Bp"], inst["eta"], inst["E0"]
    d_deg, dl = inst["d_deg"], inst["deadline_idx"]

    b = _blocks(T)
    nvar = 8 * T + (1 if demand_charge else 0)
    pk = nvar - 1
    c = np.zeros(nvar)

    if objective == "cost":
        c[b["g"]:b["g"] + T] = (p + lam * e) * dt
        c[b["bc"]:b["bc"] + T] = d_deg * dt
        c[b["bd"]:b["bd"] + T] = d_deg * dt
        c[b["u"]:b["u"] + T] = smoothing * dt
        c[b["uc"]:b["uc"] + T] = smoothing * dt
    elif objective == "carbon":
        c[b["g"]:b["g"] + T] = e * dt
        c[b["g"]:b["g"] + T] += 1e-4 * p * dt      # tie-break: prefer the cheaper plan
    else:
        raise ValueError(objective)

    rows, cols, vals = [], [], []
    r_eq, rhs_eq = [], []

    def add(r, k, t, v):
        rows.append(r)
        cols.append(b[k] + t)
        vals.append(v)

    r = 0
    # (1) power balance per slot: grid + PV + discharge = facility + charging
    for t in range(T):
        add(r, "g", t, 1.0)
        add(r, "s", t, 1.0)
        add(r, "bd", t, 1.0)
        add(r, "bc", t, -1.0)
        add(r, "x", t, -PUE)
        r_eq.append(r)
        rhs_eq.append(PUE * L[t])
        r += 1

    # (2) deferrable work must all be delivered
    for t in range(T):
        add(r, "x", t, dt)
    r_eq.append(r)
    rhs_eq.append(W)
    r += 1

    # (3) state-of-charge recursion
    for t in range(T):
        add(r, "E", t, 1.0)
        add(r, "bc", t, -eta * dt)
        add(r, "bd", t, dt / eta)
        if t > 0:
            add(r, "E", t - 1, -1.0)
            rhs = 0.0
        else:
            rhs = E0
        r_eq.append(r)
        rhs_eq.append(rhs)
        r += 1

    # (4) end the day at the starting charge (no free energy)
    add(r, "E", T - 1, 1.0)
    r_eq.append(r)
    rhs_eq.append(E0)
    r += 1

    A_eq = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    b_eq = np.array(rhs_eq)

    # (5) inequalities: power smoothing, optional carbon cap
    ur, uc, uv, ub_rhs = [], [], [], []
    nrow = 0
    for t in range(1, T):
        # x_t - x_{t-1} - u_t <= 0
        ur += [nrow, nrow, nrow]; uc += [b["x"] + t, b["x"] + t - 1, b["u"] + t]
        uv += [1.0, -1.0, -1.0]; ub_rhs.append(0.0); nrow += 1
        # x_{t-1} - x_t - u_t <= 0
        ur += [nrow, nrow, nrow]; uc += [b["x"] + t - 1, b["x"] + t, b["u"] + t]
        uv += [1.0, -1.0, -1.0]; ub_rhs.append(0.0); nrow += 1
        # charging smoothing: without it the LP's multiple optima give a spiky profile
        ur += [nrow, nrow, nrow]; uc += [b["bc"] + t, b["bc"] + t - 1, b["uc"] + t]
        uv += [1.0, -1.0, -1.0]; ub_rhs.append(0.0); nrow += 1
        ur += [nrow, nrow, nrow]; uc += [b["bc"] + t - 1, b["bc"] + t, b["uc"] + t]
        uv += [1.0, -1.0, -1.0]; ub_rhs.append(0.0); nrow += 1

    if carbon_cap is not None:
        for t in range(T):
            ur.append(nrow); uc.append(b["g"] + t); uv.append(e[t] * dt)
        ub_rhs.append(carbon_cap)
        nrow += 1

    # demand charge: Pk >= g_t, and Pk is priced in the objective => peak shaving
    if demand_charge:
        for t in range(T):
            ur += [nrow, nrow]
            uc += [b["g"] + t, pk]
            uv += [1.0, -1.0]
            ub_rhs.append(0.0)
            nrow += 1
        c[pk] = demand_charge

    if nrow:
        A_ub = coo_matrix((uv, (ur, uc)), shape=(nrow, nvar)).tocsr()
        b_ub = np.array(ub_rhs)
    else:
        A_ub, b_ub = None, None

    # variable bounds
    lb = np.zeros(nvar)
    ubv = np.zeros(nvar)
    ubv[b["g"]:b["g"] + T] = Gmax
    ubv[b["s"]:b["s"] + T] = sol
    ubv[b["E"]:b["E"] + T] = Emax
    ubv[b["u"]:b["u"] + T] = np.inf
    ubv[b["uc"]:b["uc"] + T] = np.inf
    if shift:
        ubv[b["x"]:b["x"] + T] = Xmax
        ubv[b["x"] + dl:b["x"] + T] = 0.0          # nothing after the deadline
    else:
        fixed = W / (T * dt)
        lb[b["x"]:b["x"] + T] = fixed
        ubv[b["x"]:b["x"] + T] = fixed
    if battery:
        ubv[b["bc"]:b["bc"] + T] = Bp
        ubv[b["bd"]:b["bd"] + T] = Bp
    else:
        lb[b["bc"]:b["bc"] + T] = 0.0
        ubv[b["bc"]:b["bc"] + T] = 0.0
        lb[b["bd"]:b["bd"] + T] = 0.0
        ubv[b["bd"]:b["bd"] + T] = 0.0
        lb[b["E"]:b["E"] + T] = E0
        ubv[b["E"]:b["E"] + T] = E0

    bounds = list(zip(lb, ubv))
    if demand_charge:
        bounds[pk] = (0.0, Gmax)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=method)
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    z = res.x
    out = {k: z[b[k]:b[k] + T] for k in b}
    out["obj"] = float(res.fun)
    out["status"] = res.message
    out["metrics"] = evaluate(inst, out, demand_charge)
    return out


def evaluate(inst, out, demand_charge=None):
    """Recompute the metrics from a schedule, independently of the solver."""
    dt, p, e = inst["dt"], inst["prices"], inst["carbon"]
    g, x, bc, bd, s = out["g"], out["x"], out["bc"], out["bd"], out["s"]
    grid_MWh = float(g.sum() * dt)
    cost = float((p * g).sum() * dt)
    carbon = float((e * g).sum() * dt)
    degr = float(inst["d_deg"] * (bc + bd).sum() * dt)
    solar_avail = float(inst["solar"].sum() * dt)
    solar_used = float((s * dt).sum())
    peak = float(g.max())
    dem = demand_charge * peak if demand_charge else 0.0
    return dict(
        cost=cost,                # CNY
        carbon=carbon,            # kg
        demand=dem,               # demand-charge proxy, CNY
        dcost=cost + degr + dem,  # total
        degrad=degr,
        grid_MWh=grid_MWh,
        peak_grid=peak,
        solar_avail=solar_avail,
        solar_used=solar_used,
        solar_rate=solar_used / solar_avail if solar_avail > 0 else 0.0,
        batch_MWh=float(x.sum() * dt),
        cycles=float(bd.sum() * dt / inst["Emax"]) if inst["Emax"] > 0 else 0.0,
        cost_per_MWh=cost / max(grid_MWh, 1e-9),   # CNY per purchased MWh
        carbon_intensity=carbon / max(grid_MWh, 1e-9),
    )


def baseline(inst, shift=False, battery=False):
    """"No scheduling": work spread evenly over arrival, battery idle, PV used as it comes."""
    return solve(inst, shift=shift, battery=battery)


def evaluate_under(inst, plan, solar_real):
    """Given a fixed schedule, recompute cost and emissions under the realised PV."""
    dt = inst["dt"]
    x, bc, bd = plan["x"], plan["bc"], plan["bd"]
    D = inst["PUE"] * (inst["L_rigid"] + x)
    resid = D + bc - bd
    s_used = np.minimum(solar_real, np.maximum(resid, 0.0))
    g = np.maximum(resid - s_used, 0.0)
    cost = float((inst["prices"] * g).sum() * dt) + float(inst["d_deg"] * (bc + bd).sum() * dt)
    carbon = float((inst["carbon"] * g).sum() * dt)
    return dict(cost=cost, carbon=carbon, peak=float(g.max()))


def pareto(inst, lams=None):
    """Weighted-sum sweep of the cost-carbon frontier (exact for an LP: the feasible set is convex)."""
    if lams is None:
        lams = np.concatenate([[0.0], np.logspace(-3, 1.2, 30)])
    pts = []
    for lam in lams:
        out = solve(inst, objective="cost", lam=float(lam))
        m = out["metrics"]
        pts.append(dict(lam=float(lam), cost=m["dcost"], carbon=m["carbon"],
                        peak=m["peak_grid"], solar_rate=m["solar_rate"]))
    return pts
