"""Independent verification (Phase 2.5): re-check everything from the raw arrays,
without reusing any intermediate value produced by the solver.

Four kinds of checks:
  A. Feasibility  -- power balance, SOC recursion, energy total, deadline, device limits
  B. Cross-solver -- HiGHS dual simplex vs interior point must agree
  C. Heuristic    -- a hand-written "human dispatcher" rule must not beat the LP
  D. Brute force  -- discretise a 4-slot instance and enumerate every feasible schedule
"""
import itertools
import numpy as np

import _deps
_deps.ensure()

import optimizer
import profiles


def check_feasible(inst, out):
    T, dt = inst["T"], inst["dt"]
    g, x, bc, bd, s, E = out["g"], out["x"], out["bc"], out["bd"], out["s"], out["E"]
    res = []

    bal = g + s + bd - bc - inst["PUE"] * (inst["L_rigid"] + x)
    res.append(("Power balance residual (MW)", bool(np.max(np.abs(bal)) < 1e-6),
                f"{np.max(np.abs(bal)):.2e}"))

    soc = np.empty(T)
    for t in range(T):
        prev = inst["E0"] if t == 0 else soc[t - 1]
        soc[t] = prev + inst["eta"] * bc[t] * dt - bd[t] * dt / inst["eta"]
    soc_err = np.max(np.abs(soc - E))
    res.append(("SOC recursion residual (MWh)", bool(soc_err < 1e-6), f"{soc_err:.2e}"))

    res.append(("SOC within bounds",
                bool(soc.min() >= -1e-6 and soc.max() <= inst["Emax"] + 1e-6),
                f"[{soc.min():.2f}, {soc.max():.2f}]"))

    e_batch = float(x.sum() * dt)
    res.append(("Deferrable energy conserved (MWh)", bool(abs(e_batch - inst["W"]) < 1e-6),
                f"{e_batch:.4f} vs {inst['W']}"))

    after = np.abs(x[inst["deadline_idx"]:]).max() if inst["deadline_idx"] < T else 0.0
    res.append(("No compute after deadline", bool(after < 1e-9), f"max={after:.2e}"))

    res.append(("Grid import under transformer limit",
                bool(g.max() <= inst["Gmax"] + 1e-6), f"{g.max():.3f} MW"))
    res.append(("PV within availability", bool(np.all(s <= inst["solar"] + 1e-6)), "OK"))
    res.append(("Battery power within limits",
                bool(bc.max() <= inst["Bp"] + 1e-6 and bd.max() <= inst["Bp"] + 1e-6),
                f"max={max(bc.max(), bd.max()):.3f} MW"))
    return res


def cross_method(inst):
    """Solve the same problem with a different algorithm."""
    a = optimizer.solve(inst, method="highs-ds")["obj"]
    b = optimizer.solve(inst, method="highs-ipm")["obj"]
    gap = abs(a - b) / max(abs(a), 1e-9)
    return [("Two solvers agree", bool(gap < 1e-9),
             f"gap {gap:.2e} ({a:,.1f} vs {b:,.1f})")]


def greedy(inst):
    """Independent heuristic: fill the deferrable work by ascending price,
    ignoring storage and power smoothing."""
    T, dt = inst["T"], inst["dt"]
    x = np.zeros(T)
    left = inst["W"]
    order = np.argsort(inst["prices"], kind="stable")
    for t in order:
        if t >= inst["deadline_idx"]:
            continue
        head = inst["Gmax"] / inst["PUE"] - inst["L_rigid"][t]
        take = min(inst["Xmax"], max(head, 0.0), left / dt)
        x[t] = take
        left -= take * dt
        if left <= 1e-9:
            break
    if left > 1e-6:
        raise RuntimeError("greedy has no feasible schedule")
    D = inst["PUE"] * (inst["L_rigid"] + x)
    s = np.minimum(inst["solar"], D)
    g = D - s
    return dict(g=g, x=x, bc=np.zeros(T), bd=np.zeros(T), s=s, E=np.full(T, inst["E0"]))


def brute_force(T=4, dt=1.0, start_hour=17.0, W=6.0, Xmax=4.0, step=0.2):
    """Enumerate every feasible schedule on a 4-slot, 0.2 MW grid."""
    inst = profiles.build_instance(
        T=T, dt=dt, start_hour=start_hour, W=W, Xmax=Xmax,
        Emax=0.1, E0=0.0, deadline_hour=T * dt, d_deg=0.0, Gmax=15.0)
    levels = np.round(np.arange(0.0, Xmax + 1e-9, step), 4)
    PUE, Gmax = inst["PUE"], inst["Gmax"]
    best, best_x = np.inf, None
    for combo in itertools.product(levels, repeat=T):
        if abs(sum(combo) * dt - W) > 1e-9:
            continue
        x = np.array(combo)
        D = PUE * (inst["L_rigid"] + x)
        s = np.minimum(inst["solar"], D)
        g = D - s
        if g.max() > Gmax + 1e-9:
            continue
        c = float((inst["prices"] * g).sum() * dt)
        if c < best:
            best, best_x = c, x.copy()

    lp = optimizer.solve(inst, shift=True, battery=False, smoothing=0.0)
    lp_cost = float((inst["prices"] * lp["g"]).sum() * dt)
    gap = (best - lp_cost) / max(abs(best), 1e-9)
    ok = bool(lp_cost <= best + 1e-6 and gap < 0.02)
    return [("Brute force (4 slots x 21 levels)", ok,
             f"LP={lp_cost:,.2f} / brute={best:,.2f} CNY, LP lower by {gap*100:.2f}%, "
             f"{len(levels)**T:,} combinations")]


def run_all_checks(inst, out):
    checks = []
    checks += check_feasible(inst, out)
    checks += cross_method(inst)
    checks += brute_force()
    gm = greedy(inst)
    lp_cost = optimizer.evaluate(inst, out)["dcost"]
    gc = optimizer.evaluate(inst, gm)["dcost"]
    checks.append(("Greedy heuristic not better than LP", bool(lp_cost <= gc + 1e-3),
                   f"LP {lp_cost:,.0f} <= greedy {gc:,.0f} CNY (gap {gc-lp_cost:,.0f})"))
    return checks


if __name__ == "__main__":
    inst = profiles.build_instance()
    out = optimizer.solve(inst, shift=True, battery=True)
    rows = run_all_checks(inst, out)
    n_pass = sum(1 for r in rows if r[1])
    print(f"{'check':<40}{'result':<8}evidence")
    for name, ok, detail in rows:
        print(f"{name:<40}{'PASS' if ok else 'FAIL':<8}{detail}")
    print(f"\n{n_pass}/{len(rows)} passed")
