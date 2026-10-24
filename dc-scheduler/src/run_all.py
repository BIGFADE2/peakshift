"""One command runs everything: instance -> four plans -> Pareto -> sensitivity
-> robustness -> independent checks -> figures and tables."""
import os
import time
import numpy as np
import pandas as pd

import _deps
_deps.ensure()

import profiles
import optimizer
import verify
import plots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
np.random.seed(42)

t0 = time.perf_counter()
log = []


def say(s=""):
    print(s)
    log.append(s)


say("=" * 78)
say("HackSHSID 2026 / Track 2 -- AI data center compute-power co-scheduling")
say("=" * 78)

# ------------------------------------------------------------------ 1. instance
inst = profiles.build_instance()
prof = profiles.summaries(inst)
say("\n[1] Instance (parameters calibrated from public sources, see docs/report.md)")
for k, v in prof.items():
    say(f"    {k:24s} {v:10.2f}")

# ------------------------------------------------------------------ 2. plans
say("\n[2] Plan comparison")
cases = [
    ("No scheduling", dict(shift=False, battery=False)),
    ("Shift only", dict(shift=True, battery=False)),
    ("Storage only", dict(shift=False, battery=True)),
    ("Full plan", dict(shift=True, battery=True)),
]
res = {}
for name, kw in cases:
    res[name] = optimizer.solve(inst, **kw)
res["Low-carbon"] = optimizer.solve(inst, shift=True, battery=True, objective="carbon")

base = res["No scheduling"]["metrics"]
rows = []
say(f"    {'plan':<14}{'cost/day':>12}{'CO2/day':>12}{'peak MW':>10}"
    f"{'CNY/MWh':>10}{'gCO2/kWh':>11}{'save %':>9}{'CO2 %':>8}")
for name, _ in cases + [("Low-carbon", None)]:
    m = res[name]["metrics"]
    rows.append(dict(plan=name, daily_cost=m["dcost"], energy_cost=m["cost"],
                     battery_degrad=m["degrad"], daily_emissions=m["carbon"],
                     peak_grid_MW=m["peak_grid"], grid_MWh=m["grid_MWh"],
                     pv_utilization=m["solar_rate"], battery_cycles=m["cycles"],
                     cost_per_MWh=m["cost_per_MWh"], carbon_g_per_kWh=m["carbon_intensity"],
                     cost_saving=1 - m["dcost"] / base["dcost"],
                     carbon_reduction=1 - m["carbon"] / base["carbon"]))
    say(f"    {name:<14}{m['dcost']:12,.0f}{m['carbon']:12,.0f}{m['peak_grid']:10.2f}"
        f"{m['cost_per_MWh']:10,.0f}{m['carbon_intensity']:11,.0f}"
        f"{(1-m['dcost']/base['dcost'])*100:9.1f}{(1-m['carbon']/base['carbon'])*100:8.1f}")
pd.DataFrame(rows).to_csv(os.path.join(TAB, "summary.csv"), index=False, encoding="utf-8-sig")

full = res["Full plan"]
valley = inst["prices"] < 500                      # valley tier (0.34 CNY/kWh)
stats = dict(cost_save=1 - full["metrics"]["dcost"] / base["dcost"],
             carbon_save=1 - full["metrics"]["carbon"] / base["carbon"],
             solar_rate=full["metrics"]["solar_rate"],
             valley_share=float(full["g"][valley].sum() / full["g"].sum()),
             valley_share_base=float(res["No scheduling"]["g"][valley].sum()
                                     / res["No scheduling"]["g"].sum()))

sched = pd.DataFrame({
    "hour": np.arange(inst["T"]) * inst["dt"],
    "price_CNY_per_MWh": inst["prices"], "carbon_kg_per_MWh": inst["carbon"],
    "rigid_load_MW": inst["L_rigid"], "pv_available_MW": inst["solar"],
    "deferrable_MW": full["x"], "grid_import_MW": full["g"], "pv_used_MW": full["s"],
    "charge_MW": full["bc"], "discharge_MW": full["bd"], "soc_MWh": full["E"],
    "facility_demand_MW": inst["PUE"] * (inst["L_rigid"] + full["x"]),
})
sched.to_csv(os.path.join(TAB, "schedule_full.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 3. Pareto
say("\n[3] Cost-carbon Pareto frontier (weighted sum, 40 carbon prices)")
pts = optimizer.pareto(inst)
pd.DataFrame(pts).to_csv(os.path.join(TAB, "pareto.csv"), index=False, encoding="utf-8-sig")
cost_pt = dict(cost=full["metrics"]["dcost"], carbon=full["metrics"]["carbon"])
carbon_pt = dict(cost=res["Low-carbon"]["metrics"]["dcost"],
                 carbon=res["Low-carbon"]["metrics"]["carbon"])
naive_pt = dict(cost=base["dcost"], carbon=base["carbon"])

csts = np.array([p["cost"] for p in pts]); carb = np.array([p["carbon"] for p in pts])
cn = (csts - csts.min()) / (csts.max() - csts.min() + 1e-9)
kn = (carb - carb.min()) / (carb.max() - carb.min() + 1e-9)
knee = pts[int(np.argmin(cn ** 2 + kn ** 2))]
say(f"    min cost   ({cost_pt['cost']:,.0f} CNY, {cost_pt['carbon']:,.0f} kg)")
say(f"    min carbon ({(csts.max()):,.0f} CNY, {carb.min():,.0f} kg)")
say(f"    knee point ({knee['cost']:,.0f} CNY, {knee['carbon']:,.0f} kg, "
    f"carbon price {knee['lam']:.3f} CNY/kg)")
say(f"    note: the cost-optimal set is a whole face (many equal-cost schedules); "
    f"the cleanest one reaches {carb[np.abs(csts - cost_pt['cost']) < 1.0].min():,.0f} kg "
    f"-- free abatement at the same price")

sac = []
for p in pts:
    dC = p["cost"] - cost_pt["cost"]
    dE = cost_pt["carbon"] - p["carbon"]
    if dE > 1:
        sac.append((p["lam"], dC / dE))   # CNY/kg
if sac:
    say(f"    marginal abatement cost on the frontier "
        f"{min(s[1] for s in sac):,.0f} ~ {max(s[1] for s in sac):,.0f} CNY/kg "
        f"= {min(s[1] for s in sac)*1000:,.0f} ~ {max(s[1] for s in sac)*1000:,.0f} CNY/tCO2")

# ------------------------------------------------------------------ 4. sensitivity
say("\n[4] Sensitivity analysis")
sens = {}


def sweep(key, values, maker):
    xs, cs, cd = [], [], []
    for v in values:
        i2 = maker(v)
        try:
            r = optimizer.solve(i2, shift=True, battery=True)
        except RuntimeError:
            xs.append(v); cs.append(np.nan); cd.append(np.nan)
            continue
        b2 = optimizer.solve(i2, shift=False, battery=False)
        xs.append(v)
        cs.append(1 - r["metrics"]["dcost"] / b2["metrics"]["dcost"])
        cd.append(1 - r["metrics"]["carbon"] / b2["metrics"]["carbon"])
    sens[key] = dict(x=np.array(xs, float), cost_save=np.array(cs), carbon_delta=np.array(cd))
    return sens[key]


d = sweep("battery", [0, 5, 10, 20, 30, 40],
          lambda v: profiles.build_instance(Emax=float(v), E0=min(0.5 * v, 10.0),
                                            Bp=min(5.0, max(0.1, v / 2))))
for x, a, b in zip(d["x"], d["cost_save"], d["carbon_delta"]):
    if np.isnan(a):
        say(f"    battery {x:>4.0f} MWh -> infeasible")
    else:
        say(f"    battery {x:>4.0f} MWh -> save {a*100:5.1f}%  carbon {b*100:+5.1f}%")

d = sweep("deadline", [12, 14, 16, 18, 20, 22],
          lambda v: profiles.build_instance(deadline_hour=float(v)))
for x, a, b in zip(d["x"], d["cost_save"], d["carbon_delta"]):
    if np.isnan(a):
        say(f"    deadline {x:>4.0f}:00 -> infeasible (work does not fit)")
    else:
        say(f"    deadline {x:>4.0f}:00 -> save {a*100:5.1f}%  carbon {b*100:+5.1f}%")

lams = [0, 20, 50, 100, 200, 500, 1000, 2000]
xs, cs, cd = [], [], []
for lt in lams:
    r = optimizer.solve(inst, shift=True, battery=True, lam=lt / 1e3)
    xs.append(lt)
    cs.append(1 - r["metrics"]["dcost"] / base["dcost"])
    cd.append(1 - r["metrics"]["carbon"] / base["carbon"])
sens["lam"] = dict(x=np.array(xs, float), cost_save=np.array(cs), carbon_delta=np.array(cd))
for x, a, b in zip(*[sens["lam"][k] for k in ("x", "cost_save", "carbon_delta")]):
    say(f"    carbon price {x:>5.0f} CNY/t -> save {a*100:5.1f}%  carbon {b*100:+5.1f}%")

d = sweep("solar", [0, 4, 8, 12, 16],
          lambda v: profiles.build_instance(S_peak=float(v)))
for x, a, b in zip(d["x"], d["cost_save"], d["carbon_delta"]):
    say(f"    PV peak {x:>4.0f} MW -> save {a*100:5.1f}%  carbon {b*100:+5.1f}%")


def night_profile(v):
    """Change only the 00-07 carbon intensity, keep the rest."""
    return [v] * 8 + [560.0] * 3 + [450.0] * 3 + [500.0] * 4 + [760.0] * 4 + [700.0] * 2


d = sweep("night", [400, 500, 600, 700, 800],
          lambda v: profiles.build_instance(carbon24=night_profile(v)))
for x, a, b in zip(d["x"], d["cost_save"], d["carbon_delta"]):
    say(f"    night carbon {x:>4.0f} -> save {a*100:5.1f}%  carbon {b*100:+5.1f}%")

frames = []
for k, v in sens.items():
    d = pd.DataFrame({"x": v["x"], "cost_save": v["cost_save"], "carbon_delta": v["carbon_delta"]})
    d["param"] = k
    frames.append(d)
pd.concat(frames, ignore_index=True).to_csv(os.path.join(TAB, "sensitivity.csv"),
                                            index=False, encoding="utf-8-sig")

say("\n    Demand-charge (peak shaving) experiment:")
dem = []
for dc in [0, 300, 600, 1200, 2400]:
    r = optimizer.solve(inst, shift=True, battery=True, demand_charge=dc)
    m = r["metrics"]
    dem.append(dict(demand_charge_CNY_per_MW_day=dc, peak_grid_MW=m["peak_grid"],
                    daily_cost=m["dcost"], daily_emissions=m["carbon"]))
    say(f"    {dc:>5.0f} CNY/MW/day -> peak {m['peak_grid']:5.2f} MW  cost {m['dcost']:9,.0f}  "
        f"CO2 {m['carbon']:9,.0f}")
pd.DataFrame(dem).to_csv(os.path.join(TAB, "demand_charge.csv"), index=False,
                         encoding="utf-8-sig")

say("\n    Scenario B (Shandong-style midday valley, PV-rich grid):")
instB = profiles.build_instance(scenario="B")
rB = optimizer.solve(instB, shift=True, battery=True)
bB = optimizer.solve(instB, shift=False, battery=False)
say(f"    save {(1-rB['metrics']['dcost']/bB['metrics']['dcost'])*100:.1f}%  "
    f"carbon {(1-rB['metrics']['carbon']/bB['metrics']['carbon'])*100:+.1f}%")

say("\n    Scenario C (wind-rich nights, night carbon 740 -> 430):")
CLEAN_NIGHT = ([430.0] * 8 + [560.0] * 3 + [450.0] * 3 + [500.0] * 4
               + [760.0] * 4 + [700.0] * 2)
instC = profiles.build_instance(carbon24=CLEAN_NIGHT)
rC = optimizer.solve(instC, shift=True, battery=True)
bC = optimizer.solve(instC, shift=False, battery=False)
sC = optimizer.solve(instC, shift=False, battery=True)
say(f"    full plan   : save {(1-rC['metrics']['dcost']/bC['metrics']['dcost'])*100:.1f}%  "
    f"carbon {(1-rC['metrics']['carbon']/bC['metrics']['carbon'])*100:+.1f}%")
say(f"    storage only: save {(1-sC['metrics']['dcost']/bC['metrics']['dcost'])*100:.1f}%  "
    f"carbon {(1-sC['metrics']['carbon']/bC['metrics']['carbon'])*100:+.1f}%  "
    f"(scenario A storage-only: "
    f"{(1-res['Storage only']['metrics']['carbon']/base['carbon'])*100:+.1f}%)")

# ------------------------------------------------------------------ 5. robustness
say("\n[5] Robustness: PV forecast error (200 Monte Carlo draws, seed=42)")
rng = np.random.default_rng(42)
N, T = 200, inst["T"]
noise = rng.normal(0, 1, size=(N, T))
kern = np.ones(9) / 9
err = np.array([np.convolve(n, kern, mode="same") for n in noise]) * np.sqrt(9) * 0.12
err = np.clip(err, -0.40, 0.40)
solar_real = np.clip(inst["solar"][None, :] * (1 + err), 0, None)

base_plan = dict(x=np.full(T, inst["W"] / (T * inst["dt"])), bc=np.zeros(T), bd=np.zeros(T))
save, save_ideal = [], []
for i in range(N):
    sr = solar_real[i]
    c_plan = optimizer.evaluate_under(inst, full, sr)["cost"]
    c_base = optimizer.evaluate_under(inst, base_plan, sr)["cost"]
    save.append(1 - c_plan / c_base)
    if i < 60:                                  # perfect foresight: re-optimise on truth
        inst_i = dict(inst); inst_i["solar"] = sr
        r = optimizer.solve(inst_i, shift=True, battery=True)
        save_ideal.append(1 - r["metrics"]["dcost"] / c_base)
save = np.array(save); save_ideal = np.array(save_ideal)
say(f"    fixed schedule : mean {save.mean()*100:.1f}%  5th pct {np.percentile(save,5)*100:.1f}%"
    f"  worst {save.min()*100:.1f}%")
say(f"    perfect foresight: mean {save_ideal.mean()*100:.1f}% (upper bound)")
pd.DataFrame({"cost_saving": save}).to_csv(os.path.join(TAB, "robustness.csv"),
                                           index=False, encoding="utf-8-sig")
rob = dict(save=save, save_ideal=save_ideal)

# ------------------------------------------------------------------ 6. checks
say("\n[6] Independent verification")
checks = verify.run_all_checks(inst, full)
for name, ok, detail in checks:
    say(f"    [{'PASS' if ok else 'FAIL'}] {name:<28}{detail}")
n_pass = sum(1 for c in checks if c[1])
say(f"    total {n_pass}/{len(checks)} passed")
pd.DataFrame([dict(check=n, result="PASS" if o else "FAIL", evidence=d) for n, o, d in checks]) \
    .to_csv(os.path.join(TAB, "checks.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 7. figures
say("\n[7] Figures")
plots.fig_inputs(inst, os.path.join(FIG, "fig1_inputs.png"))
say("    fig1_inputs.png")
plots.fig_schedule(inst, full, res["No scheduling"], os.path.join(FIG, "fig2_schedule.png"))
say("    fig2_schedule.png")
plots.fig_compare([(k, res[k]["metrics"]) for k, _ in cases]
                  + [("Low-carbon", res["Low-carbon"]["metrics"])],
                  os.path.join(FIG, "fig3_compare.png"))
say("    fig3_compare.png")
plots.fig_pareto(pts, naive_pt, cost_pt, carbon_pt, knee, os.path.join(FIG, "fig4_pareto.png"))
say("    fig4_pareto.png")
plots.fig_sensitivity(sens, os.path.join(FIG, "fig5_sensitivity.png"))
say("    fig5_sensitivity.png")
plots.fig_robust(rob, os.path.join(FIG, "fig6_robustness.png"))
say("    fig6_robustness.png")
plots.fig_cover(inst, full, stats, os.path.join(ROOT, "cover.png"))
say("    cover.png")

# ------------------------------------------------------------------ 8. summary
say("\n[8] Summary")
say(f"    no scheduling: {base['dcost']:,.0f} CNY/day, {base['carbon']:,.0f} kgCO2/day")
say(f"    full plan    : {full['metrics']['dcost']:,.0f} CNY/day, "
    f"{full['metrics']['carbon']:,.0f} kgCO2/day")
say(f"    cost {stats['cost_save']*100:.1f}% down, carbon {stats['carbon_save']*100:+.1f}%")
say(f"    valley-price share of energy bought: {stats['valley_share']*100:.1f}% "
    f"(no scheduling: {stats['valley_share_base']*100:.1f}%)")
say(f"    runtime {time.perf_counter()-t0:.1f} s")

out_log = os.path.join(ROOT, "logs", "run_log.txt")
os.makedirs(os.path.dirname(out_log), exist_ok=True)
with open(out_log, "w", encoding="utf-8") as f:
    f.write("\n".join(log) + "\n")
say(f"\nlog written to {out_log}")
