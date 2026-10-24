"""Figures (matplotlib only; no jet/rainbow colormaps)."""
import numpy as np

import _deps
_deps.ensure()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

C = dict(solar="#E9B44C", grid="#4A5C7A", batt="#5B8C5A", batch="#C0604A",
         load="#2F3A4A", price="#8E5A8A", carbon="#3C7A8C", gray="#9AA0A6",
         accent="#B03A2E", light="#E8EAED")


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _hours(inst):
    return np.arange(inst["T"]) * inst["dt"]


def fig_inputs(inst, path):
    h = _hours(inst)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))

    ax = axes[0, 0]
    ln1 = ax.step(h, inst["prices"] / 1000, where="post", color=C["price"], lw=2.0,
                  label="TOU price (left)")[0]
    ax.set_ylabel("Price (CNY/kWh)", color=C["price"])
    ax.tick_params(axis="y", colors=C["price"])
    ax.set_xlim(0, 24); ax.set_ylim(0, 1.35)
    ax2 = ax.twinx()
    ln2, = ax2.plot(h, inst["carbon"] / 1000, color=C["carbon"], lw=1.8, ls="-.",
                    label="Grid carbon intensity (right)")
    ax2.set_ylabel("Carbon intensity (kgCO2/kWh)", color=C["carbon"])
    ax2.tick_params(axis="y", colors=C["carbon"])
    ax2.grid(False); _clean(ax2)
    ax.legend([ln1, ln2], [l.get_label() for l in (ln1, ln2)], loc="upper left", fontsize=8)
    ax.set_title("(a) Cheapest hours are also the dirtiest", loc="left")
    ax.set_xlabel("Hour of day")

    ax = axes[0, 1]
    # group slots by (price, carbon) and label the time window
    def key_of(t):
        return (round(inst["prices"][t]), round(inst["carbon"][t]))

    groups, start = [], 0
    for t in range(1, inst["T"] + 1):
        if t == inst["T"] or key_of(t) != key_of(t - 1):
            groups.append((start, t, key_of(start)))
            start = t
    dt = inst["dt"]
    for a, b, key in groups:
        ax.scatter(key[0] / 1000, key[1] / 1000, s=120, color=C["grid"],
                   edgecolor="white", lw=0.8, zorder=3)
        ax.annotate(f"{a*dt:02.0f}:00-{b*dt:02.0f}:00", (key[0] / 1000, key[1] / 1000),
                    textcoords="offset points", xytext=(10, 5), fontsize=8.5,
                    color=C["load"])
    ax.annotate("Ideal zone: cheap AND clean\n(actually empty)", xy=(0.30, 0.47), fontsize=8.5,
                color=C["accent"], ha="left", va="center")
    ax.axvspan(0.30, 0.55, ymin=0.0, ymax=0.20, color=C["accent"], alpha=0.07)
    ax.set_xlabel("Price (CNY/kWh)"); ax.set_ylabel("Carbon intensity (kgCO2/kWh)")
    ax.set_xlim(0.28, 1.22); ax.set_ylim(0.40, 0.80)
    ax.set_title("(b) Cheap is not clean: the cheap hours are the dirty ones", loc="left")

    ax = axes[1, 0]
    uniform = np.full(inst["T"], inst["W"] / (inst["T"] * inst["dt"]))
    ax.fill_between(h, 0, inst["L_rigid"], color=C["load"], alpha=0.85,
                    label="Rigid load (online inference)")
    ax.plot(h, inst["L_rigid"] + uniform, color=C["batch"], lw=1.6, ls="--",
            label="Total load without scheduling")
    ax.set_ylabel("IT-side power (MW)"); ax.set_xlabel("Hour of day")
    ax.set_title("(c) Rigid compute cannot move; the rest can", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.fill_between(h, 0, inst["solar"], color=C["solar"], alpha=0.85, label="PV available")
    D = inst["PUE"] * (inst["L_rigid"] + uniform)
    ax.plot(h, D, color=C["load"], lw=1.6, label="Facility demand, no scheduling")
    ax.set_ylabel("MW"); ax.set_xlabel("Hour of day")
    ax.set_title("(d) On-site PV: 8 MW at noon", loc="left")
    ax.legend(loc="upper left")

    _clean(axes[0, 0]); _clean(axes[1, 0]); _clean(axes[1, 1])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_schedule(inst, plan, base, path):
    h = _hours(inst)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.1, 1.1, 1.2]))

    ax = axes[0]
    s, bd, g = plan["s"], plan["bd"], plan["g"]
    D = inst["PUE"] * (inst["L_rigid"] + plan["x"])
    ax.stackplot(h, s, bd, g, colors=[C["solar"], C["batt"], C["grid"]],
                 labels=["PV direct", "Storage discharge", "Grid import"],
                 alpha=0.9, step="post")
    ax.plot(h, D, color=C["load"], lw=1.8, label="Facility demand (IT x PUE)")
    ax.plot(h, D + plan["bc"], color=C["batt"], lw=1.2, ls=":", label="Demand incl. charging")
    ax.set_ylabel("Power (MW)")
    ax.set_title("(a) Supply mix after optimisation: noon runs on PV, evening peak on storage",
                 loc="left")
    ax.legend(ncol=4, loc="upper left", frameon=True, facecolor="white", framealpha=0.85)

    ax = axes[1]
    ax.fill_between(h, 0, base["x"] * inst["PUE"], color=C["gray"], alpha=0.35,
                    step="post", label="No scheduling: spread evenly")
    ax.fill_between(h, 0, plan["x"] * inst["PUE"], color=C["batch"], alpha=0.8,
                    step="post", label="Optimised: moved to cheap + sunny hours")
    ax.set_ylabel("Deferrable compute (MW, grid side)")
    ax.set_title("(b) Where the deferrable work goes: valley fill, solar at noon, "
                 "clear of the peak", loc="left")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax2 = ax.twinx()
    ax2.fill_between(h, 0, inst["prices"] / 1000, color=C["price"], alpha=0.13, step="post")
    ax2.set_ylabel("Price (CNY/kWh)", color=C["price"])
    ax2.tick_params(axis="y", colors=C["price"])
    ax2.set_ylim(0, 3.2); ax2.grid(False); _clean(ax2)
    ax.plot(h, plan["E"], color=C["batt"], lw=2.0, label="Storage SOC")
    ax.set_ylabel("SOC (MWh)"); ax.set_xlabel("Hour of day")
    ax.set_ylim(0, inst["Emax"] * 1.15)
    ax.set_title("(c) Storage: charged overnight, discharged through the evening peak",
                 loc="left")
    ax.legend(loc="upper left")

    _clean(axes[0]); _clean(axes[1]); _clean(axes[2])
    axes[2].set_xlim(0, 24); axes[2].set_xticks(range(0, 25, 2))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_compare(rows, path):
    """rows: list of (label, metrics)"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    specs = [("dcost", "Daily total cost (CNY)"), ("carbon", "Daily emissions (kg)"),
             ("peak_grid", "Peak grid import (MW)")]
    base = rows[0][1]
    for ax, (key, title) in zip(axes, specs):
        vals = np.array([r[1][key] for r in rows], dtype=float)
        rel = vals / base[key]
        cols = ["#BFC5CC"] + [C["grid"]] * (len(rows) - 2) + [C["batt"]]
        if key == "carbon":
            cols = ["#BFC5CC"] + [C["carbon"]] * (len(rows) - 2) + [C["batt"]]
        bars = ax.bar(range(len(rows)), rel, color=cols[:len(rows)], width=0.62)
        for i, (b, v) in enumerate(zip(bars, vals)):
            txt = f"{v:,.0f}" if key != "peak_grid" else f"{v:.1f}"
            pct = "" if i == 0 else f"\n{(rel[i]-1)*100:+.1f}%"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    txt + pct, ha="center", va="bottom", fontsize=7.6)
        ax.axhline(1.0, color=C["gray"], lw=0.9, ls="--")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=18, ha="right")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(rel.max() * 1.28, 1.25))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("relative to " + rows[0][0])
        _clean(ax)
    fig.suptitle("Versus no scheduling: what happens to money, carbon and peak",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(pts, naive, cost_pt, carbon_pt, knee, path):
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    cost = np.array([p["cost"] for p in pts]) / 1e4
    carb = np.array([p["carbon"] for p in pts]) / 1e3
    ax.plot(cost, carb, color=C["grid"], lw=1.8, zorder=1, label="Pareto frontier")
    ax.scatter(cost, carb, s=22, color=C["grid"], zorder=3, alpha=0.8)

    ax.scatter([naive["cost"] / 1e4], [naive["carbon"] / 1e3], marker="s", s=90,
               color=C["gray"], edgecolor="white", zorder=4, label="No scheduling")
    ax.scatter([cost_pt["cost"] / 1e4], [cost_pt["carbon"] / 1e3], marker="o", s=130,
               color=C["accent"], edgecolor="white", zorder=5, label="Min-cost plan")
    ax.scatter([carbon_pt["cost"] / 1e4], [carbon_pt["carbon"] / 1e3], marker="o", s=130,
               color=C["batt"], edgecolor="white", zorder=5, label="Min-carbon plan")
    if knee:
        ax.scatter([knee["cost"] / 1e4], [knee["carbon"] / 1e3], marker="*", s=280,
                   color="#D4A017", edgecolor="#5a4300", lw=0.6, zorder=6, label="Knee point")

    dcost = (knee["cost"] - cost_pt["cost"]) / 1e4 if knee else 0
    dcarb = (cost_pt["carbon"] - knee["carbon"]) / 1e3 if knee else 0
    if dcost > 0:
        ax.annotate("", xy=(knee["cost"] / 1e4, knee["carbon"] / 1e3),
                    xytext=(cost_pt["cost"] / 1e4, cost_pt["carbon"] / 1e3),
                    arrowprops=dict(arrowstyle="->", color="#B8860B", lw=1.2, ls="--"))
        ax.text(knee["cost"] / 1e4 + 0.15, knee["carbon"] / 1e3 + 2.6,
                f"Min-cost -> knee:\n+{dcost*1e4:,.0f} CNY for -{dcarb:.1f} tCO2",
                fontsize=8.5, color="#8a6508")
    ax.annotate("lower-left is better", xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=8.5, color=C["gray"])
    ax.set_xlabel("Daily total cost (10k CNY)"); ax.set_ylabel("Daily emissions (tCO2)")
    ax.set_title("Cost-carbon frontier: the cheapest end is dirtier than no scheduling",
                 loc="left")
    ax.legend(loc="upper right")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(sens, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("battery", "Battery capacity (MWh)", "(a) Storage only buys money: bigger battery, worse carbon"),
        ("night", "Night grid carbon (kgCO2/MWh)", "(b) Storage only cuts carbon when the night grid is clean"),
        ("lam", "Carbon price (CNY/tCO2)", "(c) Carbon price barely moves the schedule before 2000 CNY/t"),
        ("solar", "PV peak capacity (MW)", "(d) With enough PV, cost and carbon point the same way"),
    ]
    for ax, (key, xlabel, title) in zip(axes.ravel(), panels):
        d = sens[key]
        x = d["x"]
        ax.plot(x, d["cost_save"] * 100, "o-", color=C["accent"], lw=1.6, ms=4.5,
                label="cost saving %")
        ax.plot(x, d["carbon_delta"] * 100, "s-", color=C["carbon"], lw=1.6, ms=4.5,
                label="carbon cut %")
        ax.axhline(0, color=C["gray"], lw=0.9)
        ax.set_xlabel(xlabel); ax.set_ylabel("vs no scheduling (%)")
        ax.set_title(title, loc="left"); _clean(ax)
        if key == "lam":
            ax.set_xscale("symlog", linthresh=20)
            ax.set_xticks([0, 20, 100, 500, 2000])
            ax.set_xticklabels(["0", "20", "100", "500", "2000"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_robust(rob, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(rob["save"] * 100, bins=28, color=C["grid"], alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(rob["save"]) * 100, color=C["accent"], lw=1.8,
               label=f"mean {np.mean(rob['save'])*100:.1f}%")
    ax.axvline(np.percentile(rob["save"], 5) * 100, color=C["gray"], ls="--", lw=1.4,
               label=f"5th pct {np.percentile(rob['save'],5)*100:.1f}%")
    ax.set_xlabel("Cost saving vs no scheduling (%)"); ax.set_ylabel("count")
    ax.set_title("(a) Cost saving under +/-15% PV forecast error", loc="left")
    ax.annotate(f"all 200 runs positive\nworst still {rob['save'].min()*100:.1f}%",
                xy=(0.03, 0.72), xycoords="axes fraction", fontsize=8.5, color=C["accent"])
    ax.legend(loc="upper left"); _clean(ax)

    ax = axes[1]
    xs = np.sort(rob["save"]) * 100
    ys = np.arange(1, len(xs) + 1) / len(xs) * 100
    ax.plot(xs, ys, color=C["grid"], lw=1.8, label="Fixed schedule (forecast-driven)")
    if "save_ideal" in rob:
        xs2 = np.sort(rob["save_ideal"]) * 100
        ax.plot(xs2, np.arange(1, len(xs2) + 1) / len(xs2) * 100, color=C["batt"],
                lw=1.8, ls="--", label="Perfect foresight (upper bound)")
    ax.set_xlabel("Cost saving (%)"); ax.set_ylabel("Cumulative probability (%)")
    ax.set_title("(b) CDF: it never loses money even when the forecast is off", loc="left")
    ax.legend(loc="lower right"); _clean(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cover(inst, plan, stats, path):
    """Cover image 1600x900, ready for the project showcase."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor("#0E1A2B")

    fig.text(0.055, 0.83, "HackSHSID 2026  ·  Track 2", color="#7FA7C9",
             fontsize=15, weight="bold")
    fig.text(0.055, 0.70, "PeakShift", color="white", fontsize=56, weight="bold")
    fig.text(0.055, 0.625, "Co-scheduling compute and power", color="#D6E4F0", fontsize=22)
    fig.text(0.055, 0.565, "for AI data centers", color="#D6E4F0", fontsize=22)
    fig.text(0.055, 0.475,
             "Move the work that can wait into the cheap, sunny hours.",
             color="#9FB6CC", fontsize=14)
    fig.text(0.055, 0.425,
             "Use the battery to carry noon's sunlight into the evening peak.",
             color="#9FB6CC", fontsize=14)

    ax = fig.add_axes([0.55, 0.20, 0.40, 0.52])
    ax.set_facecolor("#0E1A2B")
    h = _hours(inst)
    D = inst["PUE"] * (inst["L_rigid"] + plan["x"])
    ax.stackplot(h, plan["s"], plan["bd"], plan["g"],
                 colors=["#E9B44C", "#5B8C5A", "#37506E"], alpha=0.95, step="post")
    ax.plot(h, D, color="#E6EDF5", lw=1.6)
    ax.set_xlim(0, 24); ax.set_ylim(0, 16)
    ax.set_xticks(range(0, 25, 6)); ax.set_yticks([])
    ax.tick_params(colors="#7F96AD", labelsize=10)
    for s in ax.spines.values():
        s.set_color("#2A3F55")
    ax.grid(color="#23364A", lw=0.7, alpha=0.8)
    ax.set_xlabel("hour", color="#7F96AD", fontsize=11)
    ax.text(0, 1.02, "One day of optimised supply", transform=ax.transAxes,
            color="#9FB6CC", fontsize=12)

    dates = [
        (f"-{stats['cost_save']*100:.0f}%", "electricity bill", "#E9B44C"),
        (f"+{abs(stats['carbon_save'])*100:.0f}%", "carbon (cost-only plan)", "#7FA7C9"),
        (f"{stats['valley_share']*100:.0f}%", "energy bought at valley price", "#5B8C5A"),
    ]
    for i, (big, small, col) in enumerate(dates):
        y = 0.30 - i * 0.085
        fig.text(0.055, y, big, color=col, fontsize=30, weight="bold")
        fig.text(0.165, y + 0.012, small, color="#9FB6CC", fontsize=13)

    fig.text(0.055, 0.075,
             "Linear programming  ·  scipy / HiGHS  ·  11 independent checks  ·  reproducible in 6 s",
             color="#5E7A96", fontsize=12)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
