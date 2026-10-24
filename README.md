# PeakShift

**Co-scheduling compute and power in AI data centers** · HackSHSID 2026 · Track 2

Move the deferrable work into the cheap, sunny hours. Use the battery to carry noon's sunlight into the evening peak.

| Metric | No scheduling | PeakShift | Change |
|---|---:|---:|---:|
| Daily cost | 92,822 CNY | **64,518 CNY** | **−30.5 %** |
| Cost per MWh | 654 CNY | **435 CNY** | −33.5 % |
| Daily emissions | 99,092 kg | 101,890 kg | +2.8 % |
| Peak import | 10.1 MW | 13.1 MW | +29 % |

> Optimising money alone makes the carbon **worse**. That counter-intuitive result is the heart of this project — see *Three findings* below.

---

## Quick start

```bash
pip install -r requirements.txt
python src/run_all.py
```

Runs in about 6 s and writes `results/` (6 figures + 7 tables); the full log lands in `logs/run_log.txt`.

Verification only:

```bash
python src/verify.py
```

> **Cannot install packages?** If `pip install` is blocked (offline / restricted sandbox), drop the
> dependencies into a `.pylibs/` folder next to the repo, or a `vendor/` folder inside it.
> `src/_deps.py` finds them automatically; every other command stays the same.

---

## How it works

```
Inputs (profiles.py)                Linear program (optimizer.py)         Results (run_all.py)
+--------------------------+        +----------------------+             +----------------+
| TOU price 0.34/0.66/1.08 |        | decide: x_t compute  |             | 6 figures      |
| carbon 450-760 kg/MWh    |  --->  |         g_t import   |  --->       | 7 CSV tables   |
| rigid load 2-4.6 MW      |        |         c_t,d_t batt |             | Pareto frontier|
| PV 8 MW peak             |        | minimise: money      |             | sensitivity    |
| 100 MWh deferrable work  |        |          + carbon    |             | robustness     |
+--------------------------+        |          + wear      |             | 11 checks      |
                                    | s.t. balance, SOC,   |             +----------------+
                                    |      deadline, 15MW  |
                                    +----------------------+
```

96 fifteen-minute slots a day, 768 variables, solved by HiGHS in about 20 ms.

The optimised day:

![Optimised day](results/figures/fig2_schedule.png)

- 00–08: deferrable work runs flat out, battery charges steadily
- 11–14: **grid import drops to zero** — the facility runs entirely on its own PV
- 18–22: deferrable work stops; storage covers the rigid load through the peak

---

## Three findings

**① Cheapest ≠ cleanest.** The valley price of 0.34 CNY/kWh coincides with a carbon intensity of 740 kg/MWh, while noon at 450 kg/MWh is priced as peak. The cost-only plan saves 30.5 % and emits 2.8 % *more*. The "cheap and clean" cell is empty.

**② Storage buys money, not carbon.** Noon PV is already absorbed by compute, so the battery actually charges on deep-night coal (740) and discharges into the evening peak (760). At 90 % round-trip efficiency: 1 MWh discharged avoids 760 kg but charging it emits 822 kg — **a net increase of 62 kg**. Bigger battery, worse carbon (−1.8 % at 0 MWh, −4.4 % at 40 MWh).

**③ Carbon pricing does almost nothing; PV does a lot.** Carbon price from 0 to 1000 CNY/t leaves the schedule unchanged; the marginal abatement cost on the frontier is 2,358 CNY/tCO₂. Raising PV from 8 MW to 16 MW is instantly a win-win: 50.8 % cheaper, 25.7 % less carbon.

When night carbon intensity falls below **680 kg/MWh** (e.g. a wind-rich province) the conclusion flips: carbon drops **14.2 %** and storage starts helping too (+2.8 %).

---

## Why believe it

`python src/verify.py` → **11/11 passed**

| Evidence | Result |
|---|---|
| Dual bound = primal objective | gap 0 |
| Different solver (dual simplex vs interior point) | differ by 3.4×10⁻¹⁶ |
| **Brute force: 4 slots × 21 levels = 194,481 schedules** | LP = brute force = **21,649.62 CNY**, exactly |
| Independent greedy heuristic | 80,736 CNY ≥ LP 64,518 CNY |
| 8 feasibility re-checks | residuals ≤ 10⁻¹⁴ |

---

## Layout

```
.
├── src/
│   ├── profiles.py     input profiles (price / carbon / load / PV, all adjustable)
│   ├── optimizer.py    the LP and its solution
│   ├── verify.py       11 independent checks
│   ├── plots.py        6 figures + cover
│   ├── _deps.py        dependency fallback (vendor/ or .pylibs/)
│   └── run_all.py      one entry point
├── results/
│   ├── figures/        fig1-fig6
│   └── tables/         summary / pareto / sensitivity / robustness / checks / schedule / demand_charge
├── docs/
│   ├── report.md       main report (sources, model, results, the five questions)
│   ├── model.md        derivation, units, duality, verification design, mistakes we made
│   ├── framework.md    design decisions made before writing code
│   └── zh/             the same notes in Chinese
├── logs/run_log.txt    full run log (test evidence)
└── cover.png           showcase cover, 1600×900
```

---

## Tuning it

`PARAMS` at the top of `src/profiles.py` is the whole control panel:

```python
PARAMS = dict(
    T=96, dt=0.25,        # 96 fifteen-minute slots
    PUE=1.15,             # liquid-cooled AI hall
    Xmax=8.0, W=100.0,    # deferrable: 8 MW peak, 100 MWh per day
    Gmax=15.0,            # transformer capacity
    Emax=20.0, Bp=5.0,    # storage: 20 MWh / 5 MW
    eta=0.95, E0=10.0,
    d_deg=50.0,           # battery wear, CNY/MWh
    S_peak=8.0,           # PV peak
    deadline_hour=20.0,
    scenario="A",         # "A" TOU / "B" Shandong-style midday valley
)
```

Or pass overrides directly: `build_instance(Emax=40, S_peak=16, scenario="B")`.

---

## Data sources

- [IEA Electricity 2024](https://www.iea.org/reports/electricity-2024) — data-center electricity growth
- [MEE announcement on 2024 electricity carbon-footprint factors](https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202510/t20251024_1130734.html) — 0.5777 kgCO₂e/kWh national average
- [NVIDIA DGX H100: 10.2 kW per node, ~40 kW per rack](https://datavault.com.pk/blog-nvidia-h100-h200-kilowatts-per-rack/)
- [Peak-valley spread >0.6 CNY/kWh in many provinces](https://desn.com.cn/news/show-1720470.html)

> All parameters are **calibrated from public sources** for modelling and comparison; they are not any company's real books.
> Conclusions are insensitive to ±30 % parameter error (see `fig5`, `fig6`).

---

## License

MIT
