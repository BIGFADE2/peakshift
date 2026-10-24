# Submission copy

Copy-paste ready for the eight open items on the project page. **[Brackets] are the only things you need to change.**

---

## 1. Project name

```
PeakShift
```

*(Full title if the form allows: `PeakShift — Co-scheduling Compute and Power for AI Data Centers`)*

---

## 2. Short introduction

> Aim for 100–200 words; most forms show two or three lines.

```
Dispatch compute and power together, not separately.

We put the data center's workload and its electricity into a single linear program that decides, for every 15-minute slot of the day, how much deferrable training work to run, how much power to buy, and when to charge or discharge the battery.

Result: the daily electricity bill drops 30.5% (92,822 -> 64,518 CNY) and for three hours around noon the facility runs entirely on its own solar power.

But we found something counter-intuitive: optimising money alone makes carbon emissions 2.8% worse, and storage on its own does not cut emissions at all. The real lever is on-site solar, not carbon pricing.
```

---

## 3. Track

```
Track 2 — AI Data Center Power & Computing Co-Scheduling
```

---

## 4. Cover image

Upload `cover.png` from the repository root (1600×900, dark card style).

It carries three numbers: **−30 % electricity bill / +3 % carbon (cost-only plan) / 71 % of energy bought at valley price**, plus a small chart of the optimised 24-hour supply mix (dark = grid, yellow = PV, green = storage discharge).

---

## 5. Inspiration and description

> Aim for 400–800 words. This is the first thing judges read: say why you did it and what you found, not how the maths works.

```
This started from a simple puzzle: on a data center's electricity bill, why are the cheapest hours
not the cleanest ones?

Digging into it: Chinese industrial time-of-use tariffs have peak-to-valley ratios around 3x
(0.34 CNY/kWh in the valley, 1.08 at peak), while the grid is dirtiest at night, when coal
baseload runs and there is no solar. So the price signal actively pushes compute into the
highest-carbon hours.

There was no dataset, so the first job was to photograph reality. From NVIDIA's DGX H100
reference design (10.2 kW per node, about 40 kW per rack) we sized a facility at 300 racks,
roughly 10,000 GPUs, 12 MW of IT load. From the Ministry of Ecology and Environment's 2024
national electricity carbon-footprint factor (0.5777 kgCO2e/kWh) and the daily solar share we
built a 450-760 kg/MWh intensity curve. Prices come from published TOU tiers. Every parameter
is in the data table of the report, and the conclusions were tested against +/-30% error.

The model itself is deliberately plain: one linear program. The decisions are how much deferrable
compute to run in each 15-minute slot, how much power to buy, and how to charge the battery. The
constraints are power balance, 100 MWh of work that must be finished by 20:00, battery state of
charge and efficiency, and a 15 MW transformer limit. We chose an LP not because it is fashionable
but because it can be proved optimal, because every constraint can be said in one sentence, and
because 768 variables solve in 20 ms - fast enough to sit inside a live controller.

The schedule that comes out is clear: fill the valley at night with training work while charging
the battery steadily, drop grid import to zero between 11:00 and 14:00 (entirely solar), clear the
whole evening peak and let storage carry the rigid load. The bill falls from 92,822 to 64,518 CNY
a day.

What made us want to keep digging were three results we did not expect.

First, the cost-only plan emits 2.8% MORE carbon. Cheap and clean really are in conflict on this grid.

Second, storage buys money but not carbon. Noon solar is already fully absorbed by compute, so the
battery actually charges on deep-night coal (740 kg/MWh) and discharges into the evening peak
(760 kg/MWh). With 90% round-trip efficiency, discharging 1 MWh avoids 760 kg but charging it emits
822 kg - a net increase of 62 kg. A bigger battery makes it worse.

Third, carbon pricing barely matters. Raising the carbon price from 0 to 1000 CNY/t leaves the
schedule untouched; it takes 2000 CNY/t, more than 20 times the national ETS price, because the
marginal abatement cost on the Pareto frontier is 2,358 CNY/tCO2. Raising solar from 8 MW to
16 MW, on the other hand, is immediately a win-win: 50.8% cheaper and 25.7% less carbon.

So our conclusion is that to make an AI data center green you should build solar rather than buy
allowances, and that the value of a battery depends on what it charges from, not on how big it is.

Everything is open source and reproduces in about six seconds, including 11 independent checks -
one of which enumerates 194,481 possible schedules by brute force and finds exactly the same
optimum as the solver.
```

---

## 6. Public GitHub repository link

```bash
cd dc-scheduler
git init
git add .
git commit -m "PeakShift: compute-power co-scheduling for AI data centers"
gh repo create peakshift --public --source=. --push      # with GitHub CLI
# or: create an empty repo on github.com, then
# git remote add origin https://github.com/[your-username]/peakshift.git && git push -u origin main
```

Then fill in:

```
https://github.com/[your-username]/peakshift
```

**What judges will actually click on:**

- [ ] README shows the results table and one chart above the fold
- [ ] `pip install -r requirements.txt && python src/run_all.py` works
- [ ] `results/` already contains figures and tables — no need to run anything to see them
- [ ] `logs/run_log.txt` is present (that is the test evidence)
- [ ] no `.pylibs/`, `__pycache__/`, `out.txt` pushed (`.gitignore` handles it)

---

## 7. Test evidence

Paste or screenshot these:

**① The full one-command log** — `logs/run_log.txt`, containing the plan comparison, sensitivity sweep, Monte Carlo and all 11 checks.

**② Independent verification** — output of `python src/verify.py`:

```
check                                   result  evidence
Power balance residual (MW)             PASS    1.78e-15
SOC recursion residual (MWh)            PASS    8.88e-15
SOC within bounds                       PASS    [0.50, 20.00]
Deferrable energy conserved (MWh)       PASS    100.0000 vs 100.0
No compute after deadline               PASS    max=0.00e+00
Grid import under transformer limit     PASS    13.100 MW
PV within availability                  PASS    OK
Battery power within limits             PASS    max=5.000 MW
Two solvers agree                       PASS    gap 3.38e-16
Brute force (4 slots x 21 levels)       PASS    LP=21,649.62 / brute=21,649.62 CNY
Greedy heuristic not better than LP     PASS    LP 64,518 <= greedy 80,736 CNY

11/11 passed
```

**③ Result tables** — `results/tables/summary.csv`, `pareto.csv`, `sensitivity.csv`, `robustness.csv`.

**④ Reproduce**

```bash
pip install -r requirements.txt
python src/run_all.py     # about 6 s, regenerates every figure and table
python src/verify.py      # 11 independent checks
```

---

## 8. AI usage statement

> Be specific — judges trust this far more than a blanket "no AI was used".

```
This project used an AI coding assistant (DeepSeek Harness with the UltraMath modelling workflow) as a tool. The split was:

AI was used for:
- Code scaffolding and debugging: assembling the LP matrices, calling the scipy solver, and writing the matplotlib figure scripts;
- Turning our modelling decisions into formulas and prose in the documentation;
- Literature lookup: H100 power figures, TOU tariffs, the national electricity carbon-footprint factor, and their sources;
- Copy-editing the report and README.

We did ourselves:
- Problem framing and modelling choices: identifying the mismatch as a transport problem in time, choosing an LP, choosing a Pareto frontier to handle the money-carbon conflict;
- Choosing and sanity-checking every parameter (facility size, PUE, storage sizing, deadline);
- Interpreting and challenging the results: when the optimiser reported "cheaper but dirtier" we went back to check units, the carbon curve and the battery accounting before accepting that it was a real effect and not a bug;
- All conclusions, including verifying the two counter-intuitive findings about storage and carbon pricing.

Every line of generated code was actually executed (see logs/run_log.txt), and every number can be
reproduced with python src/run_all.py. No experimental result in this submission is fabricated.
```
