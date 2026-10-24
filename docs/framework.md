# Framework design (decisions made before writing code)

> Four things worth settling before a single line of code. This explains why the project ended up shaped the way it is.

---

## ① Core idea in one sentence

**This is a transport problem in time: move the work that can wait from expensive hours into cheap or clean ones, and move the electricity that can be stored from sunny hours into dark ones.**

The three "varies" in the prompt — price, demand, supply — are three faces of one conflict:
**the cheapest hour, the busiest hour and the cleanest hour do not coincide.**

So there is exactly one lever: **time**. Spatial flexibility (shipping jobs to another data center) is deliberately out of scope; the prompt does not provide it and adding it would make this a different problem.

---

## ② Core quantities (only three)

| Quantity | One-line definition | Unit |
|---|---|---|
| $x_t$ | how much **deferrable compute** runs in slot $t$ | MW |
| $g_t$ | how much power is **bought from the grid** in slot $t$ | MW |
| $E_t$ | **energy left in the battery** at the end of slot $t$ | MWh |

Everything else is a function of these. **The decision is the time series of these three.**

---

## ③ The prompt's five questions → modelling actions

The five questions in section 2 of the prompt are not five chapters. They are **five things the model structure must answer**:

| Question | Modelling action | Output |
|---|---|---|
| How do you define good? | Four metrics: money / carbon / peak / SLA, plus a per-MWh efficiency metric | metrics table + `summary.csv` |
| Better than no scheduling, by how much? | Build a "no scheduling" baseline (work spread evenly, battery idle) and run it through the same model | `fig3_compare.png` |
| Why believe it is near-optimal? | Dual bound + second solver + **small-instance enumeration** + independent greedy | 11 checks in `verify.py` |
| Does it hold when conditions change? | Sensitivity sweeps + Monte Carlo + three scenarios (A/B/C) | `fig5`, `fig6`, scenario table |
| Are the assumptions reasonable? | Grade assumptions as strong/weak and state what flips if each fails | report Q5 table |

**That table dictated the shape of the model**:

- Answering "define good" → the objective must decompose (money and carbon separately tunable), so weighted sum rather than a single scalar;
- Answering "how much better" → a baseline is mandatory, so the model must support switching a decision off (`shift=False`, `battery=False`);
- Answering "near-optimal" → it must be checkable by enumeration, so an LP rather than a heuristic or RL;
- Answering "conditions change" → every parameter must be injectable (`build_instance(**overrides)`);
- Answering "assumptions wrong" → the carbon curve must be swappable (`carbon24=`), otherwise the weakest assumption cannot be tested at all.

---

## ④ Proposition tree (two levels)

```
"Can an AI data center run cheaper and greener?"
│
├─ P1  Are cheap and clean actually mismatched in time?
│     └─ Evidence: fig 1(b); the valley is cheapest and dirtiest; the "cheap and clean" cell is empty
│
├─ P2  Does price-only scheduling save money AND carbon?
│     └─ No: 30.5% cheaper, 2.8% more carbon  <- the core counter-intuitive result
│
├─ P3  Can storage close the gap?
│     └─ It fixes money, not carbon: a net +62 kg per MWh moved, worsening with capacity
│
├─ P4  What does work?
│     ├─ carbon price: almost nothing (needs 2000 CNY/t before the schedule changes)
│     └─ PV: yes (win-win from 12 MW; at 16 MW, -50.8% cost / -25.7% carbon)
│
└─ P5  Is that stable?
      ├─ +/-15% PV forecast error: all 200 runs positive, worst still 27.6% saved
      ├─ clean nights (430 kg/MWh): conclusion flips, carbon -14.2%
      └─ deadline pulled to 12:00: infeasible -- the physical boundary of the method
```

**P1 is the premise, P2 is the discovery, P3 eliminates the obvious wrong answer, P4 gives the right one, P5 draws the boundary.**

---

## ⑤ Ideas we had first and dropped

| Initial idea | Why it was dropped |
|---|---|
| MILP with every training job as an integer variable | explodes at 96 slots × several jobs; <2 % gain on a single day |
| Reinforcement learning for "smart scheduling" | not explainable, no simulator available, and no answer when a judge asks "why this schedule" |
| Define "green" as minimum carbon | produces a plan that costs 12 % more; nobody would operate it |
| Assume "storage must cut carbon" | the data disagrees — this turned out to be the most valuable negative result in the project |
| Put the demand charge straight into the main model | a one-day model cannot set a monthly peak and would systematically overstate it; became a separate experiment instead |

---

## ⑥ Scope

**In scope**: single day, deterministic, single site, energy-level scheduling.
**Out of scope**: cross-region load migration, job-level integer scheduling, real-time rolling horizon, carbon capture or certificate trading.

Writing the boundary down is better than pretending to do everything.
