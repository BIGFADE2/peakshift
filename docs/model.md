# Model notes (derivation and implementation details)

> Why the model looks the way it does. Code lives in `src/optimizer.py`.
> Prerequisites: arithmetic, plus knowing that "linear programming" means finding the lowest point
> of a region bounded by straight lines.

---

## 1. There are only three core quantities

| Quantity | Meaning | Unit | Why it is core |
|---|---|---|---|
| $x_t$ | deferrable compute power in slot $t$ | MW (IT side) | the only thing that can be moved freely |
| $g_t$ | grid import in slot $t$ | MW | source of all cost and most emissions |
| $E_t$ | stored energy | MWh | the only way to move electricity in time |

Everything else is a function of these three: facility demand $=\mathrm{PUE}\cdot(L^{rig}_t+x_t)$, PV $s_t$ is dictated by weather, and charging/discharging only changes $E_t$.

---

## 2. The full model

### 2.1 Objective

$$
J=\sum_{t=1}^{T}\Big[\underbrace{p_t g_t\Delta t}_{\text{electricity}}+\underbrace{\lambda e_t g_t\Delta t}_{\text{carbon}}\Big]+d_{deg}\sum_t (c_t+d_t)\Delta t+\varepsilon\sum_t u_t\Delta t
$$

| Symbol | Meaning | Unit | Dimensional check |
|---|---|---|---|
| $p_t$ | TOU price | CNY/MWh | $p\,g\,\Delta t$ = CNY/MWh × MW × h = **CNY** ✓ |
| $e_t$ | grid carbon intensity | kgCO₂/MWh | $e\,g\,\Delta t$ = **kg** ✓ |
| $\lambda$ | carbon price | CNY/kg | $\lambda e g\Delta t$ = **CNY** ✓ |
| $d_{deg}$ | battery wear | CNY/MWh | **CNY** ✓ |
| $u_t$ | auxiliary: power change | MW | **CNY** ✓ |
| $\Delta t$ | 15 min = 0.25 h | h | |

> Unit errors are the most common way to break this model. $\Delta t$ must be in **hours** and power in MW,
> so their product is MWh. We actually hit this bug: prices were typed in CNY/kWh while the model worked
> in MW, and the daily bill came out as 93 CNY — which made the error obvious immediately.

### 2.2 Constraints

**(1) Power balance** (one per slot)
$$g_t+s_t+d_t=c_t+\mathrm{PUE}\cdot(L^{rig}_t+x_t)$$
Three supplies (grid, PV, discharge) on the left, two demands (facility, charging) on the right. Units: MW.

**(2) Deferrable work total**
$$\sum_{t=1}^{T}x_t\Delta t=W=100\ \text{MWh}$$
The work must be delivered.

**(3) Deadline**
$$x_t=0\quad\text{for all } t\ge t_{dl}$$
Implementation detail: rather than writing the constraint, we set the upper bound to 0 — cheaper for the solver.

**(4) Storage dynamics with efficiency**
$$E_t=E_{t-1}+\eta\,c_t\Delta t-\frac{d_t\Delta t}{\eta},\qquad 0\le E_t\le E_{max}$$

$\eta=0.95$ one-way, so 90 % round trip. Charging multiplies by $\eta$ (discount on the way in), discharging divides (you need more out). This term is the origin of the "storage does not cut carbon" result later.

**(5) Cyclic constraint**
$$E_T=E_0$$
Otherwise the model treats the battery as free energy that never has to be paid back.

**(6) Limits**
$$0\le g_t\le G_{max}=15\ \text{MW},\quad 0\le s_t\le \bar s_t,\quad 0\le c_t,d_t\le B_p=5\ \text{MW}$$

**(7) Power smoothing** (linearising $|\cdot|$)

Penalising $|x_t-x_{t-1}|$ directly is not linear. The standard trick:

$$u_t\ge x_t-x_{t-1},\qquad u_t\ge x_{t-1}-x_t,\qquad u_t\ge 0$$

Because the objective minimises $u_t$ with a positive coefficient, the optimum drives it down to $|x_t-x_{t-1}|$.
**Relaxed constraint + minimisation ⇒ exact.** The same is done for battery charging; without it, the
LP's multiple optima produce a spiky charge profile that is both ugly and unrealistic.

### 2.3 Size

| Item | Count |
|---|---|
| Variables | $8\times96=768$ (including smoothing auxiliaries) |
| Equality constraints | $2T+2=194$ |
| Inequality constraints | $4(T-1)=380$ |
| Solve time | about 20 ms (HiGHS) |

---

## 3. Why an LP and not something else

| Alternative | Why not |
|---|---|
| MILP | Indivisible jobs and start-up costs are genuinely integer. But 96 slots × several jobs explodes, for <2 % gain on a single day |
| Dynamic programming | State is (t, energy left, SOC); SOC is continuous, so it must be discretised — cost in both accuracy and dimensionality |
| Reinforcement learning | Needs a large simulator and is **not explainable** — when a judge asks "why this schedule", there is no answer |
| Greedy / rule-based | We implemented one (`verify.greedy`); it is 25 % worse and only useful as an upper bound |
| **Linear programming** ✓ | Provably optimal, explainable, 20 ms, and new constraints are one extra row |

**A bonus of the LP formulation**: dual variables are shadow prices.
The dual of the grid constraint is the marginal cost of a MWh; the dual of the work constraint is
**how much one extra MWh of quota is worth** — which directly answers "should we expand, should we buy storage".

---

## 4. How the Pareto frontier is traced

Weighted-sum method: sweep $\lambda$ from 0 to very large, solve one LP per value, collect (cost, carbon).

**Why weighted sum is exact here**: the LP feasible set is a convex polytope and the objective is linear,
so every efficient point lies on the lower boundary of the convex hull and the weighted sum reaches all of them.
An ε-constraint method is not needed (although `carbon_cap` is available in the code).

**Defining the knee**: normalise cost and carbon to [0,1] and take the point closest to the origin,
$$\text{knee}=\arg\min_{k}\ \Big[\big(\tfrac{J_k-J_{min}}{J_{max}-J_{min}}\big)^2+\big(\tfrac{E_k-E_{min}}{E_{max}-E_{min}}\big)^2\Big]$$

For this instance that lands at $\lambda=1.535$ CNY/kg (1,535 CNY/tCO₂).

---

## 5. Verification design (Phase 2.5)

| Check | What it does | Why it proves something |
|---|---|---|
| Feasibility re-check | recompute balance, SOC recursion, totals, deadline from the raw solution | rules out "solved, but physically impossible" |
| Cross-solver | dual simplex vs interior point | rules out a solver bug |
| Brute force | 4 slots × 21 levels = 194,481 schedules enumerated | the hardest evidence: direct comparison |
| Independent greedy | a separately written rule | rules out a wrong objective creating a fake optimum |

The brute-force test deserves a note: it discretises the continuous problem and searches it exhaustively.
If the LP value were far below the enumeration, constraints would be missing; if they agree, the model is tight.
Measured: **exactly equal** (21,649.62 CNY). That is the most convincing single number in the project.

---

## 6. Mistakes we made (for whoever comes next)

1. **Units** — CNY/kWh used as CNY/MWh, bill 1000× too small.
2. **Singularity** — $E_0=10$ with $E_{max}=0$ is infeasible; the sensitivity sweep forgot to move the initial SOC.
3. **Multiple optima** — the LP optimum is not unique; dozens of schedules cost the same and the charging profile came out as pulses. Adding smoothing fixed it.
4. **Degenerate test** — the first brute-force case had PV above demand, so grid import (and therefore cost) was identically zero. It "passed" while testing nothing. Moving it to 17:00–20:00 made it meaningful.
5. **Fonts** — matplotlib's default font has no CJK glyphs and no subscript ₂, so figures came out as boxes until we set the font stack.
