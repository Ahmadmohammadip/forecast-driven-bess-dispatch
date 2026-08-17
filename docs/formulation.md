# Formulation

A linear program that decides, for each period of a horizon, how much to import,
export, charge, discharge and curtail at a behind-the-meter site with PV and a
battery — minimising cost against **forecast** load, PV and price.

## Provenance

This is a standard behind-the-meter storage dispatch model. It is a
well-established formulation, not an original one; the contribution of this
repository is the measurement around it — what the forecasts are worth, and
which of them matters.

## 1. Sets and indices

| Symbol | Description | Code |
|---|---|---|
| $t \in T$ | Periods, $0 \dots T-1$ | `m.T` |
| $\Delta t$ | Period length in hours | `m.dt` |

## 2. Parameters

| Symbol | Description | Code |
|---|---|---|
| $L_t$ | Forecast load (MW) | `m.p_load` |
| $G_t$ | Forecast PV generation (MW) | `m.p_pv` |
| $\pi^{buy}_t$ | Delivered import price (EUR/MWh) | `m.buy` |
| $\pi^{sell}_t$ | Export compensation (EUR/MWh) | `m.sell` |
| $\eta_c, \eta_d$ | Charge / discharge efficiency | `charge_efficiency`, … |
| $\underline{E}, \overline{E}$ | Usable energy band (MWh) | `soc_min_mwh`, `soc_max_mwh` |
| $E_0$ | Opening state of charge (MWh) | `initial_soc_mwh` |
| $\overline{P}^c, \overline{P}^d$ | Power limits (MW) | `p_charge_max_mw`, … |
| $\overline{I}, \overline{X}$ | Grid import / export limits (MW) | `import_limit_mw`, … |
| $c^{deg}$ | Degradation cost (EUR/MWh throughput) | `degradation_cost_eur_mwh` |
| $c^{dem}$ | Demand charge (EUR/MW of peak) | `demand_charge_eur_mw` |

## 3. Decision variables

All continuous. **There are no binary variables anywhere in this model** — see
§7.

| Symbol | Description | Code |
|---|---|---|
| $i_t \ge 0$ | Grid import | `m.g_imp` |
| $x_t \ge 0$ | Grid export | `m.g_exp` |
| $p^c_t \ge 0$ | Charging power | `m.p_ch` |
| $p^d_t \ge 0$ | Discharging power | `m.p_dis` |
| $e_t$ | State of charge | `m.soc` |
| $s_t \ge 0$ | PV curtailment | `m.curtail` |
| $\rho \ge 0$ | Peak grid import over the horizon | `m.peak` |

## 4. Constraints

### 4.1 Power balance

$$
L_t + p^c_t + x_t \;=\; (G_t - s_t) + p^d_t + i_t \qquad \forall t
$$

Consumption and charging and export on the left; usable generation, discharge
and import on the right.

**Curtailment is a decision variable, not a slack.** Without it, a period whose
PV exceeds load plus charging plus the export limit would be reported infeasible
where reality simply spills the surplus.

### 4.2 State of charge

$$
e_t = e_{t-1} + \eta_c\, p^c_t\, \Delta t - \frac{p^d_t\, \Delta t}{\eta_d},
\qquad e_{-1} \equiv E_0
$$

Charging is derated going in and discharging going out, so a full cycle returns
$\eta_c \eta_d$ of what went in — 0.9025 at the reference case's 0.95/0.95.

### 4.3 Bounds

$$
\underline{E} \le e_t \le \overline{E}, \quad
p^c_t \le \overline{P}^c, \quad
p^d_t \le \overline{P}^d, \quad
i_t \le \overline{I}, \quad
x_t \le \overline{X}, \quad
s_t \le G_t
$$

### 4.4 Terminal state of charge

$$
e_{T-1} = E_0
$$

Optional (`enforce_terminal_soc`), and on by default. Released, a finite horizon
ends by selling the battery empty and the reported cost is flattered by energy
that was never paid for. This is pinned by a test.

### 4.5 Peak import

$$
\rho \ge i_t \qquad \forall t
$$

A variable with a $\ge$ constraint rather than a $\max$, which keeps the model
linear.

## 5. Objective

$$
\min \;
\underbrace{\sum_t \left( \pi^{buy}_t i_t - \pi^{sell}_t x_t \right) \Delta t}_{\text{energy}}
\;+\;
\underbrace{c^{deg} \sum_t \left( p^c_t + p^d_t \right) \Delta t}_{\text{degradation}}
\;+\;
\underbrace{c^{dem} \rho}_{\text{demand charge}}
$$

The brief asks for three variants, and all three are built:

| Variant | Terms |
|---|---|
| `cost` | energy |
| `cost_degradation` | energy + degradation |
| `cost_degradation_demand` | energy + degradation + demand charge |

They genuinely differ, and the difference is instructive. On 2020-01-15 under
perfect foresight:

| Variant | Cost (EUR) | Peak import (MW) | Cycles |
|---|---:|---:|---:|
| no battery | 1608.01 | 0.943 | — |
| `cost` | 1590.41 | **1.155** | 1.20 |
| `cost_degradation` | 1593.51 | 1.072 | 0.40 |
| `cost_degradation_demand` | 1598.38 | **0.913** | 0.40 |

The cost-only variant pushes peak import *above* the no-battery peak, because
charging adds to the peak and nothing penalises it. Adding the demand charge
pulls it below. A single objective would have hidden that entirely.

Each cost term is a named Pyomo `Expression`, and the reported breakdown reads
those same objects rather than recomputing — so `cost_breakdown()` sums to the
objective by construction, not by a second calculation that could disagree.

## 6. Every coefficient

The brief asks for these to be documented rather than assumed.

| Coefficient | Value | Where it comes from |
|---|---|---|
| $\eta_c = \eta_d$ | 0.95 | The brief's §5 reference case. |
| Usable band | 10%–90% | The brief's §5. |
| $c^{deg}$ | 2 EUR/MWh | Round number chosen to be visible against a median daily spread of 28.7 EUR/MWh without dominating it. **An economic proxy, not electrochemistry** — see §8. |
| $c^{dem}$ | 5 EUR/MW/day | Small by design; the sweep runs 0, 5 and 50. See the caveat in §8. |
| Import markup | 60 EUR/MWh | Network charges, levies, taxes, margin — everything that is not the energy. Also above the 27 EUR/MWh arbitrage floor derived in §7.2. |
| Export ratio | 0.70 × wholesale | A plausible commercial feed-in arrangement. |

## 7. Two things that look like modelling choices and are measured results

### 7.1 No binary variables

The brief's §10 asks to avoid unnecessary binaries and lists
no-simultaneous-charge-and-discharge as optional. It is genuinely unnecessary
here, and that was checked rather than assumed: across export ratios 0.7×, 1.0×
and 1.3×, with and without degradation cost and with and without a demand
charge, **no optimum ever charged and discharged in the same period**. Round-trip
losses already make it strictly wasteful, so the constraint would never bind.
Two tests pin it.

Keeping the model an LP is not just tidiness. It is why 2,880 rolling-horizon
solves finish in minutes.

### 7.2 The guard that a binary would have hidden

A probe found a failure mode that *looks* like the same thing and is not: with
export compensation above the import price, the optimum imports and exports
simultaneously **in every hour, with no battery in the model at all**. That is
the meter being gamed.

The tempting fix is a binary forbidding simultaneous import and export. It is
the wrong fix — it would make an unphysical tariff solve slowly instead of
failing. The guard belongs in `Tariff` validation.

Negative prices make this subtle, and this dataset has 484 of them. A tariff of
the form $\pi^{buy} = w + m$, $\pi^{sell} = r \cdot w$ is safe for $w > 0$
whenever $r \le 1$, and **inverts below zero**: at $w = -90$ EUR/MWh with
$r = 0.7$, importing pays you 90 while exporting costs you 63, so the round trip
nets $+27$ EUR/MWh forever. The requirement is

$$
m \;\ge\; \max_t \; w_t (r - 1)
$$

which on this data is 27.0 EUR/MWh at $r = 0.7$. `TariffPolicy.minimum_safe_markup`
computes it, and returns a hair above the analytic bound rather than exactly on
it — at the bound, floating point lands on either side, and a helper that
returns a value its own validator rejects is a bug.

## 8. Known simplifications

Stated plainly, because several of them would matter to anyone using this for a
real decision.

**The demand charge is levied on the daily peak.** A real one is monthly. The
horizon here is a day, so the model minimises each day's peak independently, and
minimising 60 daily peaks is *not* the same as minimising the month's — one bad
day sets the monthly bill regardless. The KPI table reports the realised peak
over the whole window, which is the number that would actually be charged.

**Degradation is a throughput proxy.** $c^{deg}(p^c + p^d)$ knows nothing about
cycle depth, temperature, calendar ageing, or C-rate. It is an economic
stand-in for wear that makes the optimizer trade cycling against revenue, and no
claim of electrochemical accuracy is made or implied.

**Load and PV are national aggregates rescaled to one site.** The shapes are
real measurements; the magnitudes are assumptions. A national load curve is
smoother, and therefore easier to forecast, than a single building's — so
load-forecast accuracy here is optimistic. Rather than leave that as a caveat,
the ablation measures how much it matters: making the load forecast perfect
closes **0.8%** of the gap to perfect foresight. Even a much worse load
forecaster would barely move the result.

**Point forecasts only.** No prediction intervals, so the optimizer cannot be
made risk-aware. This is the single most valuable extension, and it is the
brief's own stretch goal.

**Perfect efficiency in the replay.** When a forecast-driven schedule is
re-priced against actuals, battery actions are clipped at the energy band but
otherwise executed as planned. A real controller would re-optimise; the rolling
arm is what models that.

**No transformer, thermal or degradation-state limits**, no reactive power, no
multi-year horizon, and no capital cost — savings are operational only, so
nothing here says whether the battery pays for itself.

## 9. A behaviour worth knowing before it surprises someone

**The forecast-driven arm can lose money.** At half the observed price
volatility it saves −3.89 EUR over the test window — worse than not having the
battery — and at a degradation cost of 10 EUR/MWh, −0.11 EUR. Perfect foresight
stays positive in both cases.

This is not the battery failing. It is forecast error exceeding the spread the
battery is trying to trade: acting confidently on a noisy signal is worse than
not acting. A system worth installing in one market can destroy value in
another, and a single reference-case number would never have revealed it.
