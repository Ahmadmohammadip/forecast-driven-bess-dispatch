# Forecasting

Three series are forecast — load, PV and wholesale price — one day ahead, 24
hourly steps, issued at 00:00 UTC. The forecasts exist for one reason: to be
handed to the dispatch optimizer. Nothing here is optimised for a leaderboard.

Everything below is reproducible:

```bash
python -m bess_dispatch.forecasting.selection
```

which writes `results/forecast_selection.csv`.

## Leakage: how it is prevented, and how that is proved

This is the claim the whole project depends on, so it gets structural
protection rather than care.

**The barrier.** The first statement of `build_features` is

```python
past = frame.loc[frame.index < issue_time]
```

and everything afterwards operates on `past`. A feature *cannot* reference the
future, because the rows are not in scope. That is deliberately stronger than
computing features over the full frame and masking them afterwards, which is
correct only for as long as everyone remembers to mask.

**One code path.** `build_features` serves training and inference alike. There
is no separate serving path that could drift — the usual way leakage enters a
codebase that is careful in principle.

**The lag guard.** Lags are taken relative to the *target* timestamp, so
`lag_24h` means "same hour yesterday" at every step of the horizon. That is only
safe while the lag exceeds the horizon length; otherwise step `k ≥ L` would read
inside the forecast window. `FeatureSpec.check_horizon` refuses the combination
rather than trusting the caller:

```
ValueError: lag(s) [24] are shorter than the 48-step horizon.
```

**The proof.** `assert_no_future_reference` builds features on the real frame,
then again on a frame whose entire future has been overwritten with noise around
10⁶, and requires the two to be bit-identical. `tests/test_forecast_leakage.py`
runs it for all three targets.

A test that cannot fail proves nothing, so the suite also inserts a deliberate
leak — a feature reading the target at its own timestamp — and asserts the check
catches it:

```
AssertionError: features changed when the future was corrupted, so they leak.
Offending column(s): ['oops_same_hour']
```

## Features

19 columns for a single-target spec. All derive from data before the issue time,
or from the calendar, which is knowable arbitrarily far ahead.

| Group | Columns | Source |
|---|---|---|
| Lags | `lag_24h`, `lag_48h`, `lag_168h` | Observed value at target − lag |
| Window statistics | `last_24h_{mean,std,min,max}`, `last_168h_{…}` | Windows ending at the issue time |
| Last observation | `last_observed` | Final value before issue |
| Position | `horizon_step` | 0–23, how far ahead this row is |
| Calendar | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dayofweek`, `is_weekend` | Local-time calendar |

Hour and month are encoded as sine/cosine pairs so hour 23 sits adjacent to hour
0 rather than 23 units away — a tree can learn the wraparound, a linear model
cannot. Calendar features use **local** time: solar position and human routine
follow the local clock, and a model given UTC hour has to learn the DST offset as
noise.

Month-of-year is in the set because the EDA showed the tradeable intraday shape
*inverts* between seasons — midday is cheap from March to September and dear from
November to January. A model without month cannot represent that.

## Model selection: measured, not assumed

Fitted on 2018-10-01 → 2019-10-01, scored on 2019-10-01 → 2020-01-01, 92 issue
times, 2,208 forecast hours.

### Load (MW)

| Model | MAE | RMSE | R² | Skill vs persistence |
|---|---:|---:|---:|---:|
| persistence | 0.0558 | 0.0850 | 0.577 | — |
| weekly persistence | 0.0377 | 0.0647 | 0.755 | +32.4% |
| ridge | 0.0326 | 0.0453 | 0.880 | +41.7% |
| **gradient boosting** | **0.0228** | 0.0382 | 0.915 | **+59.1%** |
| xgboost *(optional extra)* | 0.0210 | 0.0345 | 0.930 | +62.4% |
| published TSO forecast *(benchmark)* | 0.0237 | **0.0291** | **0.950** | +57.5% |

The interesting row is the last one. The transmission operator publishes its own
day-ahead load forecast, produced with far more information than this site has,
and it is included as an external benchmark rather than a model this repo can
claim. The gradient-boosted model **beats it on MAE** (0.0228 vs 0.0237) and
**loses to it on RMSE** (0.0382 vs 0.0291) and R².

That is not a contradiction, it is the shape of the errors: this model is more
accurate on a typical hour and worse on the hard ones. For a dispatch decision
that is the less comfortable trade of the two, because the expensive mistakes
live in the tail.

### PV (MW)

| Model | MAE | RMSE | R² | Skill vs persistence |
|---|---:|---:|---:|---:|
| persistence | 0.0159 | 0.0374 | 0.819 | — |
| weekly persistence | 0.0219 | 0.0511 | 0.663 | −38.2% |
| **ridge** | **0.0151** | **0.0330** | **0.859** | **+5.0%** |
| gradient boosting | 0.0173 | 0.0399 | 0.794 | −9.0% |
| xgboost *(optional extra)* | 0.0193 | 0.0448 | 0.740 | −21.4% |

**Both gradient-boosted models lose to "yesterday, same hour."** Ridge beats
persistence by 5%, which is close to nothing. This is reported rather than
buried: on this series, machine learning did not earn its keep, and the more
flexible the model the worse it did — the signature of overfitting on 8,592
training rows.

All-hours PV metrics also flatter every model, because roughly half of all hours
are night and predicting zero is free. Over daylight hours only, ridge's MAE is
**0.0360**, not 0.0151. The daylight number is the one that means anything.

### Price (EUR/MWh)

| Model | MAE | RMSE | R² | Skill vs persistence |
|---|---:|---:|---:|---:|
| persistence | 8.99 | 13.39 | 0.130 | — |
| weekly persistence | 10.12 | 14.54 | −0.026 | −12.6% |
| **ridge** | **7.12** | **10.28** | **0.488** | **+20.8%** |
| gradient boosting | 7.44 | 11.00 | 0.413 | +17.2% |
| xgboost *(optional extra)* | 7.41 | 11.28 | 0.383 | +17.6% |

Price is the hardest of the three and the linear model wins it. Even the best
model leaves a mean absolute error of 7.12 EUR/MWh against a median daily spread
of 28.7 — so roughly a quarter of the arbitrage signal is noise to the
controller. This is the number that limits how much of the perfect-foresight
value the forecast-driven dispatch can capture.

### What was selected

```python
BEST_BY_TARGET = {
    "load_mw": "gradient_boosting",
    "pv_mw": "ridge",
    "price_eur_mwh": "ridge",
}
```

One model per series, chosen by validation MAE. Picking a single family for all
three would have been wrong on two of them.

The published TSO forecast and xgboost are excluded from the default slot —
the first because it is someone else's forecast rather than something this repo
produces, the second because it sits behind an optional extra. Both remain in
the table, because the comparison is the point.

## Was XGBoost justified?

The brief says "XGBoost if justified". Measured against
`HistGradientBoostingRegressor` on the validation window:

| Series | MAE change | Verdict |
|---|---:|---|
| Load | −7.9% | Better |
| Price | −0.5% | A tie |
| PV | +11.4% | Worse |

Justified on one series out of three. It stays behind the optional `boost`
extra, and is documented rather than defaulted, because a compiled dependency
for one series' worth of gain is the user's call to make:

```bash
pip install -e ".[boost]"
```

## Error structure

Both breakdowns the brief asks for, on the selected models.

**By horizon step** — MAE at the first, middle and last hour of the window:

| Series | step 0 | step 11 | step 23 | ratio |
|---|---:|---:|---:|---:|
| Load | 0.0121 | 0.0271 | 0.0170 | 1.41× |
| Price | 5.19 | 7.44 | 10.04 | 1.93× |

Price error nearly doubles across the day. That is the expected and healthy
shape: the model is genuinely leaning on recent observations, so its advantage
decays as the forecast reaches further. A model whose error were flat across the
horizon would be relying on the calendar alone.

Load's mid-window bump is an artefact of issuing at 00:00 UTC — step 11 is around
local midday, the hardest part of the load curve, not the furthest ahead.

**In peak hours** (local 17:00–20:00, where the demand charge is usually set):

| Series | MAE, all hours | MAE, peak hours |
|---|---:|---:|
| Load | 0.0228 | 0.0253 |
| Price | 7.12 | 7.73 |

Both are modestly worse exactly where errors cost most.

## On percentage error metrics

The brief asks for MAPE "where mathematically appropriate" and sMAPE "where MAPE
is problematic". On this data the honest answer is that neither works for two of
the three series, so they are reported as `NaN` rather than computed over a
filtered subset — which would quietly change the question being answered.

- **Price** crosses zero: 484 negative hours plus many near zero. MAPE is
  undefined and sMAPE explodes.
- **PV** is exactly zero every night, so the same applies, and *any* metric
  averaged over all hours largely measures how well darkness is predicted.
  Daylight-only figures are reported alongside.
- **Load** stays well away from zero, so both are meaningful and are given.

MAE and RMSE are primary throughout: they are in the units the optimizer cares
about.

## Known limitations

- **The load series is a rescaled national aggregate**, which is smoother and
  more forecastable than a single building. Load accuracy here is optimistic for
  a real behind-the-meter site. The ablation in `docs/formulation.md` measures
  how much that actually matters to the cost, rather than leaving it as a
  caveat.
- **Point forecasts only.** No prediction intervals, so the optimizer cannot be
  made risk-aware. Probabilistic forecasting is listed as a stretch goal in the
  brief and is not attempted.
- **No exogenous weather inputs.** Temperature and irradiance forecasts would
  plainly help PV and load; the dataset does not carry them, and inventing them
  from the target would be leakage.
- **Retraining is not simulated.** Models are fitted once on the training window
  and never updated, so performance on the COVID-era `shift` split measures a
  stale model meeting a regime change — which is reported separately for exactly
  that reason.
