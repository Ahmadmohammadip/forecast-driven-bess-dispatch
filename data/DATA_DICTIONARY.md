# Data dictionary

`data/processed/site_hourly.csv` — 17,544 hourly rows, 2018-10-01 00:00 UTC to
2020-09-30 23:00 UTC. Committed, so a clone runs without downloading anything.

| Column | Unit | Type | Description |
|---|---|---|---|
| `utc_timestamp` | — | ISO 8601, `+00:00` | Index. UTC, hourly, gapless, monotonic. |
| `local_timestamp` | — | ISO 8601, `+0100`/`+0200` | Same instant in Europe/Berlin. Carried so calendar features use *local* hour-of-day, which is what drives both human behaviour and solar position. |
| `load_mw` | MW | float | Site electrical demand. |
| `pv_mw` | MW | float | Site PV generation. |
| `price_eur_mwh` | EUR/MWh | float | Day-ahead wholesale price, DE/LU bidding zone. Basis for the import tariff. |
| `tso_load_forecast_mw` | MW | float | The transmission operator's own published day-ahead load forecast, rescaled identically to `load_mw`. **An external benchmark, not a model input.** |
| `is_imputed` | — | bool | True if any value in the row was filled by interpolation. |

## Units and conventions

- Power in **MW**, energy in **MWh**, price in **EUR/MWh**, all at a one-hour
  step, so a MW value and its MWh contribution are numerically equal.
- Timestamps are **timezone-aware UTC**. Nothing in this project uses a naive
  timestamp; local time is derived, never assumed.
- Positive `load_mw` is consumption, positive `pv_mw` is generation. Grid
  import/export sign conventions belong to the model, not the data — see
  `docs/formulation.md`.

## Provenance and the rescaling step

Source is Open Power System Data, hourly time series, snapshot `2020-10-06`
(see `ATTRIBUTION.md`). Four of its 300 columns are used:

| OPSD column | becomes |
|---|---|
| `DE_LU_price_day_ahead` | `price_eur_mwh` (unchanged) |
| `DE_LU_load_actual_entsoe_transparency` | `load_mw` (rescaled) |
| `DE_LU_solar_generation_actual` | `pv_mw` (rescaled) |
| `DE_LU_load_forecast_entsoe_transparency` | `tso_load_forecast_mw` (rescaled) |

**The load and PV series are national aggregates rescaled to one site.** Each is
divided by its in-window maximum and multiplied by a site rating (1.0 MW peak
load, 0.8 MWp PV). The *shapes* are real measurements; the *magnitudes* are
assumptions about a hypothetical site. Price is not rescaled at all.

This has a consequence that matters, and it is not hidden anywhere in this repo:
**a national load curve is smoother, and therefore easier to forecast, than a
single building's.** Load-forecast accuracy reported here is optimistic for a
real behind-the-meter site. Rather than caveat it and move on, the evaluation
runs an ablation isolating which of the three forecasts the money actually
depends on.

## Window

2018-10-01 is not a convenience cutoff. The German/Austrian bidding zone split on
that date, so a `DE_LU` price series does not exist before it. The earlier
`DE_AT_LU` series covers a *different* market; concatenating the two would
silently join two price regimes, so it is not done.

## Missing values

Gaps of up to **three consecutive hours** are filled by time-weighted linear
interpolation; anything longer is left as `NaN` rather than invented.
Extrapolation past the first or last real observation is refused outright. Every
filled row is flagged in `is_imputed`.

What that leaves:

| Column | NaN remaining | Where |
|---|---|---|
| `price_eur_mwh` | 1 | final hour of the window |
| `load_mw` | 8 | a six-hour outage on 2019-02-03, plus two isolated hours |
| `pv_mw` | 5 | first five hours of the window, before the series begins |
| `tso_load_forecast_mw` | 822 | mostly late 2018, when publication was patchy |

80 rows carry at least one interpolated value. Code that needs clean data filters
on `is_imputed` and on completeness rather than assuming.

## Outliers

**None are removed.** The extremes here are real market events, not sensor
faults: 484 negative-price hours (2.8%), a minimum of −90 EUR/MWh and a maximum
of 200 EUR/MWh. Negative prices are precisely when a battery earns by charging,
so discarding them would delete the phenomenon under study.

## Splits

Chronological, never random — a random split over a time series leaks the future
into the past.

| Split | Range | Hours | Complete | Purpose |
|---|---|---:|---:|---|
| train | 2018-10-01 → 2019-10-01 | 8,760 | 8,747 | Fit forecasters |
| validation | 2019-10-01 → 2020-01-01 | 2,208 | 2,208 | Model selection |
| test | 2020-01-01 → 2020-03-01 | 1,440 | 1,440 | **Headline numbers.** Zero imputed rows |
| shift | 2020-03-01 → 2020-10-01 | 5,136 | 5,135 | COVID-era distribution shift, reported separately |

The test window stops at 2020-03-01 deliberately. European load and prices break
structurally in March 2020; folding that into the headline would confound
forecast quality with a once-in-a-decade shock. It is examined on its own
instead, because a forecaster meeting a regime change is a real thing that
happens.

## Synthetic fallback

`data/synthetic.py` generates a seeded stand-in with the same columns, calibrated
to reproduce the real data's marginal statistics (capacity factor, negative-price
share, price mean and spread). It exists for tests, CI, and running offline. It
is **not** used for any reported result: on synthetic data the gap between
forecast-driven and perfect-foresight dispatch would measure the generator's own
injected noise rather than genuine forecasting difficulty.
