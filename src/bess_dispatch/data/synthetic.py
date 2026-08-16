"""Seeded synthetic site data -- a fallback, not the primary dataset.

Why this is *not* what the headline numbers are computed from: the central
result of this project is the gap between forecast-driven and perfect-foresight
dispatch. On synthetic data that gap measures whatever noise this file injects,
not how hard the real series are to forecast. Using it for headline numbers
would make the result circular.

It exists for three honest uses:

1. tests, which must run in milliseconds and must not depend on a 1.5 MB CSV;
2. CI, which must not reach the network;
3. anyone who wants to run the pipeline before downloading anything.

The coefficients below were **calibrated against the real prepared dataset's
summary statistics**, not guessed. Over a synthetic year they reproduce:

| statistic                | real  | synthetic |
|--------------------------|-------|-----------|
| PV capacity factor       | 0.152 | 0.153     |
| negative-price hours     | 2.76% | 2.75%     |
| mean price (EUR/MWh)     | 35.8  | 36.6      |
| price std dev            | 18.1  | 17.1      |
| mean load (MW)           | 0.72  | 0.73      |

Matching the marginal distributions does **not** make this a substitute for the
real series. The temporal dependence structure -- which is what a forecaster
actually learns -- is far simpler here than in reality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEED = 20260816

# Calibrated -- see the table in the module docstring.
_LOAD_BASE = 0.74
_PV_GAIN = 1.1
_PV_EXPONENT = 2.4
_PRICE_INTERCEPT = 4.0
_PRICE_SLOPE = 56.0
_PRICE_PV_DEPRESSION = 1.00
_PRICE_NOISE_SD = 9.0


def make_synthetic_site(
    periods: int = 24 * 60,
    *,
    start: str = "2019-01-01",
    peak_load_mw: float = 1.0,
    pv_capacity_mw: float = 0.8,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate `periods` hours of synthetic load, PV and price.

    Returns a frame with the same columns, dtypes and index as the real
    prepared dataset, so downstream code cannot tell them apart structurally.

    Note the default window starts in January: a 60-day default sees almost no
    PV and no negative prices. Ask for `periods=8760` if you need a full year's
    behaviour.
    """
    if periods < 24:
        raise ValueError(f"periods must be at least 24 (one day), got {periods}")

    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC", name="utc_timestamp")
    hour = index.hour.to_numpy()
    dayofweek = index.dayofweek.to_numpy()
    dayofyear = index.dayofyear.to_numpy()

    # --- load: twin daily peaks, damped at weekends, mild winter seasonality
    daily = (
        _LOAD_BASE
        + 0.20 * np.sin((hour - 7) * np.pi / 12)
        + 0.14 * np.exp(-((hour - 19) ** 2) / 8)
    )
    weekend = np.where(dayofweek >= 5, 0.82, 1.0)
    seasonal = 1.0 + 0.08 * np.cos(2 * np.pi * dayofyear / 365.25)
    load = peak_load_mw * daily * weekend * seasonal
    load = np.clip(load * (1 + rng.normal(0, 0.03, periods)), 0.05 * peak_load_mw, None)

    # --- pv: daylight window widening in summer, scaled by a cloud factor
    day_length = 5.0 + 2.6 * np.cos(2 * np.pi * (dayofyear - 172) / 365.25)
    clear_sky = np.clip(np.cos((hour - 12) * np.pi / (2 * day_length)), 0, None) ** _PV_EXPONENT
    clear_sky[(hour < 4) | (hour > 20)] = 0.0
    # Cloud cover persists within a day rather than resampling every hour --
    # independent hourly noise would make PV far easier to forecast than it is.
    cloud = np.repeat(rng.beta(5, 2, periods // 24 + 1), 24)[:periods]
    pv = pv_capacity_mw * _PV_GAIN * clear_sky * cloud
    pv = np.clip(pv * (1 + rng.normal(0, 0.05, periods)), 0, pv_capacity_mw)

    # --- price: driven by residual load, so it correlates with load and PV the
    # way the real series does, and goes negative when PV floods a light hour
    residual = load / peak_load_mw - (pv / pv_capacity_mw) * _PRICE_PV_DEPRESSION
    price = _PRICE_INTERCEPT + _PRICE_SLOPE * residual
    price = price + rng.normal(0, _PRICE_NOISE_SD, periods)
    spike = rng.random(periods) < 0.004
    price[spike] += rng.uniform(40, 120, int(spike.sum()))

    return pd.DataFrame(
        {
            "local_timestamp": index.tz_convert("Europe/Berlin").strftime("%Y-%m-%dT%H:%M:%S%z"),
            "load_mw": load,
            "pv_mw": pv,
            "price_eur_mwh": price,
            # A deliberately imperfect "external" forecast, so code paths that
            # consume the TSO benchmark column have something to consume.
            "tso_load_forecast_mw": load * (1 + rng.normal(0, 0.025, periods)),
            "is_imputed": False,
        },
        index=index,
    ).round(6)


if __name__ == "__main__":
    year = make_synthetic_site(periods=8760)
    print(f"{len(year)} hours from {year.index[0]} to {year.index[-1]}\n")
    print(f"  PV capacity factor    {year.pv_mw.mean() / 0.8:.4f}   (real 0.152)")
    print(f"  negative-price hours  {(year.price_eur_mwh < 0).mean():.4f}   (real 0.0276)")
    print(f"  mean price EUR/MWh    {year.price_eur_mwh.mean():.1f}     (real 35.8)")
    print(f"  price std dev         {year.price_eur_mwh.std():.1f}     (real 18.1)")
    print(f"  mean load MW          {year.load_mw.mean():.3f}    (real 0.720)")
