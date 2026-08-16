"""Reading the prepared dataset into validated objects.

Nothing downstream of this module touches a file path. The Pyomo builder takes
a `ForecastResult` and a `SiteConfig`; the forecasters take frames that came
from here. That separation is what keeps "where did this number come from"
answerable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bess_dispatch.data.schema import TimeSeriesData
from bess_dispatch.data.synthetic import make_synthetic_site

REQUIRED_COLUMNS = ("load_mw", "pv_mw", "price_eur_mwh")

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "site_hourly.csv"
)

# Chronological, never random -- a random split over a time series leaks the
# future into the past. Ends are exclusive. See data/DATA_DICTIONARY.md for why
# the test window stops before March 2020.
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2018-10-01", "2019-10-01"),
    "validation": ("2019-10-01", "2020-01-01"),
    "test": ("2020-01-01", "2020-03-01"),
    "shift": ("2020-03-01", "2020-10-01"),
}


def load_site_frame(path: str | Path | None = None) -> pd.DataFrame:
    """Read the prepared dataset as a UTC-indexed frame, unmodified."""
    path = Path(path) if path is not None else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It should be committed; if you are working from a "
            "partial checkout, regenerate it with "
            "`python data/download_opsd.py && python data/prepare_dataset.py`."
        )
    frame = pd.read_csv(path, index_col="utc_timestamp", parse_dates=["utc_timestamp"])
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    return frame.sort_index()


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Slice one of the named chronological splits out of a frame."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    start, end = SPLITS[split]
    window = frame.loc[(frame.index >= start) & (frame.index < end)]
    if window.empty:
        raise ValueError(
            f"split {split!r} ({start} to {end}) selected no rows from a frame "
            f"covering {frame.index[0]} to {frame.index[-1]}"
        )
    return window


def complete_days(frame: pd.DataFrame, *, allow_imputed: bool = False) -> pd.DataFrame:
    """Keep only whole 24-hour days with no missing and (by default) no imputed values.

    Dispatch is solved a day at a time, so a day with a hole in it is not a
    partially usable day — it is an unusable one. Dropping at day granularity
    keeps the horizon intact rather than silently shortening it.
    """
    usable = frame[list(REQUIRED_COLUMNS)].notna().all(axis=1)
    if not allow_imputed and "is_imputed" in frame.columns:
        usable &= ~frame["is_imputed"].astype(bool)

    by_day = usable.groupby(frame.index.floor("D"))
    whole = by_day.transform("sum").eq(24) & by_day.transform("size").eq(24)
    return frame.loc[usable & whole]


def frame_to_timeseries(frame: pd.DataFrame) -> TimeSeriesData:
    """Convert a prepared frame into a validated `TimeSeriesData`."""
    index = frame.index
    if index.tz is not None:
        # Hand numpy naive UTC: it has no timezone-aware dtype, and every
        # timestamp in this project is UTC by convention anyway.
        index = index.tz_convert("UTC").tz_localize(None)
    return TimeSeriesData(
        timestamps=index.to_numpy(),
        load_mw=frame["load_mw"].to_numpy(dtype=float),
        pv_mw=frame["pv_mw"].to_numpy(dtype=float),
        price_eur_mwh=frame["price_eur_mwh"].to_numpy(dtype=float),
    )


def load_timeseries(
    path: str | Path | None = None,
    *,
    split: str | None = None,
    whole_days_only: bool = True,
) -> TimeSeriesData:
    """Load the prepared dataset, optionally restricted to one split."""
    frame = load_site_frame(path)
    if split is not None:
        frame = split_frame(frame, split)
    if whole_days_only:
        frame = complete_days(frame)
    return frame_to_timeseries(frame)


def load_synthetic_timeseries(
    periods: int = 24 * 60, *, seed: int | None = None, **kwargs
) -> TimeSeriesData:
    """A seeded synthetic stand-in, for tests and offline use.

    Not used for any reported result — see `bess_dispatch.data.synthetic`.
    """
    if seed is not None:
        kwargs["seed"] = seed
    return frame_to_timeseries(make_synthetic_site(periods, **kwargs))


def iter_days(data: TimeSeriesData, periods_per_day: int = 24):
    """Yield `(day_index, TimeSeriesData)` for each whole day in `data`.

    Any trailing partial day is dropped rather than solved over a short horizon,
    which would make its cost incomparable with the others.
    """
    n_days = len(data) // periods_per_day
    for day in range(n_days):
        start = day * periods_per_day
        yield day, data.slice(start, start + periods_per_day)


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    """Summary used by the EDA notebook and the README, in one place."""
    rows = []
    for column in REQUIRED_COLUMNS:
        series = frame[column].dropna()
        rows.append(
            {
                "series": column,
                "n": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p05": series.quantile(0.05),
                "median": series.median(),
                "p95": series.quantile(0.95),
                "max": series.max(),
            }
        )
    summary = pd.DataFrame(rows).set_index("series")
    summary.loc["price_eur_mwh", "negative_hours"] = int(
        (frame["price_eur_mwh"] < 0).sum()
    )
    return summary


def daily_price_spread(frame: pd.DataFrame) -> pd.Series:
    """Max-minus-min wholesale price within each local day.

    The single most useful number for judging whether arbitrage is worth
    anything: it bounds what one perfect charge/discharge cycle can earn.
    """
    price = frame["price_eur_mwh"].dropna()
    by_day = price.groupby(price.index.floor("D"))
    spread = by_day.max() - by_day.min()
    return spread[by_day.size().eq(24).reindex(spread.index, fill_value=False)]


def train_test_report(path: str | Path | None = None) -> pd.DataFrame:
    """Row counts and completeness per split — what the data dictionary claims."""
    frame = load_site_frame(path)
    rows = []
    for name in SPLITS:
        window = split_frame(frame, name)
        core = window[list(REQUIRED_COLUMNS)]
        rows.append(
            {
                "split": name,
                "start": SPLITS[name][0],
                "end": SPLITS[name][1],
                "rows": len(window),
                "complete": int(core.notna().all(axis=1).sum()),
                "imputed": int(window.get("is_imputed", pd.Series(dtype=bool)).sum()),
                "whole_days": len(complete_days(window)) // 24,
                "mean_price": float(np.nanmean(window["price_eur_mwh"])),
            }
        )
    return pd.DataFrame(rows).set_index("split")
