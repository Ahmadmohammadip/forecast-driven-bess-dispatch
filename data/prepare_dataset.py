"""Derive the committed site dataset from the raw OPSD snapshot.

    python data/download_opsd.py
    python data/prepare_dataset.py

What this does, and what it does not:

* It **slices** the OPSD hourly file to the window where a DE_LU day-ahead
  price actually exists (see WINDOW below), and keeps four series.
* It **rescales** national aggregates to a single site. The load and PV shapes
  are real measurements; the magnitudes are assumptions about a hypothetical
  site. Nothing here is a claim about a real facility.
* It **does not** smooth, denoise, or otherwise improve the data. Gaps are
  interpolated only where they are short, and every interpolated value is
  flagged in an `is_imputed` column so downstream code can exclude them.

The output, data/processed/site_hourly.csv, is committed. Re-running this
should reproduce it byte for byte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# The DE/AT bidding zone split on 2018-10-01: before that date there is no
# DE_LU price series at all, and the earlier DE_AT_LU series is a different
# market. Joining them would silently concatenate two price regimes, so the
# window simply starts where DE_LU does. It ends where OPSD stopped publishing.
WINDOW_START = "2018-10-01"
WINDOW_END = "2020-10-01"  # exclusive

# Site ratings. These define the reference case in configs/base.yaml.
SITE_PEAK_LOAD_MW = 1.0
SITE_PV_CAPACITY_MW = 0.8

# Gaps longer than this are left as NaN rather than invented.
MAX_INTERPOLATE_HOURS = 3

SOURCE_COLUMNS = {
    "DE_LU_price_day_ahead": "price_eur_mwh",
    "DE_LU_load_actual_entsoe_transparency": "_load_national_mw",
    "DE_LU_load_forecast_entsoe_transparency": "_load_forecast_national_mw",
    "DE_LU_solar_generation_actual": "_solar_national_mw",
}

DEFAULT_RAW = Path(__file__).parent / "raw" / "time_series_60min_singleindex.csv"
DEFAULT_OUT = Path(__file__).parent / "processed" / "site_hourly.csv"


def prepare(raw_path: Path = DEFAULT_RAW, out_path: Path = DEFAULT_OUT) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python data/download_opsd.py` first."
        )

    usecols = ["utc_timestamp", "cet_cest_timestamp", *SOURCE_COLUMNS]
    df = pd.read_csv(raw_path, usecols=usecols, parse_dates=["utc_timestamp"])
    df = df.rename(columns=SOURCE_COLUMNS).set_index("utc_timestamp").sort_index()

    df = df.loc[(df.index >= WINDOW_START) & (df.index < WINDOW_END)]
    print(f"window {WINDOW_START} .. {WINDOW_END} -> {len(df)} hourly rows")

    # Reindex onto a complete hourly grid so a missing *row* becomes a missing
    # *value*, which the gap report below can then see.
    full = pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC", name="utc_timestamp")
    missing_rows = len(full) - len(df)
    df = df.reindex(full)
    if missing_rows:
        print(f"  {missing_rows} timestamps absent from the source file")

    value_cols = list(SOURCE_COLUMNS.values())
    gaps_before = df[value_cols].isna().sum()

    # Interpolate short gaps only. limit_area="inside" refuses to extrapolate
    # past the first and last real observation.
    filled = df[value_cols].interpolate(
        method="time", limit=MAX_INTERPOLATE_HOURS, limit_area="inside"
    )
    imputed_mask = df[value_cols].isna() & filled.notna()
    df[value_cols] = filled

    print("  gaps per column (before -> after short-gap interpolation):")
    for col in value_cols:
        print(f"    {col:<30} {gaps_before[col]:>5} -> {int(df[col].isna().sum()):>5}")

    # --- rescale national aggregates to one site -------------------------
    # Normalize by the in-window maximum, so the site's peak load equals its
    # rating and its PV reaches nameplate exactly once. Both are shape-preserving
    # linear maps; neither changes forecastability.
    load_peak = df["_load_national_mw"].max()
    solar_peak = df["_solar_national_mw"].max()
    df["load_mw"] = SITE_PEAK_LOAD_MW * df["_load_national_mw"] / load_peak
    df["pv_mw"] = SITE_PV_CAPACITY_MW * df["_solar_national_mw"] / solar_peak
    # The TSO forecast is rescaled by the *load* factor, not its own, so it
    # stays comparable with load_mw as a forecast of the same quantity.
    df["tso_load_forecast_mw"] = (
        SITE_PEAK_LOAD_MW * df["_load_forecast_national_mw"] / load_peak
    )

    df["is_imputed"] = imputed_mask.any(axis=1)
    df["local_timestamp"] = df["cet_cest_timestamp"]

    out = df[
        [
            "local_timestamp",
            "load_mw",
            "pv_mw",
            "price_eur_mwh",
            "tso_load_forecast_mw",
            "is_imputed",
        ]
    ].round(6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, lineterminator="\n")
    print(f"\nwrote {out_path}  ({len(out)} rows, {out_path.stat().st_size / 1e6:.2f} MB)")
    clean = int((out.notna().all(axis=1) & ~out["is_imputed"]).sum())
    price = out["price_eur_mwh"]
    print(f"  complete rows (no NaN, not imputed): {clean}")
    print(f"  price EUR/MWh  min {price.min():.1f}  max {price.max():.1f}"
          f"  negative hours {int((price < 0).sum())}")
    print(f"  load MW        min {out['load_mw'].min():.3f}  max {out['load_mw'].max():.3f}")
    print(f"  pv MW          min {out['pv_mw'].min():.3f}  max {out['pv_mw'].max():.3f}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    prepare(args.raw, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
