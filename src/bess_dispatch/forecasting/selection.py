"""Choose a forecaster per series, on the validation window, by measurement.

The result is not what reputation predicts, which is the reason this module
exists rather than a hard-coded default: gradient boosting wins convincingly on
load and *loses to a naive baseline* on PV. Picking one model for all three
series would have been wrong twice.

Run it to regenerate the evidence:

    python -m bess_dispatch.forecasting.selection
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bess_dispatch.data.loaders import load_site_frame, split_frame
from bess_dispatch.forecasting.evaluate import compare, skill_score
from bess_dispatch.forecasting.features import FeatureSpec, daily_issue_times
from bess_dispatch.forecasting.models import (
    GradientBoostingForecaster,
    PersistenceForecaster,
    PublishedForecast,
    RidgeForecaster,
    WeeklyPersistenceForecaster,
    XGBoostForecaster,
)

TARGETS = ("load_mw", "pv_mw", "price_eur_mwh")

# Selected on the validation window (2019-10-01 to 2020-01-01) by MAE, with the
# training window ending 2019-10-01. Regenerate with the command above; the
# measured table is in docs/forecasting.md.
#
# Note what this says: the gradient-boosted model is the right choice for load
# and the wrong choice for the other two.
BEST_BY_TARGET: dict[str, str] = {
    "load_mw": "gradient_boosting",
    "pv_mw": "ridge",
    "price_eur_mwh": "ridge",
}

# XGBoost is behind an optional extra and is not in BEST_BY_TARGET, because it
# only justifies itself on one of the three series. Measured deltas against
# HistGradientBoostingRegressor on the validation window:
#   load   -7.9% MAE  (better)
#   price  -0.5% MAE  (a tie)
#   PV    +11.4% MAE  (worse)
# A compiled dependency for one series' worth of gain is the caller's call, so
# it is available and documented rather than default.
XGBOOST_JUSTIFIED_FOR = ("load_mw",)


def candidate_models(target: str, spec: FeatureSpec, *, include_xgboost: bool = False):
    models = [
        PersistenceForecaster(target=target, spec=spec),
        WeeklyPersistenceForecaster(target=target, spec=spec),
        RidgeForecaster(target=target, spec=spec),
        GradientBoostingForecaster(target=target, spec=spec),
    ]
    if include_xgboost:
        try:
            models.append(XGBoostForecaster(target=target, spec=spec))
        except ImportError:
            pass  # optional extra not installed; the comparison is still valid
    if target == "load_mw":
        models.append(PublishedForecast(target=target, spec=spec))
    return models


def run_selection(
    horizon: int = 24,
    spec: FeatureSpec | None = None,
    include_xgboost: bool = True,
) -> pd.DataFrame:
    """Fit on train, score on validation, for every target and candidate."""
    spec = spec or FeatureSpec()
    frame = load_site_frame()
    train = split_frame(frame, "train")

    train_issues = daily_issue_times(train, horizon, spec)
    validation_issues = [
        issue
        for issue in daily_issue_times(frame, horizon, spec)
        if pd.Timestamp("2019-10-01", tz="UTC")
        <= issue
        < pd.Timestamp("2020-01-01", tz="UTC")
    ]

    blocks = []
    for target in TARGETS:
        table, raw = compare(
            candidate_models(target, spec, include_xgboost=include_xgboost),
            frame,
            validation_issues,
            horizon,
            train_frame=train,
            train_issue_times=train_issues,
        )
        baseline = raw["persistence"]
        table["skill_vs_persistence"] = [
            skill_score(raw[name], baseline) for name in table.index
        ]
        table["target"] = target
        # "selected" means: the best model this project would actually ship by
        # default. That excludes the published TSO forecast, which is an
        # external benchmark rather than something this repo can produce, and
        # xgboost, which lives behind an optional extra. Both stay in the table
        # -- the comparison is the point -- but neither can win the default slot.
        eligible = table.drop(index=["published (TSO)", "xgboost"], errors="ignore")
        table["selected"] = table.index == eligible["MAE"].idxmin()
        table["eligible_as_default"] = table.index.isin(eligible.index)
        blocks.append(table.reset_index())

    return pd.concat(blocks, ignore_index=True).set_index(["target", "model"])


def main() -> int:
    results = run_selection()
    out_dir = Path(__file__).resolve().parents[3] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "forecast_selection.csv"
    results.to_csv(path)

    for target in TARGETS:
        block = results.loc[target]
        print(f"\n=== {target} ===")
        columns = [c for c in ("MAE", "RMSE", "R2", "skill_vs_persistence") if c in block]
        print(block[columns].round(4).to_string())
        print(f"  lowest MAE: {block['MAE'].idxmin()}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
