"""Forecast accuracy: metrics, backtests, and the breakdowns the brief asks for.

A note on percentage errors, because the brief asks for MAPE "where
mathematically appropriate" and sMAPE "where MAPE is problematic", and on this
data the honest answer is that neither works for two of the three series:

* **Price crosses zero.** 484 hours are negative and many more sit near zero, so
  MAPE is undefined and sMAPE explodes. Both are reported as NaN for price
  rather than computed on a filtered subset, which would silently change the
  question being answered.
* **PV is zero every night.** Roughly half of all hours have a true value of
  exactly zero. MAPE is undefined there too, and any metric averaged over all
  hours mostly measures how well the model predicts darkness. PV metrics are
  therefore also reported over **daylight hours only**.
* **Load** is comfortably bounded away from zero, so MAPE and sMAPE are both
  meaningful, and are reported.

MAE and RMSE are the primary metrics throughout, because they are in the units
the optimizer actually cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_dispatch.forecasting.features import LOCAL_TZ

# Hours (local) treated as the evening peak, where a load error costs most:
# it is when the demand charge is usually set and when prices peak.
PEAK_HOURS = (17, 18, 19, 20)

# A PV hour counts as daylight if the *actual* generation exceeded this.
DAYLIGHT_THRESHOLD_MW = 1e-3


def _safe(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual has shape {actual.shape} but predicted has {predicted.shape}"
        )
    keep = np.isfinite(actual) & np.isfinite(predicted)
    return actual[keep], predicted[keep]


def mae(actual, predicted) -> float:
    a, p = _safe(actual, predicted)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual, predicted) -> float:
    a, p = _safe(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def bias(actual, predicted) -> float:
    """Mean signed error. A model can have good MAE and still lean one way."""
    a, p = _safe(actual, predicted)
    return float(np.mean(p - a)) if a.size else float("nan")


def mape(actual, predicted, *, min_denominator: float = 1e-9) -> float:
    """Mean absolute percentage error. NaN if any true value is at or near zero."""
    a, p = _safe(actual, predicted)
    if not a.size or np.any(np.abs(a) < min_denominator):
        return float("nan")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def smape(actual, predicted, *, min_denominator: float = 1e-9) -> float:
    """Symmetric MAPE. Still NaN where the series crosses zero -- see the module docstring."""
    a, p = _safe(actual, predicted)
    if not a.size:
        return float("nan")
    denominator = (np.abs(a) + np.abs(p)) / 2
    if np.any(denominator < min_denominator):
        return float("nan")
    return float(np.mean(np.abs(a - p) / denominator) * 100)


def r2(actual, predicted) -> float:
    a, p = _safe(actual, predicted)
    if a.size < 2:
        return float("nan")
    total = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - p) ** 2) / total) if total > 0 else float("nan")


def metrics(actual, predicted, *, target: str | None = None) -> dict[str, float]:
    """All metrics at once. Percentage metrics are suppressed where meaningless."""
    result = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "bias": bias(actual, predicted),
        "R2": r2(actual, predicted),
        "MAPE_%": mape(actual, predicted),
        "sMAPE_%": smape(actual, predicted),
        "n": int(np.isfinite(np.asarray(actual, dtype=float)).sum()),
    }
    if target == "price_eur_mwh":
        # Not a computation failure -- a statement that the question is ill-posed
        # for a series that crosses zero.
        result["MAPE_%"] = float("nan")
        result["sMAPE_%"] = float("nan")
    return result


def backtest(
    forecaster,
    frame: pd.DataFrame,
    issue_times,
    horizon: int = 24,
) -> pd.DataFrame:
    """Run one forecaster over many issue times.

    Returns tidy rows: issue_time, target_time, horizon_step, actual, predicted.
    Issue times whose history is incomplete are skipped and counted rather than
    imputed; the count is attached to the frame's `.attrs`.
    """
    records: list[pd.DataFrame] = []
    skipped = 0
    for issue_time in issue_times:
        try:
            predicted = forecaster.predict(frame, issue_time, horizon)
        except ValueError:
            skipped += 1
            continue
        actual = frame[forecaster.target].reindex(predicted.index)
        records.append(
            pd.DataFrame(
                {
                    "issue_time": issue_time,
                    "target_time": predicted.index,
                    "horizon_step": np.arange(horizon),
                    "actual": actual.to_numpy(),
                    "predicted": predicted.to_numpy(),
                }
            )
        )

    if not records:
        raise ValueError(
            f"{forecaster.name} produced no forecasts across {len(issue_times)} issue "
            "times -- every one lacked usable history"
        )
    out = pd.concat(records, ignore_index=True)
    out.attrs["skipped_issue_times"] = skipped
    out.attrs["model"] = forecaster.name
    out.attrs["target"] = forecaster.target
    return out


def summarise(results: pd.DataFrame, target: str | None = None) -> dict[str, float]:
    """Headline metrics for one backtest."""
    target = target or results.attrs.get("target")
    clean = results.dropna(subset=["actual", "predicted"])
    summary = metrics(clean["actual"], clean["predicted"], target=target)
    summary["skipped_issue_times"] = float(results.attrs.get("skipped_issue_times", 0))
    return summary


def summarise_daylight(results: pd.DataFrame) -> dict[str, float]:
    """PV metrics over daylight hours only.

    Averaged over all hours, PV error mostly measures how well the model
    predicts that it is dark. This is the number that means something.
    """
    lit = results[results["actual"] > DAYLIGHT_THRESHOLD_MW]
    if lit.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
    return metrics(lit["actual"], lit["predicted"], target="pv_mw")


def error_by_hour(results: pd.DataFrame) -> pd.DataFrame:
    """MAE and bias by local hour of day — the brief's "error by hour"."""
    working = results.dropna(subset=["actual", "predicted"]).copy()
    stamps = pd.DatetimeIndex(working["target_time"])
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    working["hour"] = stamps.tz_convert(LOCAL_TZ).hour
    grouped = working.groupby("hour")
    return pd.DataFrame(
        {
            "MAE": grouped.apply(
                lambda g: mae(g["actual"], g["predicted"]), include_groups=False
            ),
            "bias": grouped.apply(
                lambda g: bias(g["actual"], g["predicted"]), include_groups=False
            ),
            "n": grouped.size(),
        }
    )


def error_by_horizon_step(results: pd.DataFrame) -> pd.DataFrame:
    """MAE against how far ahead the forecast reaches.

    A day-ahead forecast issued at midnight should degrade with horizon step. If
    it does not, the model is probably leaning entirely on calendar features and
    ignoring recent history.
    """
    working = results.dropna(subset=["actual", "predicted"])
    grouped = working.groupby("horizon_step")
    return pd.DataFrame(
        {
            "MAE": grouped.apply(
                lambda g: mae(g["actual"], g["predicted"]), include_groups=False
            ),
            "n": grouped.size(),
        }
    )


def error_in_peak_hours(results: pd.DataFrame, peak_hours=PEAK_HOURS) -> dict[str, float]:
    """Accuracy restricted to the evening peak, where errors cost most."""
    working = results.dropna(subset=["actual", "predicted"]).copy()
    stamps = pd.DatetimeIndex(working["target_time"])
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    working["hour"] = stamps.tz_convert(LOCAL_TZ).hour
    peak = working[working["hour"].isin(peak_hours)]
    if peak.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n": 0}
    return {
        "MAE": mae(peak["actual"], peak["predicted"]),
        "RMSE": rmse(peak["actual"], peak["predicted"]),
        "n": int(len(peak)),
    }


def compare(
    forecasters,
    frame: pd.DataFrame,
    issue_times,
    horizon: int = 24,
    train_frame: pd.DataFrame | None = None,
    train_issue_times=None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit each forecaster, backtest it, and tabulate.

    Returns the comparison table and the raw backtests, keyed by model name, so
    that per-hour and peak-hour breakdowns can be taken without re-running.
    """
    rows, raw = [], {}
    for forecaster in forecasters:
        if train_frame is not None:
            forecaster.fit(train_frame, train_issue_times, horizon)
        else:
            forecaster.fit(frame, issue_times, horizon)

        results = backtest(forecaster, frame, issue_times, horizon)
        raw[forecaster.name] = results

        summary = summarise(results)
        summary["model"] = forecaster.name
        if forecaster.target == "pv_mw":
            summary["MAE_daylight"] = summarise_daylight(results)["MAE"]
        summary["MAE_peak"] = error_in_peak_hours(results)["MAE"]
        rows.append(summary)

    table = pd.DataFrame(rows).set_index("model")
    ordered = [
        column
        for column in (
            "MAE",
            "RMSE",
            "MAE_daylight",
            "MAE_peak",
            "bias",
            "R2",
            "MAPE_%",
            "sMAPE_%",
            "n",
            "skipped_issue_times",
        )
        if column in table.columns
    ]
    return table[ordered], raw


def skill_score(results: pd.DataFrame, baseline: pd.DataFrame) -> float:
    """Fractional MAE improvement over a baseline. Negative means worse.

    Reported because an absolute MAE says nothing about whether a model earned
    its complexity.
    """
    model_mae = mae(results["actual"], results["predicted"])
    base_mae = mae(baseline["actual"], baseline["predicted"])
    if not np.isfinite(base_mae) or base_mae == 0:
        return float("nan")
    return float(1 - model_mae / base_mae)
