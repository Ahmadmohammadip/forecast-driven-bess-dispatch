"""Day-ahead benchmark: what is the forecast actually worth?

Every arm is solved over the same days, with the same site, and priced against
the same actuals by the same function. The only thing that varies is what the
controller was allowed to know when it decided.

The arms, and why each one is here:

| Arm | Knows | Answers |
|---|---|---|
| no battery | — | What does doing nothing cost? |
| rule-based | present load and PV | Does *optimizing* beat the obvious controller? |
| forecast | forecasts of all three | What does this system actually deliver? |
| actual price | true price, forecast load/PV | How much of the loss is price error? |
| actual load | true load, forecast price/PV | How much is load error? |
| actual PV | true PV, forecast price/load | How much is PV error? |
| perfect foresight | everything | What is the ceiling? |

The three ablation arms exist because "forecasts cost you X" is not actionable.
Knowing *which* forecast costs you X tells you where to spend effort, and on
this data the answer is not evenly split.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from bess_dispatch.data.loaders import load_site_frame, split_frame
from bess_dispatch.data.schema import SiteConfig
from bess_dispatch.forecasting.features import FeatureSpec, daily_issue_times
from bess_dispatch.forecasting.interface import (
    ForecastResult,
    actuals_for,
    forecast_horizon,
)
from bess_dispatch.forecasting.models import build_forecaster
from bess_dispatch.forecasting.selection import BEST_BY_TARGET
from bess_dispatch.optimization.builder import build_dispatch_model, build_no_battery_model
from bess_dispatch.optimization.rules import rule_based_schedule
from bess_dispatch.optimization.solve import evaluate_schedule, solve_dispatch

TARGETS = ("load_mw", "pv_mw", "price_eur_mwh")


@dataclass(frozen=True)
class Arm:
    """One controller under test."""

    name: str
    kind: str  # no_battery | rule | optimized
    actual_series: tuple[str, ...] = ()  # which series are handed their true values
    perfect: bool = False


DEFAULT_ARMS: tuple[Arm, ...] = (
    Arm("no battery", "no_battery"),
    Arm("rule-based", "rule"),
    Arm("forecast", "optimized"),
    Arm("forecast + actual price", "optimized", actual_series=("price",)),
    Arm("forecast + actual load", "optimized", actual_series=("load",)),
    Arm("forecast + actual PV", "optimized", actual_series=("pv",)),
    Arm("perfect foresight", "optimized", perfect=True),
)


def fit_forecasters(
    train_frame: pd.DataFrame,
    horizon: int = 24,
    spec: FeatureSpec | None = None,
    models: dict[str, str] | None = None,
) -> dict[str, object]:
    """Fit one forecaster per series, using the validation-selected defaults."""
    spec = spec or FeatureSpec()
    models = models or BEST_BY_TARGET
    issue_times = daily_issue_times(train_frame, horizon, spec)
    return {
        target: build_forecaster(models[target], target, spec).fit(
            train_frame, issue_times, horizon
        )
        for target in TARGETS
    }


def _schedule_for_arm(
    arm: Arm,
    forecast: ForecastResult,
    truth,
    site: SiteConfig,
    objective: str,
    solver_name: str,
):
    if arm.kind == "no_battery":
        model = build_no_battery_model(
            ForecastResult.from_actuals(truth), site, objective=objective
        )
        return solve_dispatch(model, solver_name)
    if arm.kind == "rule":
        from bess_dispatch.optimization.builder import OBJECTIVES

        return rule_based_schedule(truth, site, OBJECTIVES[objective])

    plan = ForecastResult.from_actuals(truth) if arm.perfect else forecast
    if arm.actual_series:
        plan = plan.with_actual(truth, *arm.actual_series)
    return solve_dispatch(
        build_dispatch_model(plan, site, objective=objective), solver_name
    )


def run_benchmark(
    site: SiteConfig,
    split: str = "test",
    objective: str = "cost_degradation_demand",
    horizon: int = 24,
    arms: tuple[Arm, ...] = DEFAULT_ARMS,
    spec: FeatureSpec | None = None,
    frame: pd.DataFrame | None = None,
    fitted: dict | None = None,
    solver_name: str = "appsi_highs",
    max_days: int | None = None,
) -> pd.DataFrame:
    """Solve every arm on every whole day of `split`, priced against actuals.

    Returns one row per (day, arm). Days whose forecast cannot be produced —
    a gap in the history before the issue time — are skipped for *every* arm,
    so the arms always cover an identical set of days and their totals stay
    comparable.
    """
    spec = spec or FeatureSpec()
    frame = frame if frame is not None else load_site_frame()
    fitted = fitted or fit_forecasters(split_frame(frame, "train"), horizon, spec)

    window = split_frame(frame, split)
    issue_times = [
        issue
        for issue in daily_issue_times(frame, horizon, spec)
        if window.index[0] <= issue <= window.index[-1] - pd.Timedelta(hours=horizon - 1)
    ]
    if max_days is not None:
        issue_times = issue_times[:max_days]

    records: list[dict] = []
    skipped = 0

    # State of charge carries from one day to the next, per arm.
    #
    # This is not a refinement, it is a correctness fix. Restarting every day at
    # the configured initial SOC hands any arm that ends a day depleted a free
    # refill overnight. The rule-based controller does exactly that in winter --
    # it discharges 0.38 MWh a day and never recharges, because PV never exceeds
    # load -- so it was being gifted 0.4 MWh every morning and appeared to
    # capture 309% of the perfect-foresight saving. An arm cannot beat perfect
    # foresight; the number was an artefact of teleporting energy into the
    # battery. Arms that enforce terminal SOC are unaffected, since they end
    # each day where they started.
    soc_state = {arm.name: site.battery.initial_soc_frac for arm in arms}

    for issue_time in issue_times:
        try:
            forecast = forecast_horizon(fitted, frame, issue_time, horizon)
            truth = actuals_for(frame, forecast)
        except ValueError:
            skipped += 1
            continue

        for arm in arms:
            day_site = replace(
                site,
                battery=replace(
                    site.battery, initial_soc_frac=soc_state[arm.name]
                ),
            )
            schedule = _schedule_for_arm(
                arm, forecast, truth, day_site, objective, solver_name
            )
            scored = evaluate_schedule(schedule, truth, day_site)
            # Clamp into the band before carrying. The replay arithmetic lands
            # on 0.09999999999999998 against a floor of 0.1, which Battery
            # rightly rejects; that is float noise, not a physics violation.
            soc_state[arm.name] = float(
                np.clip(
                    scored["terminal_soc_mwh"] / day_site.battery.energy_capacity_mwh,
                    day_site.battery.soc_min_frac,
                    day_site.battery.soc_max_frac,
                )
            )
            records.append(
                {
                    "issue_time": issue_time,
                    "arm": arm.name,
                    "realised_cost_eur": scored["realised_cost_eur"],
                    "planned_cost_eur": schedule.total_cost_eur,
                    "peak_import_mw": scored["peak_import_mw"],
                    "grid_import_mwh": scored["grid_import_mwh"],
                    "grid_export_mwh": scored["grid_export_mwh"],
                    "curtailed_mwh": scored["curtailed_mwh"],
                    "throughput_mwh": scored["throughput_mwh"],
                    "wholesale_energy_cost_eur": scored["wholesale_energy_cost_eur"],
                    "terminal_soc_mwh": scored["terminal_soc_mwh"],
                    "clipped_periods": scored["clipped_periods"],
                    "mean_soc_mwh": float(schedule.soc_mwh.mean()),
                    "solve_time_s": schedule.solve_time_s,
                }
            )

    if not records:
        raise ValueError(f"no usable days in split {split!r}")

    results = pd.DataFrame(records)
    results.attrs["skipped_days"] = skipped
    results.attrs["objective"] = objective
    results.attrs["split"] = split
    return results


def summarise_benchmark(results: pd.DataFrame, baseline_arm: str = "no battery") -> pd.DataFrame:
    """Totals per arm, plus savings against the no-battery baseline.

    `value_captured` is the share of the perfect-foresight saving that an arm
    actually achieved — the number the whole project exists to produce.
    """
    grouped = results.groupby("arm")
    table = pd.DataFrame(
        {
            "days": grouped.size(),
            "total_cost_eur": grouped["realised_cost_eur"].sum(),
            "peak_import_mw": grouped["peak_import_mw"].max(),
            "throughput_mwh": grouped["throughput_mwh"].sum(),
            "curtailed_mwh": grouped["curtailed_mwh"].sum(),
            "wholesale_cost_eur": grouped["wholesale_energy_cost_eur"].sum(),
            "mean_solve_s": grouped["solve_time_s"].mean(),
        }
    )

    baseline = table.loc[baseline_arm, "total_cost_eur"]
    table["saving_eur"] = baseline - table["total_cost_eur"]
    table["saving_pct"] = 100 * table["saving_eur"] / baseline
    # Second denominator: the same saving against the wholesale energy
    # component alone. Network charges and levies ride along with every MWh
    # whenever it is taken, so the battery cannot touch them, and expressing
    # the saving against the full delivered bill understates what the
    # controller is doing by roughly the ratio of the two.
    wholesale_baseline = table.loc[baseline_arm, "wholesale_cost_eur"]
    table["saving_pct_wholesale"] = 100 * table["saving_eur"] / wholesale_baseline

    if "perfect foresight" in table.index:
        ceiling = table.loc["perfect foresight", "saving_eur"]
        table["value_captured_pct"] = (
            100 * table["saving_eur"] / ceiling if ceiling else np.nan
        )

    order = [arm for arm in results["arm"].drop_duplicates() if arm in table.index]
    return table.loc[order]


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Attribute the forecast-driven shortfall to each series.

    Reads as: "if this one forecast were perfect, how much of the gap between
    the forecast-driven arm and perfect foresight would close?"
    """
    needed = {"forecast", "perfect foresight"}
    if not needed <= set(summary.index):
        raise KeyError(f"summary must contain {sorted(needed)} to attribute the gap")

    forecast_cost = summary.loc["forecast", "total_cost_eur"]
    perfect_cost = summary.loc["perfect foresight", "total_cost_eur"]
    gap = forecast_cost - perfect_cost

    rows = []
    for series, arm in (
        ("price", "forecast + actual price"),
        ("load", "forecast + actual load"),
        ("pv", "forecast + actual PV"),
    ):
        if arm not in summary.index:
            continue
        recovered = forecast_cost - summary.loc[arm, "total_cost_eur"]
        rows.append(
            {
                "series made perfect": series,
                "cost_eur": summary.loc[arm, "total_cost_eur"],
                "gap_closed_eur": recovered,
                "gap_closed_pct": 100 * recovered / gap if gap else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("series made perfect")
    table.attrs["total_gap_eur"] = gap
    return table


def main() -> int:
    from bess_dispatch.config import load_config

    config = load_config()
    site = config.site()
    results = run_benchmark(site, objective=config.objective)

    out_dir = Path(__file__).resolve().parents[3] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "benchmark_daily.csv", index=False)

    summary = summarise_benchmark(results)
    summary.to_csv(out_dir / "benchmark_summary.csv")
    print(summary.round(3).to_string())
    print()
    print(ablation_table(summary).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
