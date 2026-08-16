"""The brief's section 14 comparison: day-ahead against rolling, both arms.

Four controllers over identical hours:

* **no battery** — the do-nothing floor.
* **day-ahead forecast** — one plan per day, made at midnight, executed whole.
* **rolling forecast** — re-forecast and re-solve every hour, commit one hour.
* **perfect foresight** — the ceiling.

Running rolling in *both* a forecast and a perfect-foresight variant is what
separates two effects that are usually conflated. Rolling costs something on its
own — a controller that re-plans hourly is myopic about the horizon boundary in
a way that a single day-ahead plan is not — and it gains something, because each
solve sees one more hour of observed data. Comparing rolling-forecast against
day-ahead-forecast alone cannot tell you which of the two dominates. Comparing
the perfect-foresight variants of each isolates the structural cost with
forecast error held at zero.

**A confound to name rather than hide.** The day-ahead arm re-plans once a day
and is pinned to end each day at the state of charge it started from, because
each day is a separate solve with a terminal constraint. The rolling arm's
terminal constraint moves with its window, so it is never pinned at midnight.
The difference between the two is therefore *both* information timing and daily
energy-neutrality, and the second matters more than it sounds: a battery forced
back to 50% every midnight cannot carry cheap overnight energy into the morning
peak. That is a real property of operating day-ahead, not an artefact, but it is
not what "rolling vs day-ahead" makes most people think of.

Every arm is settled for the state of charge it ends on, so none of them can
profit by finishing depleted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bess_dispatch.data.loaders import load_site_frame, split_frame
from bess_dispatch.data.schema import SiteConfig
from bess_dispatch.evaluation.benchmark import Arm, run_benchmark, summarise_benchmark
from bess_dispatch.forecasting.features import FeatureSpec, periodic_issue_times
from bess_dispatch.forecasting.models import build_forecaster
from bess_dispatch.forecasting.selection import BEST_BY_TARGET
from bess_dispatch.optimization.rolling import run_rolling_horizon

TARGETS = ("load_mw", "pv_mw", "price_eur_mwh")

# Issue times used to train the forecasters the rolling controller serves.
#
# Measured, not assumed: models trained only on 00:00 issues learn horizon_step
# as a proxy for hour-of-day, because at that issue time the two coincide.
# Serving them at 13:00 breaks the association silently. Training every 3 hours
# instead cut the rolling controller's 14-day cost from 20,803.62 to 20,783.25
# EUR. The cost is training time, which is paid once.
ROLLING_TRAIN_STEP_HOURS = 3


def fit_rolling_forecasters(
    train_frame: pd.DataFrame,
    horizon: int = 24,
    spec: FeatureSpec | None = None,
    models: dict[str, str] | None = None,
    step_hours: int = ROLLING_TRAIN_STEP_HOURS,
) -> dict[str, object]:
    spec = spec or FeatureSpec()
    models = models or BEST_BY_TARGET
    issue_times = periodic_issue_times(train_frame, horizon, spec, step_hours)
    return {
        target: build_forecaster(models[target], target, spec).fit(
            train_frame, issue_times, horizon
        )
        for target in TARGETS
    }


def run_controller_comparison(
    site: SiteConfig,
    split: str = "test",
    objective: str = "cost_degradation_demand",
    lookahead: int = 24,
    spec: FeatureSpec | None = None,
    frame: pd.DataFrame | None = None,
    solver_name: str = "appsi_highs",
) -> pd.DataFrame:
    """Run all four controllers over the same window and tabulate."""
    spec = spec or FeatureSpec()
    frame = frame if frame is not None else load_site_frame()
    train = split_frame(frame, "train")
    window = split_frame(frame, split)
    start, end = window.index[0], window.index[-1] + pd.Timedelta(hours=1)

    day_ahead_arms = (
        Arm("no battery", "no_battery"),
        Arm("forecast", "optimized"),
        Arm("perfect foresight", "optimized", perfect=True),
    )
    day_ahead = summarise_benchmark(
        run_benchmark(
            site,
            split=split,
            objective=objective,
            horizon=lookahead,
            arms=day_ahead_arms,
            spec=spec,
            frame=frame,
            solver_name=solver_name,
        )
    )

    rolling_models = fit_rolling_forecasters(train, lookahead, spec)
    rows = [
        {
            "controller": "no battery",
            "total_cost_eur": day_ahead.loc["no battery", "total_cost_eur"],
            "peak_import_mw": day_ahead.loc["no battery", "peak_import_mw"],
            "throughput_mwh": 0.0,
            "solves": int(day_ahead.loc["no battery", "days"]),
        },
        {
            "controller": "day-ahead forecast",
            "total_cost_eur": day_ahead.loc["forecast", "total_cost_eur"],
            "peak_import_mw": day_ahead.loc["forecast", "peak_import_mw"],
            "throughput_mwh": day_ahead.loc["forecast", "throughput_mwh"],
            "solves": int(day_ahead.loc["forecast", "days"]),
        },
    ]

    for label, perfect in (("rolling forecast", False), ("rolling perfect foresight", True)):
        result = run_rolling_horizon(
            frame,
            site,
            rolling_models,
            start,
            end,
            lookahead=lookahead,
            commit_periods=1,
            objective=objective,
            solver_name=solver_name,
            perfect_foresight=perfect,
        )
        rows.append(
            {
                "controller": label,
                "total_cost_eur": result.total_cost_eur,
                "peak_import_mw": result.peak_import_mw,
                "throughput_mwh": result.throughput_mwh,
                "solves": result.n_solves,
                "hours": len(result.timestamps),
                "soc_settlement_eur": result.soc_settlement_eur,
                "terminal_soc_mwh": result.terminal_soc_mwh,
            }
        )

    rows.append(
        {
            "controller": "day-ahead perfect foresight",
            "total_cost_eur": day_ahead.loc["perfect foresight", "total_cost_eur"],
            "peak_import_mw": day_ahead.loc["perfect foresight", "peak_import_mw"],
            "throughput_mwh": day_ahead.loc["perfect foresight", "throughput_mwh"],
            "solves": int(day_ahead.loc["perfect foresight", "days"]),
        }
    )

    table = pd.DataFrame(rows).set_index("controller")
    baseline = table.loc["no battery", "total_cost_eur"]
    table["saving_eur"] = baseline - table["total_cost_eur"]
    table["saving_pct"] = 100 * table["saving_eur"] / baseline
    # The ceiling is whichever arm actually did best, which is the rolling
    # perfect-foresight one -- NOT the day-ahead perfect-foresight arm. Using
    # the latter as a denominator produced values above 100% and implied the
    # rolling arms were doing something impossible, when in fact the day-ahead
    # framing is simply a worse way to operate the same battery.
    ceiling = table["saving_eur"].max()
    table["value_captured_pct"] = 100 * table["saving_eur"] / ceiling
    return table


def main() -> int:
    from bess_dispatch.config import load_config

    config = load_config()
    table = run_controller_comparison(config.site(), objective=config.objective)
    out_dir = Path(__file__).resolve().parents[3] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "controller_comparison.csv")
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
