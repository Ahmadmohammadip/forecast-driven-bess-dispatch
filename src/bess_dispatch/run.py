"""One command that reproduces the baseline experiment.

    python -m bess_dispatch.run                        # the reference case
    python -m bess_dispatch.run --config configs/scenarios/high_prices.yaml
    python -m bess_dispatch.run --stage all            # everything, ~10 minutes

Stages are separable because they cost very different amounts. The day-ahead
benchmark is seconds; the rolling controller is 2,880 solves; the sweeps are
twenty benchmark runs. Defaulting to `benchmark` means the headline numbers
come back quickly and the expensive parts are opt-in.

Everything writes to `results/`, and every table there is regenerated from
scratch — nothing is appended to, so a stale row cannot survive a rerun.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bess_dispatch.config import load_config
from bess_dispatch.data.loaders import load_site_frame, split_frame
from bess_dispatch.evaluation.benchmark import (
    ablation_table,
    fit_forecasters,
    run_benchmark,
    summarise_benchmark,
)
from bess_dispatch.evaluation.controllers import run_controller_comparison
from bess_dispatch.evaluation.kpis import kpi_table
from bess_dispatch.evaluation.scenarios import run_scenarios, run_sensitivities
from bess_dispatch.forecasting.features import FeatureSpec

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
STAGES = ("benchmark", "rolling", "scenarios", "forecasts", "all")


def _write(table: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    table.to_csv(path)
    print(f"  wrote {path.relative_to(RESULTS_DIR.parent)}")
    return path


def stage_benchmark(config, frame, fitted) -> pd.DataFrame:
    print("\n=== day-ahead benchmark ===")
    results = run_benchmark(
        config.site(),
        split=config.split,
        objective=config.objective,
        horizon=config.horizon,
        frame=frame,
        fitted=fitted,
    )
    summary = summarise_benchmark(results)
    _write(results.set_index(["issue_time", "arm"]), "benchmark_daily.csv")
    _write(summary, "benchmark_summary.csv")

    print(
        summary[
            ["total_cost_eur", "saving_eur", "saving_pct_wholesale", "value_captured_pct"]
        ]
        .round(3)
        .to_string()
    )

    ablation = ablation_table(summary)
    _write(ablation, "ablation.csv")
    print("\n--- where the shortfall comes from ---")
    print(ablation.round(2).to_string())

    pv_generated = float(
        split_frame(frame, config.split)["pv_mw"].sum() * config.dt_hours
    )
    kpis = kpi_table(
        results,
        summary,
        energy_capacity_mwh=config.site().battery.energy_capacity_mwh,
        pv_generated_mwh=pv_generated,
    )
    _write(kpis, "kpis.csv")
    return summary


def stage_rolling(config, frame) -> pd.DataFrame:
    print("\n=== rolling horizon (2,880 solves — a few minutes) ===")
    table = run_controller_comparison(
        config.site(),
        split=config.split,
        objective=config.objective,
        lookahead=config.horizon,
        frame=frame,
    )
    _write(table, "controller_comparison.csv")
    print(
        table[["total_cost_eur", "saving_eur", "value_captured_pct"]].round(3).to_string()
    )
    return table


def stage_scenarios(config, frame) -> None:
    print("\n=== scenarios ===")
    scenarios = run_scenarios(
        config.site(), objective=config.objective, split=config.split, frame=frame
    )
    _write(scenarios, "scenarios.csv")
    print(scenarios["saving_eur"].unstack("arm").round(2).to_string())

    print("\n=== sensitivities ===")
    sensitivities = run_sensitivities(
        config.site(), objective=config.objective, split=config.split, frame=frame
    )
    _write(sensitivities, "sensitivities.csv")
    print(sensitivities["saving_eur"].unstack("arm").round(2).to_string())


def stage_forecasts(config) -> None:
    from bess_dispatch.forecasting.selection import run_selection

    print("\n=== forecast model selection ===")
    selection = run_selection(horizon=config.horizon)
    _write(selection, "forecast_selection.csv")
    for target in selection.index.get_level_values("target").unique():
        block = selection.loc[target]
        print(f"\n{target}: lowest MAE = {block['MAE'].idxmin()}")
        print(block[["MAE", "RMSE", "R2"]].round(4).to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--stage", default="benchmark", choices=STAGES)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    np.random.seed(config.seed)

    print(f"experiment: {config.name}")
    print(f"  objective {config.objective}, split {config.split}, horizon {config.horizon}")
    battery = config.site().battery
    print(
        f"  battery {battery.energy_capacity_mwh} MWh / {battery.p_charge_max_mw} MW, "
        f"round-trip {battery.round_trip_efficiency:.3f}"
    )

    started = time.perf_counter()
    frame = load_site_frame()

    if args.stage in ("forecasts", "all"):
        stage_forecasts(config)

    fitted = fit_forecasters(
        split_frame(frame, "train"), config.horizon, FeatureSpec(), config.models or None
    )

    if args.stage in ("benchmark", "all"):
        stage_benchmark(config, frame, fitted)
    if args.stage in ("rolling", "all"):
        stage_rolling(config, frame)
    if args.stage in ("scenarios", "all"):
        stage_scenarios(config, frame)

    print(f"\ndone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
