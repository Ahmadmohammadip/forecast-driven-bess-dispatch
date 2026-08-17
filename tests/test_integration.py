"""End to end: data in, priced schedule out.

Also the tests that guard the *interface* contract — that the optimizer cannot
tell a forecast from actuals, which is what makes the perfect-foresight
comparison trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_dispatch.config import DEFAULT_CONFIG, load_config
from bess_dispatch.data.loaders import (
    complete_days,
    daily_price_spread,
    frame_to_timeseries,
    load_synthetic_timeseries,
    split_frame,
    train_test_report,
)
from bess_dispatch.evaluation.benchmark import (
    DEFAULT_ARMS,
    ablation_table,
    fit_forecasters,
    run_benchmark,
    summarise_benchmark,
)
from bess_dispatch.evaluation.kpis import KPI_ORDER, describe_kpis, kpi_table
from bess_dispatch.forecasting.features import FeatureSpec, daily_issue_times
from bess_dispatch.forecasting.interface import (
    ForecastResult,
    actuals_for,
    forecast_horizon,
)
from bess_dispatch.forecasting.models import RidgeForecaster
from bess_dispatch.optimization.builder import build_dispatch_model
from bess_dispatch.optimization.solve import solve_dispatch

TOL = 1e-6


# --- the interface contract ---------------------------------------------


def test_optimizer_cannot_distinguish_forecast_from_actuals(one_day, site):
    """Perfect foresight must not be a separate code path.

    Handing the builder a `ForecastResult` built from actuals and one built
    from identical arrays must give byte-identical schedules.
    """
    from_actuals = ForecastResult.from_actuals(one_day)
    hand_built = ForecastResult(
        timestamps=one_day.timestamps,
        load_forecast_mw=one_day.load_mw,
        pv_forecast_mw=one_day.pv_mw,
        price_forecast_eur_mwh=one_day.price_eur_mwh,
    )
    a = solve_dispatch(build_dispatch_model(from_actuals, site))
    b = solve_dispatch(build_dispatch_model(hand_built, site))
    assert np.allclose(a.charge_mw, b.charge_mw)
    assert a.total_cost_eur == pytest.approx(b.total_cost_eur)


def test_with_actual_substitutes_only_the_named_series(one_day):
    noisy = ForecastResult(
        timestamps=one_day.timestamps,
        load_forecast_mw=one_day.load_mw * 1.2,
        pv_forecast_mw=one_day.pv_mw * 0.8,
        price_forecast_eur_mwh=one_day.price_eur_mwh + 15.0,
    )
    swapped = noisy.with_actual(one_day, "price")
    assert np.allclose(swapped.price_forecast_eur_mwh, one_day.price_eur_mwh)
    assert np.allclose(swapped.load_forecast_mw, noisy.load_forecast_mw)
    assert swapped.metadata.extra["actual_series"] == ["price"]


def test_with_actual_rejects_a_length_mismatch(one_day):
    forecast = ForecastResult.from_actuals(one_day)
    with pytest.raises(ValueError, match="periods but the forecast covers"):
        forecast.slice(0, 12).with_actual(one_day, "price")


def test_actuals_for_refuses_an_incomplete_horizon(synthetic_frame, one_day):
    forecast = ForecastResult.from_actuals(one_day)
    holed = synthetic_frame.copy()
    # Punch the hole inside the forecast's own horizon. Holing an arbitrary
    # stretch of the frame proves nothing -- the first draft of this test did
    # exactly that and passed for the wrong reason.
    horizon = pd.DatetimeIndex(forecast.timestamps).tz_localize("UTC")
    holed.loc[horizon[5:8], "price_eur_mwh"] = np.nan
    with pytest.raises(ValueError, match="no observed value"):
        actuals_for(holed, forecast)


def test_forecast_horizon_requires_all_three_series(synthetic_frame):
    with pytest.raises(KeyError, match="no forecaster supplied"):
        forecast_horizon({}, synthetic_frame, synthetic_frame.index[300], 24)


# --- end to end ----------------------------------------------------------


def test_full_pipeline_on_synthetic_data(synthetic_frame, site):
    """Fit, forecast, solve, price — the whole chain on a small window."""
    spec = FeatureSpec()
    train = synthetic_frame.iloc[: 24 * 60]
    issues = daily_issue_times(train, 24, spec)
    fitted = {
        target: RidgeForecaster(target=target, spec=spec).fit(train, issues, 24)
        for target in ("load_mw", "pv_mw", "price_eur_mwh")
    }

    issue = synthetic_frame.index[24 * 70]
    forecast = forecast_horizon(fitted, synthetic_frame, issue, 24)
    truth = actuals_for(synthetic_frame, forecast)

    result = solve_dispatch(build_dispatch_model(forecast, site))
    assert result.termination == "optimal"
    assert len(result.charge_mw) == 24
    assert sum(result.cost_breakdown().values()) == pytest.approx(
        result.total_cost_eur, abs=TOL
    )

    perfect = solve_dispatch(
        build_dispatch_model(ForecastResult.from_actuals(truth), site)
    )
    # Perfect information cannot be worse once both are priced on truth.
    from bess_dispatch.optimization.solve import evaluate_schedule

    assert (
        evaluate_schedule(perfect, truth, site)["realised_cost_eur"]
        <= evaluate_schedule(result, truth, site)["realised_cost_eur"] + TOL
    )


def test_benchmark_arms_cover_identical_days(synthetic_frame, site):
    """Otherwise the arm totals are not comparable."""
    spec = FeatureSpec()
    train = synthetic_frame.iloc[: 24 * 50]
    fitted = fit_forecasters(train, 24, spec, models=dict.fromkeys(
        ("load_mw", "pv_mw", "price_eur_mwh"), "ridge"
    ))
    results = _benchmark_on(synthetic_frame, site, fitted, spec)
    counts = results.groupby("arm")["issue_time"].nunique()
    assert counts.nunique() == 1, "arms cover different day sets"


def _benchmark_on(frame, site, fitted, spec, max_days=6):
    """Run the benchmark over a synthetic frame by faking the split windows."""
    import bess_dispatch.data.loaders as loaders

    original = loaders.SPLITS.copy()
    start = frame.index[24 * 55]
    stop = frame.index[24 * 65]
    try:
        loaders.SPLITS["test"] = (str(start.date()), str(stop.date()))
        return run_benchmark(
            site, split="test", frame=frame, fitted=fitted, spec=spec, max_days=max_days
        )
    finally:
        loaders.SPLITS.update(original)


def test_ablation_and_kpis_are_consistent(synthetic_frame, site):
    spec = FeatureSpec()
    train = synthetic_frame.iloc[: 24 * 50]
    fitted = fit_forecasters(train, 24, spec, models=dict.fromkeys(
        ("load_mw", "pv_mw", "price_eur_mwh"), "ridge"
    ))
    results = _benchmark_on(synthetic_frame, site, fitted, spec)
    summary = summarise_benchmark(results)

    assert set(summary.index) == {arm.name for arm in DEFAULT_ARMS}
    assert summary.loc["no battery", "saving_eur"] == pytest.approx(0.0, abs=TOL)
    # Perfect foresight is the ceiling among the day-ahead arms.
    assert summary["saving_eur"].max() == pytest.approx(
        summary.loc["perfect foresight", "saving_eur"], abs=1e-6
    )

    ablation = ablation_table(summary)
    assert set(ablation.index) == {"price", "load", "pv"}

    kpis = kpi_table(results, summary, energy_capacity_mwh=1.0, pv_generated_mwh=10.0)
    assert list(kpis.columns) == list(KPI_ORDER)
    assert len(describe_kpis()) == len(KPI_ORDER)


def test_ablation_requires_both_reference_arms():
    frame = pd.DataFrame({"total_cost_eur": [1.0]}, index=pd.Index(["forecast"], name="arm"))
    with pytest.raises(KeyError, match="perfect foresight"):
        ablation_table(frame)


# --- loaders -------------------------------------------------------------


def test_synthetic_loader_round_trips():
    data = load_synthetic_timeseries(48)
    assert len(data) == 48
    assert (data.load_mw >= 0).all()
    assert (data.pv_mw >= 0).all()


def test_complete_days_keeps_only_whole_days(synthetic_frame):
    holed = synthetic_frame.copy()
    holed.loc[holed.index[30], "load_mw"] = np.nan
    kept = complete_days(holed)
    assert len(kept) % 24 == 0
    assert len(kept) == len(synthetic_frame) - 24


def test_split_frame_rejects_an_unknown_split(synthetic_frame):
    with pytest.raises(KeyError, match="unknown split"):
        split_frame(synthetic_frame, "holdout")


def test_daily_price_spread_is_non_negative(synthetic_frame):
    spread = daily_price_spread(synthetic_frame)
    assert (spread >= 0).all()
    assert len(spread) == len(synthetic_frame) // 24


def test_frame_to_timeseries_drops_the_timezone(synthetic_frame):
    data = frame_to_timeseries(synthetic_frame.iloc[:24])
    assert np.issubdtype(data.timestamps.dtype, np.datetime64)


# --- config --------------------------------------------------------------


def test_base_config_loads_and_builds_a_site():
    config = load_config()
    assert DEFAULT_CONFIG.exists()
    site = config.site()
    assert site.battery.energy_capacity_mwh == 1.0
    assert site.enforce_terminal_soc is True


def test_config_overrides_are_deep(tmp_path):
    config = load_config()
    changed = config.with_overrides(battery={"energy_capacity_mwh": 3.0})
    assert changed.site().battery.energy_capacity_mwh == 3.0
    # Untouched keys survive the merge.
    assert changed.site().battery.p_charge_max_mw == config.site().battery.p_charge_max_mw


def test_missing_config_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_config("no_such_config.yaml")


@pytest.mark.parametrize("split", ["train", "validation", "test", "shift"])
def test_real_dataset_splits_are_populated(real_frame, split):
    """Guards the claims in data/DATA_DICTIONARY.md."""
    report = train_test_report()
    assert report.loc[split, "rows"] > 0
    assert report.loc[split, "complete"] > 0


def test_real_dataset_test_window_is_clean(real_frame):
    report = train_test_report()
    assert report.loc["test", "imputed"] == 0
    assert report.loc["test", "rows"] == report.loc["test", "complete"]
