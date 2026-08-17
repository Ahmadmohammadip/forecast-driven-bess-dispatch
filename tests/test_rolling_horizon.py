"""The receding-horizon controller.

Two properties matter more than the rest: that only the committed head of each
plan is executed, and that the controller cannot see actuals it should not have.
The second is inherited from `build_features` rather than enforced here, and
these tests confirm the inheritance actually holds end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_dispatch.forecasting.features import FeatureSpec, periodic_issue_times
from bess_dispatch.forecasting.models import RidgeForecaster
from bess_dispatch.optimization.rolling import run_rolling_horizon

HORIZON = 24
TOL = 1e-6


# Module-scoped, and deliberately so: fitting three ridge models per test
# dominated the runtime of this file (60s down to a few seconds). The
# forecasters are only ever read, so sharing them is safe. `synthetic_frame`
# stays function-scoped in conftest because tests here do mutate copies of it.


@pytest.fixture(scope="module")
def shared_frame():
    from bess_dispatch.data.synthetic import make_synthetic_site

    return make_synthetic_site(periods=24 * 60)


@pytest.fixture(scope="module")
def fitted(shared_frame):
    """Cheap ridge models across issue hours — this is not an accuracy test."""
    spec = FeatureSpec()
    issues = periodic_issue_times(shared_frame, HORIZON, spec, step_hours=6)
    return {
        target: RidgeForecaster(target=target, spec=spec).fit(
            shared_frame, issues, HORIZON
        )
        for target in ("load_mw", "pv_mw", "price_eur_mwh")
    }


@pytest.fixture
def synthetic_frame(shared_frame):
    """A private copy, so a test that corrupts it cannot affect its neighbours."""
    return shared_frame.copy()


@pytest.fixture
def window(shared_frame):
    start = shared_frame.index[24 * 25]
    return start, start + pd.Timedelta(hours=24)


def test_covers_every_committed_hour(synthetic_frame, site, fitted, window):
    start, end = window
    result = run_rolling_horizon(synthetic_frame, site, fitted, start, end)
    assert len(result.timestamps) == 24
    assert result.timestamps[0] == start
    assert result.timestamps[-1] == end - pd.Timedelta(hours=1)


def test_one_solve_per_committed_period(synthetic_frame, site, fitted, window):
    start, end = window
    result = run_rolling_horizon(synthetic_frame, site, fitted, start, end)
    assert result.n_solves == 24


def test_committing_more_periods_means_fewer_solves(synthetic_frame, site, fitted, window):
    start, end = window
    one = run_rolling_horizon(synthetic_frame, site, fitted, start, end, commit_periods=1)
    six = run_rolling_horizon(synthetic_frame, site, fitted, start, end, commit_periods=6)
    assert six.n_solves == one.n_solves // 6
    assert len(six.timestamps) == len(one.timestamps)


def test_soc_stays_in_band_and_follows_the_recursion(
    synthetic_frame, site, fitted, window, battery
):
    start, end = window
    result = run_rolling_horizon(synthetic_frame, site, fitted, start, end)
    assert result.soc_mwh.min() >= battery.soc_min_mwh - TOL
    assert result.soc_mwh.max() <= battery.soc_max_mwh + TOL

    previous = np.concatenate([[battery.initial_soc_mwh], result.soc_mwh[:-1]])
    expected = (
        previous
        + battery.charge_efficiency * result.charge_mw
        - result.discharge_mw / battery.discharge_efficiency
    )
    assert np.abs(expected - result.soc_mwh).max() < TOL


def test_future_actuals_do_not_reach_the_controller(synthetic_frame, site, fitted, window):
    """Corrupt everything after the run and the executed schedule is unchanged.

    The lookahead may legitimately read beyond `end`, so the corruption starts
    one full lookahead after it — past anything the controller may see.
    """
    start, end = window
    honest = run_rolling_horizon(synthetic_frame, site, fitted, start, end)

    corrupted = synthetic_frame.copy()
    beyond = corrupted.index >= end + pd.Timedelta(hours=HORIZON)
    assert beyond.any(), "need data past the lookahead for this to mean anything"
    for column in ("load_mw", "pv_mw", "price_eur_mwh"):
        corrupted.loc[beyond, column] = 1e5

    tampered = run_rolling_horizon(corrupted, site, fitted, start, end)
    assert np.allclose(honest.charge_mw, tampered.charge_mw)
    assert np.allclose(honest.discharge_mw, tampered.discharge_mw)
    assert honest.total_cost_eur == pytest.approx(tampered.total_cost_eur)


def test_perfect_foresight_beats_forecast(synthetic_frame, site, fitted, window):
    start, end = window
    forecast = run_rolling_horizon(synthetic_frame, site, fitted, start, end)
    perfect = run_rolling_horizon(
        synthetic_frame, site, fitted, start, end, perfect_foresight=True
    )
    assert perfect.total_cost_eur <= forecast.total_cost_eur + TOL


def test_settlement_is_included_and_signed_correctly(
    synthetic_frame, site, fitted, window, battery
):
    start, end = window
    result = run_rolling_horizon(synthetic_frame, site, fitted, start, end)
    if result.terminal_soc_mwh < battery.initial_soc_mwh - TOL:
        assert result.soc_settlement_eur > 0
    elif result.terminal_soc_mwh > battery.initial_soc_mwh + TOL:
        assert result.soc_settlement_eur < 0
    assert result.terminal_soc_mwh == pytest.approx(result.soc_mwh[-1])


def test_cost_breakdown_matches_the_objective_terms(synthetic_frame, site, fitted, window):
    start, end = window
    result = run_rolling_horizon(
        synthetic_frame, site, fitted, start, end, objective="cost_degradation"
    )
    assert tuple(result.cost_breakdown()) == ("energy", "degradation")


def test_bad_commit_periods_are_refused(synthetic_frame, site, fitted, window):
    start, end = window
    with pytest.raises(ValueError, match="commit_periods must be >= 1"):
        run_rolling_horizon(synthetic_frame, site, fitted, start, end, commit_periods=0)
    with pytest.raises(ValueError, match="cannot exceed lookahead"):
        run_rolling_horizon(
            synthetic_frame, site, fitted, start, end, lookahead=4, commit_periods=8
        )


def test_to_frame_round_trips(synthetic_frame, site, fitted, window):
    start, end = window
    result = run_rolling_horizon(synthetic_frame, site, fitted, start, end)
    frame = result.to_frame()
    assert len(frame) == len(result.timestamps)
    assert np.allclose(frame["soc_mwh"].to_numpy(), result.soc_mwh)
