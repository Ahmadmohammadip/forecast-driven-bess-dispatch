"""The most important tests in this repository.

Every headline number depends on the forecasts having been produced without
seeing the future they predict. If that is false, nothing else here means
anything — and the failure is invisible in accuracy metrics, because a leaking
model simply looks unusually good.

So these tests check three things, in increasing order of what they are worth:

1. that features are unchanged when the future is replaced with noise;
2. that the check is **not vacuous** — a deliberately leaking feature is caught;
3. that the lag/horizon guard refuses a configuration that would leak.

Point 2 is the one that matters. A leakage test that cannot fail is worse than
no leakage test, because it produces confidence rather than information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bess_dispatch.forecasting import features as F
from bess_dispatch.forecasting.features import (
    FeatureSpec,
    assert_no_future_reference,
    build_features,
    build_training_set,
    daily_issue_times,
)

TARGETS = ("load_mw", "pv_mw", "price_eur_mwh")


@pytest.fixture
def issue_time(synthetic_frame) -> pd.Timestamp:
    return daily_issue_times(synthetic_frame, 24, FeatureSpec())[10]


@pytest.mark.parametrize("target", TARGETS)
def test_features_ignore_the_future(synthetic_frame, issue_time, target):
    """Corrupting everything at or after the issue time changes no feature."""
    assert_no_future_reference(synthetic_frame, issue_time, 24, target, FeatureSpec())


@pytest.mark.parametrize("target", TARGETS)
def test_features_ignore_the_future_when_it_is_deleted(
    synthetic_frame, issue_time, target
):
    """Stronger: deleting the future entirely gives identical features.

    Noise could in principle be absorbed by some pathological aggregation.
    Truncation cannot be.
    """
    spec = FeatureSpec()
    full = build_features(synthetic_frame, issue_time, 24, target, spec)
    truncated_source = synthetic_frame.loc[synthetic_frame.index < issue_time]
    truncated = build_features(truncated_source, issue_time, 24, target, spec)
    pd.testing.assert_frame_equal(full, truncated)


def test_leakage_check_catches_a_deliberate_leak(synthetic_frame, issue_time, monkeypatch):
    """The check must be able to fail, or it proves nothing.

    A feature reading the target at its own timestamp is the most plausible
    real-world leak: it looks like an ordinary column and is catastrophic.
    """
    original = F.build_features

    def leaking(frame, issue, horizon, target, spec=None):
        out = original(frame, issue, horizon, target, spec)
        out["leaked_same_hour"] = frame[target].reindex(out.index).to_numpy()
        return out

    monkeypatch.setattr(F, "build_features", leaking)

    with pytest.raises(AssertionError, match="leaked_same_hour"):
        F.assert_no_future_reference(
            synthetic_frame, issue_time, 24, "price_eur_mwh", FeatureSpec()
        )


def test_horizon_longer_than_lag_is_refused(synthetic_frame, issue_time):
    """A 24-hour lag over a 48-step horizon would read inside the window."""
    spec = FeatureSpec(lags_hours=(24, 48, 168))
    with pytest.raises(ValueError, match="shorter than the 48-step horizon"):
        build_features(synthetic_frame, issue_time, 48, "price_eur_mwh", spec)


def test_a_long_horizon_is_allowed_with_long_enough_lags(synthetic_frame, issue_time):
    """The guard blocks unsafe combinations, not long horizons as such."""
    spec = FeatureSpec(lags_hours=(168,), rolling_windows_hours=(168,))
    out = build_features(synthetic_frame, issue_time, 48, "price_eur_mwh", spec)
    assert len(out) == 48
    assert_no_future_reference(synthetic_frame, issue_time, 48, "price_eur_mwh", spec)


def test_no_history_before_issue_time_is_an_error(synthetic_frame):
    first = synthetic_frame.index[0]
    with pytest.raises(ValueError, match="no history before issue_time"):
        build_features(synthetic_frame, first, 24, "price_eur_mwh")


def test_training_labels_come_from_the_future_but_features_do_not(synthetic_frame):
    """Labels are supervision, not leakage — but only labels may look ahead.

    Corrupting the future must move `y` and leave `X` untouched.
    """
    spec = FeatureSpec()
    issues = daily_issue_times(synthetic_frame, 24, spec)[:8]
    X, y = build_training_set(synthetic_frame, issues, 24, "price_eur_mwh", spec)

    corrupted = synthetic_frame.copy()
    after = corrupted.index >= issues[0]
    corrupted.loc[after, "price_eur_mwh"] += 500.0
    X2, y2 = build_training_set(corrupted, issues, 24, "price_eur_mwh", spec)

    # Features built from history before the *first* issue time are shared; the
    # later windows draw on corrupted history, so compare only the first window.
    first = X.index < issues[1]
    pd.testing.assert_frame_equal(X.loc[first], X2.loc[first])
    assert np.allclose(y.loc[first].to_numpy() + 500.0, y2.loc[first].to_numpy())


def test_assert_no_future_reference_refuses_a_vacuous_check(synthetic_frame):
    """Asking to corrupt a future that does not exist must raise, not pass."""
    last = synthetic_frame.index[-1] + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="vacuously"):
        assert_no_future_reference(synthetic_frame, last, 24, "price_eur_mwh")


@pytest.mark.parametrize("target", TARGETS)
def test_window_statistics_are_constant_across_the_horizon(
    synthetic_frame, issue_time, target
):
    """They describe the moment of issue, so they cannot vary by target hour.

    If one ever did vary, it would mean it had been recomputed per target
    timestamp — which is exactly how a window statistic starts leaking.
    """
    out = build_features(synthetic_frame, issue_time, 24, target, FeatureSpec())
    for column in out.columns:
        if column.startswith("last_"):
            assert out[column].nunique() == 1, f"{column} varies across the horizon"
