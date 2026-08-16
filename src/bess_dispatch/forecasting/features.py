"""Feature construction, built so that leakage is structurally impossible.

The whole project rests on one claim: that the forecasts were produced without
seeing the future they predict. Two design choices defend it, and they matter
more than any individual feature.

**One function, both paths.** `build_features` is used for training *and* for
inference. There is no separate serving path that could drift from the training
path — the classic way leakage enters a codebase that is careful about it in
principle.

**Slice before you look.** The first statement of `build_features` discards
every row at or after `issue_time`. Everything after that operates on `past`,
so a feature *cannot* reference the future even if a later edit tries to: the
data is simply not in scope. This is deliberately stronger than computing
features and then masking them.

Lags are taken relative to the **target** timestamp, not the issue time, so
`lag_24h` means "the same hour yesterday" for every step of the horizon. That
is only safe while `lag > horizon_step`, and `_check_lag_is_safe` enforces it
rather than trusting the caller to pick a compatible horizon.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The local zone the site sits in. Calendar features use local time: solar
# position and human routine both follow the local clock, and a model given UTC
# hour has to learn the DST offset as noise.
LOCAL_TZ = "Europe/Berlin"


@dataclass(frozen=True)
class FeatureSpec:
    """Which features to build.

    Defaults are the ones that survived validation-set comparison; see
    `docs/forecasting.md` for what each is worth.
    """

    # Hours before the *target* timestamp. Every entry must exceed the horizon
    # length, or the lag would reach into the forecast window.
    lags_hours: tuple[int, ...] = (24, 48, 168)
    # Windows, in hours, ending at the issue time. These summarise "what the
    # series was doing when the forecast was made" and are constant across the
    # horizon.
    rolling_windows_hours: tuple[int, ...] = (24, 168)
    include_calendar: bool = True
    # Exogenous columns whose *lagged* values may help, e.g. PV lags for price.
    exogenous: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.lags_hours:
            raise ValueError("lags_hours must not be empty")
        if any(lag <= 0 for lag in self.lags_hours):
            raise ValueError(f"lags_hours must all be positive, got {self.lags_hours}")
        if any(window <= 0 for window in self.rolling_windows_hours):
            raise ValueError(
                f"rolling_windows_hours must all be positive, "
                f"got {self.rolling_windows_hours}"
            )

    @property
    def max_lookback_hours(self) -> int:
        """How much history must exist before the first issue time."""
        return max((*self.lags_hours, *self.rolling_windows_hours))

    def check_horizon(self, horizon: int) -> None:
        """Refuse a horizon that would let a lag reach into the forecast window."""
        offenders = [lag for lag in self.lags_hours if lag < horizon]
        if offenders:
            raise ValueError(
                f"lag(s) {offenders} are shorter than the {horizon}-step horizon. "
                f"A lag of L applied to horizon step k reads the value at "
                f"target - L, which lies inside the forecast window whenever "
                f"k >= L. Use lags of at least {horizon} hours, or shorten the "
                "horizon."
            )


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features of the target timestamps.

    These are not leakage: the calendar is known arbitrarily far ahead. Hour and
    month are encoded as sine/cosine pairs so that hour 23 sits next to hour 0
    rather than 23 units away, which a tree can learn but a linear model cannot.
    """
    local = index.tz_convert(LOCAL_TZ)
    hour = local.hour.to_numpy()
    month = local.month.to_numpy()
    dayofweek = local.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "dayofweek": dayofweek.astype(float),
            "is_weekend": (dayofweek >= 5).astype(float),
        },
        index=index,
    )


def build_features(
    frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int,
    target: str,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Features for the `horizon` periods starting at `issue_time`.

    Uses **only** rows strictly before `issue_time`. The returned frame is
    indexed by the target timestamps it describes.

    Rows whose lag values are unavailable (too little history) come back with
    NaN, which the models drop at fit time and which `predict` refuses at
    inference time rather than quietly imputing.
    """
    spec = spec or FeatureSpec()
    spec.check_horizon(horizon)

    issue_time = pd.Timestamp(issue_time)
    if issue_time.tz is None:
        issue_time = issue_time.tz_localize("UTC")

    # Everything downstream sees only `past`. This is the leakage barrier.
    past = frame.loc[frame.index < issue_time]
    if past.empty:
        raise ValueError(
            f"no history before issue_time {issue_time}; the frame starts at "
            f"{frame.index[0]}"
        )

    step = pd.Timedelta(hours=1)
    targets = pd.DatetimeIndex(
        [issue_time + step * k for k in range(horizon)], name=frame.index.name
    )

    columns: dict[str, np.ndarray] = {}
    series_names = (target, *spec.exogenous)

    for name in series_names:
        if name not in past.columns:
            raise KeyError(f"column {name!r} is not in the frame")
        history = past[name]
        prefix = "" if name == target else f"{name}_"

        for lag in spec.lags_hours:
            wanted = targets - step * lag
            # reindex rather than .loc: a missing timestamp becomes NaN instead
            # of raising, and NaN is the honest answer for "not observed yet".
            columns[f"{prefix}lag_{lag}h"] = history.reindex(wanted).to_numpy()

        # Window statistics ending at the issue time. Constant across the
        # horizon by construction -- they describe the moment of issue, not the
        # target -- which is exactly right for a day-ahead forecast.
        for window in spec.rolling_windows_hours:
            recent = history.loc[history.index >= issue_time - step * window]
            for statistic, value in (
                ("mean", recent.mean()),
                ("std", recent.std()),
                ("min", recent.min()),
                ("max", recent.max()),
            ):
                columns[f"{prefix}last_{window}h_{statistic}"] = np.full(
                    horizon, float(value) if pd.notna(value) else np.nan
                )

        columns[f"{prefix}last_observed"] = np.full(horizon, float(history.iloc[-1]))

    features = pd.DataFrame(columns, index=targets)
    # How far ahead each row is. A day-ahead model should be allowed to be less
    # confident about hour 23 than hour 1.
    features["horizon_step"] = np.arange(horizon, dtype=float)

    if spec.include_calendar:
        features = features.join(_calendar_features(targets))

    return features


def build_training_set(
    frame: pd.DataFrame,
    issue_times: Sequence[pd.Timestamp],
    horizon: int,
    target: str,
    spec: FeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack `build_features` over many issue times into one design matrix.

    The labels are read from `frame` at the target timestamps. That is not
    leakage — they are the supervision signal — but it is the only place the
    future is touched, and it is confined to this function.

    Rows with any missing feature or a missing label are dropped, so the caller
    never trains on silently imputed values.
    """
    spec = spec or FeatureSpec()
    blocks, labels = [], []
    for issue_time in issue_times:
        features = build_features(frame, issue_time, horizon, target, spec)
        y = frame[target].reindex(features.index)
        blocks.append(features)
        labels.append(y)

    if not blocks:
        raise ValueError("issue_times is empty, so there is nothing to train on")

    X = pd.concat(blocks)
    y = pd.concat(labels)
    keep = X.notna().all(axis=1) & y.notna()
    return X.loc[keep], y.loc[keep]


def daily_issue_times(
    frame: pd.DataFrame,
    horizon: int = 24,
    spec: FeatureSpec | None = None,
    hour_utc: int = 0,
) -> pd.DatetimeIndex:
    """Issue times for a day-ahead schedule: one per day, at `hour_utc`.

    Skips the opening stretch of the frame where there is not yet enough
    history for the longest lag, and any issue time whose full horizon does not
    fit inside the frame.
    """
    spec = spec or FeatureSpec()
    step = pd.Timedelta(hours=1)
    earliest = frame.index[0] + step * spec.max_lookback_hours
    latest = frame.index[-1] - step * (horizon - 1)

    candidates = pd.date_range(
        frame.index[0].normalize(), frame.index[-1].normalize(), freq="D", tz="UTC"
    ) + pd.Timedelta(hours=hour_utc)
    return pd.DatetimeIndex([t for t in candidates if earliest <= t <= latest])


def feature_names(
    target: str, spec: FeatureSpec | None = None, horizon: int = 24
) -> list[str]:
    """Column order produced by `build_features`, without building it."""
    spec = spec or FeatureSpec()
    names: list[str] = []
    for name in (target, *spec.exogenous):
        prefix = "" if name == target else f"{name}_"
        names += [f"{prefix}lag_{lag}h" for lag in spec.lags_hours]
        for window in spec.rolling_windows_hours:
            names += [
                f"{prefix}last_{window}h_{statistic}"
                for statistic in ("mean", "std", "min", "max")
            ]
        names.append(f"{prefix}last_observed")
    names.append("horizon_step")
    if spec.include_calendar:
        names += [
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "dayofweek",
            "is_weekend",
        ]
    return names


def assert_no_future_reference(
    frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int,
    target: str,
    spec: FeatureSpec | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Prove empirically that features ignore everything at or after `issue_time`.

    Builds features once on the real frame, then again on a frame whose future
    has been replaced with noise, and requires the two to be identical.

    This is the check the test suite runs, exposed here so it can also be run
    ad hoc against a new feature. A feature that fails it is leaking, however
    reasonable it looked.
    """
    rng = rng or np.random.default_rng(0)
    issue_time = pd.Timestamp(issue_time)
    if issue_time.tz is None:
        issue_time = issue_time.tz_localize("UTC")

    honest = build_features(frame, issue_time, horizon, target, spec)

    corrupted = frame.copy()
    future = corrupted.index >= issue_time
    if not future.any():
        raise ValueError(
            f"nothing at or after {issue_time} to corrupt -- this check would pass "
            "vacuously"
        )
    for column in corrupted.columns:
        # Floats only. is_numeric_dtype() is True for bool as well, and pandas
        # refuses to write float noise into a bool column.
        if pd.api.types.is_float_dtype(corrupted[column]):
            corrupted.loc[future, column] = rng.normal(1e6, 1e5, int(future.sum()))
        elif pd.api.types.is_bool_dtype(corrupted[column]):
            corrupted.loc[future, column] = ~corrupted.loc[future, column]

    tampered = build_features(corrupted, issue_time, horizon, target, spec)

    if not honest.equals(tampered):
        differing = [
            column
            for column in honest.columns
            if not honest[column].equals(tampered[column])
        ]
        raise AssertionError(
            f"features changed when the future was corrupted, so they leak. "
            f"Offending column(s): {differing}"
        )


def describe_features(
    frame: pd.DataFrame,
    target: str,
    spec: FeatureSpec | None = None,
    horizon: int = 24,
) -> pd.DataFrame:
    """One row per feature: what it is and where it gets its information."""
    spec = spec or FeatureSpec()
    rows: list[dict[str, str]] = []
    for name in feature_names(target, spec, horizon):
        if name.startswith(("hour_", "month_", "dayofweek", "is_weekend")):
            source = "calendar (known arbitrarily far ahead)"
        elif name == "horizon_step":
            source = "position within the forecast window"
        elif "lag_" in name:
            source = "observed value at target minus lag, always before issue time"
        elif "last_observed" in name:
            source = "final observation before issue time"
        else:
            source = "window statistic ending at issue time"
        rows.append({"feature": name, "information source": source})
    return pd.DataFrame(rows).set_index("feature")


def iter_windows(
    frame: pd.DataFrame, issue_times: Iterable[pd.Timestamp], horizon: int
):
    """Yield `(issue_time, target_index)` pairs, for backtest loops."""
    step = pd.Timedelta(hours=1)
    for issue_time in issue_times:
        issue_time = pd.Timestamp(issue_time)
        yield issue_time, pd.DatetimeIndex(
            [issue_time + step * k for k in range(horizon)]
        )
