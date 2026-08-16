"""Forecasters for load, PV and price.

Baselines first, per the brief. A gradient-boosted model that cannot beat
"yesterday, same hour" has not earned its place, and on at least one of these
three series that is a live possibility rather than a rhetorical one — which is
why the persistence baseline is a first-class model here and not a footnote.

Every forecaster takes the same two arguments at predict time — the frame and
an issue time — and gets its features from `features.build_features`, so none
of them can see past the leakage barrier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bess_dispatch.forecasting.features import FeatureSpec, build_features, build_training_set

# Physical bounds used to clip predictions. Not cosmetic: a negative PV
# forecast or a negative load forecast is not merely inaccurate, it is
# unphysical, and handing one to the optimizer produces a schedule that cannot
# be executed. Price is deliberately absent -- negative prices are real.
NON_NEGATIVE_TARGETS = ("load_mw", "pv_mw")


@dataclass
class Forecaster(ABC):
    """Common interface. `name` is what appears in results tables."""

    target: str
    spec: FeatureSpec = field(default_factory=FeatureSpec)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(
        self, frame: pd.DataFrame, issue_times, horizon: int
    ) -> Forecaster: ...

    @abstractmethod
    def _raw_predict(
        self, frame: pd.DataFrame, issue_time: pd.Timestamp, horizon: int
    ) -> np.ndarray: ...

    def predict(
        self, frame: pd.DataFrame, issue_time: pd.Timestamp, horizon: int
    ) -> pd.Series:
        """Forecast `horizon` periods from `issue_time`, clipped to physical bounds."""
        values = np.asarray(self._raw_predict(frame, issue_time, horizon), dtype=float)
        if self.target in NON_NEGATIVE_TARGETS:
            values = np.clip(values, 0.0, None)
        index = pd.DatetimeIndex(
            [pd.Timestamp(issue_time) + pd.Timedelta(hours=k) for k in range(horizon)]
        )
        return pd.Series(values, index=index, name=f"{self.target}_forecast")

    def _features(self, frame, issue_time, horizon) -> pd.DataFrame:
        return build_features(frame, issue_time, horizon, self.target, self.spec)


@dataclass
class PersistenceForecaster(Forecaster):
    """Yesterday, same hour. The benchmark everything else has to beat.

    For a series with strong daily seasonality this is a genuinely hard
    baseline, not a straw man: it carries the full shape of the day at zero
    parameters. It is `lag_24h` used directly as the prediction.
    """

    @property
    def name(self) -> str:
        return "persistence"

    def fit(self, frame, issue_times, horizon) -> PersistenceForecaster:
        return self  # nothing to learn

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        features = self._features(frame, issue_time, horizon)
        prediction = features["lag_24h"].to_numpy(dtype=float)
        # Fall back to the last observation where yesterday is missing, rather
        # than emitting NaN into a downstream optimizer.
        fallback = float(features["last_observed"].iloc[0])
        return np.where(np.isnan(prediction), fallback, prediction)


@dataclass
class WeeklyPersistenceForecaster(Forecaster):
    """Last week, same hour and weekday. Better than daily persistence for load."""

    @property
    def name(self) -> str:
        return "weekly persistence"

    def fit(self, frame, issue_times, horizon) -> WeeklyPersistenceForecaster:
        return self

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        features = self._features(frame, issue_time, horizon)
        prediction = features["lag_168h"].to_numpy(dtype=float)
        fallback = features["lag_24h"].to_numpy(dtype=float)
        prediction = np.where(np.isnan(prediction), fallback, prediction)
        return np.where(
            np.isnan(prediction), float(features["last_observed"].iloc[0]), prediction
        )


@dataclass
class _SklearnForecaster(Forecaster):
    """Shared fit/predict plumbing for the supervised models."""

    model: object = None
    _columns: list[str] = field(default_factory=list, repr=False)

    def fit(self, frame, issue_times, horizon) -> _SklearnForecaster:
        X, y = build_training_set(frame, issue_times, horizon, self.target, self.spec)
        if X.empty:
            raise ValueError(
                f"no usable training rows for {self.target}; every candidate row had "
                "a missing feature or label"
            )
        self._columns = list(X.columns)
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        if not self._columns:
            raise RuntimeError(f"{self.name} has not been fitted")
        features = self._features(frame, issue_time, horizon)[self._columns]
        if features.isna().to_numpy().any():
            # Imputing here would hide a genuine history gap and quietly degrade
            # the forecast. Better to say so.
            missing = features.columns[features.isna().any()].tolist()
            raise ValueError(
                f"cannot forecast {self.target} at {issue_time}: feature(s) {missing} "
                "are unavailable, which means the history before this issue time has "
                "a gap. Choose an issue time with complete history."
            )
        return self.model.predict(features.to_numpy())


@dataclass
class RidgeForecaster(_SklearnForecaster):
    """Linear regression with L2 regularisation, on standardised features."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))]
        )

    @property
    def name(self) -> str:
        return "ridge"


@dataclass
class GradientBoostingForecaster(_SklearnForecaster):
    """scikit-learn's histogram gradient boosting — LightGBM-class, no extra dependency."""

    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int | None = 6
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    @property
    def name(self) -> str:
        return "gradient boosting"


@dataclass
class XGBoostForecaster(_SklearnForecaster):
    """XGBoost, behind the optional `boost` extra.

    Reported on only if it measurably beats gradient boosting on the validation
    window. The brief says "XGBoost if justified"; whether it was is recorded in
    docs/forecasting.md rather than assumed here.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 0

    def __post_init__(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "XGBoostForecaster needs the optional extra: "
                'pip install -e ".[boost]"'
            ) from exc
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            tree_method="hist",
        )

    @property
    def name(self) -> str:
        return "xgboost"


@dataclass
class PublishedForecast(Forecaster):
    """Replays a forecast someone else published, from a column in the frame.

    Used for the transmission operator's own day-ahead load forecast. That
    column is a genuine external benchmark: a professional forecast of the same
    quantity, produced with more information than this site has, and published
    before the day it covers.

    It reads the column at the *target* timestamps, which for any other series
    would be leakage. It is legitimate here only because the value was
    published before the issue time, and it is confined to this one class so
    that the exception is visible rather than scattered.
    """

    column: str = "tso_load_forecast_mw"

    @property
    def name(self) -> str:
        return "published (TSO)"

    def fit(self, frame, issue_times, horizon) -> PublishedForecast:
        if self.column not in frame.columns:
            raise KeyError(f"column {self.column!r} is not in the frame")
        return self

    def _raw_predict(self, frame, issue_time, horizon) -> np.ndarray:
        index = pd.DatetimeIndex(
            [pd.Timestamp(issue_time) + pd.Timedelta(hours=k) for k in range(horizon)]
        )
        return frame[self.column].reindex(index).to_numpy(dtype=float)


MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "persistence": PersistenceForecaster,
    "weekly_persistence": WeeklyPersistenceForecaster,
    "ridge": RidgeForecaster,
    "gradient_boosting": GradientBoostingForecaster,
    "xgboost": XGBoostForecaster,
    "published": PublishedForecast,
}


def build_forecaster(kind: str, target: str, spec: FeatureSpec | None = None, **kwargs):
    """Construct a forecaster by name, for config-driven experiments."""
    if kind not in MODEL_REGISTRY:
        raise KeyError(f"unknown forecaster {kind!r}; expected one of {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[kind](target=target, spec=spec or FeatureSpec(), **kwargs)


def default_forecasters(target: str, spec: FeatureSpec | None = None) -> list[Forecaster]:
    """The comparison set: two baselines, then two learned models."""
    spec = spec or FeatureSpec()
    models: list[Forecaster] = [
        PersistenceForecaster(target=target, spec=spec),
        WeeklyPersistenceForecaster(target=target, spec=spec),
        RidgeForecaster(target=target, spec=spec),
        GradientBoostingForecaster(target=target, spec=spec),
    ]
    if target == "load_mw":
        models.append(PublishedForecast(target=target, spec=spec))
    return models
