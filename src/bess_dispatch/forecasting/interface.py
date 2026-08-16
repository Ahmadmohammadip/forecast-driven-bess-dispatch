"""The boundary between forecasting and optimization.

The brief calls this a critical software-engineering requirement, and it earns
that description for a reason that only shows up later: **the optimizer accepts
a `ForecastResult` and nothing else.**

The consequence is that perfect foresight stops being a special case.
`ForecastResult.from_actuals` wraps observed data in the same object a model
produces, and it travels through the identical builder and the identical
solver. So when the two are compared, the only difference between them is the
numbers in the arrays — not a second code path that might differ in a way
nobody noticed.

That is what makes "the value of forecasting" a measurement rather than an
assertion. It also makes the ablations in `evaluation/` almost free: swapping
one series for its actual value is a field substitution, not a new pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from bess_dispatch.data.schema import TimeSeriesData


@dataclass(frozen=True)
class ForecastMetadata:
    """Provenance for a forecast. Carried so a result can always be traced back.

    `is_perfect_foresight` is not decoration: several reported numbers are only
    meaningful when it is known which arm produced them, and a bare array of
    floats cannot say.
    """

    issue_time: pd.Timestamp | None = None
    models: dict[str, str] = field(default_factory=dict)
    training_window: tuple[str, str] | None = None
    is_perfect_foresight: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """Short name for tables and plot legends."""
        if self.is_perfect_foresight:
            return "perfect foresight"
        if self.models:
            unique = sorted(set(self.models.values()))
            return " + ".join(unique) if len(unique) <= 2 else "forecast"
        return "forecast"


@dataclass(frozen=True)
class ForecastResult:
    """Forecast load, PV and price over one horizon.

    All three arrays share `timestamps` and are in the units the optimizer
    expects: MW, MW and EUR/MWh.
    """

    timestamps: np.ndarray
    load_forecast_mw: np.ndarray
    pv_forecast_mw: np.ndarray
    price_forecast_eur_mwh: np.ndarray
    metadata: ForecastMetadata = field(default_factory=ForecastMetadata)

    def __post_init__(self) -> None:
        stamps = np.asarray(self.timestamps)
        if stamps.ndim != 1 or stamps.size == 0:
            raise ValueError(f"timestamps must be a non-empty 1-D array, got {stamps.shape}")
        object.__setattr__(self, "timestamps", stamps)

        for name in ("load_forecast_mw", "pv_forecast_mw", "price_forecast_eur_mwh"):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.size != stamps.size:
                raise ValueError(
                    f"{name} has {array.size} values but there are {stamps.size} timestamps"
                )
            if not np.isfinite(array).all():
                bad = int(np.argmax(~np.isfinite(array)))
                raise ValueError(
                    f"{name} contains a non-finite value at index {bad}: {array[bad]}. "
                    "A forecast with a hole in it cannot be optimized against -- fix "
                    "the forecaster rather than letting the solver meet a NaN."
                )
            object.__setattr__(self, name, array)

        # Physical bounds. Price is deliberately unconstrained: negative prices
        # are real and are exactly when charging earns.
        for name in ("load_forecast_mw", "pv_forecast_mw"):
            array = getattr(self, name)
            if (array < 0).any():
                bad = int(np.argmax(array < 0))
                raise ValueError(
                    f"{name} must be >= 0, got {array[bad]} at index {bad}. "
                    "A negative load or PV forecast is unphysical, and a schedule "
                    "built on one cannot be executed."
                )

    def __len__(self) -> int:
        return int(self.timestamps.size)

    @property
    def n_periods(self) -> int:
        return len(self)

    @property
    def net_load_forecast_mw(self) -> np.ndarray:
        return self.load_forecast_mw - self.pv_forecast_mw

    @classmethod
    def from_actuals(
        cls, data: TimeSeriesData, notes: str = "actuals used as a forecast"
    ) -> ForecastResult:
        """Wrap observed data as a forecast — the perfect-foresight arm.

        This is the whole trick. The optimizer cannot distinguish this from a
        model's output, so the perfect-foresight benchmark exercises exactly the
        same code as the forecast-driven run.
        """
        return cls(
            timestamps=data.timestamps,
            load_forecast_mw=data.load_mw,
            pv_forecast_mw=data.pv_mw,
            price_forecast_eur_mwh=data.price_eur_mwh,
            metadata=ForecastMetadata(is_perfect_foresight=True, notes=notes),
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        metadata: ForecastMetadata | None = None,
        columns: tuple[str, str, str] = ("load_mw", "pv_mw", "price_eur_mwh"),
    ) -> ForecastResult:
        """Build from a frame whose index is the horizon."""
        load, pv, price = columns
        index = frame.index
        if getattr(index, "tz", None) is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        return cls(
            timestamps=index.to_numpy(),
            load_forecast_mw=frame[load].to_numpy(dtype=float),
            pv_forecast_mw=frame[pv].to_numpy(dtype=float),
            price_forecast_eur_mwh=frame[price].to_numpy(dtype=float),
            metadata=metadata or ForecastMetadata(),
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "load_mw": self.load_forecast_mw,
                "pv_mw": self.pv_forecast_mw,
                "price_eur_mwh": self.price_forecast_eur_mwh,
            },
            index=pd.DatetimeIndex(self.timestamps, name="timestamp"),
        )

    def with_actual(self, data: TimeSeriesData, *series: str) -> ForecastResult:
        """Replace named series with their observed values.

        The ablation primitive. `forecast.with_actual(truth, "price")` answers
        "how much of the loss came from price error alone?" without building a
        second pipeline, because the optimizer sees the same type either way.
        """
        if len(data) != len(self):
            raise ValueError(
                f"actuals cover {len(data)} periods but the forecast covers {len(self)}"
            )
        known = {
            "load": "load_forecast_mw",
            "pv": "pv_forecast_mw",
            "price": "price_forecast_eur_mwh",
        }
        source = {"load": data.load_mw, "pv": data.pv_mw, "price": data.price_eur_mwh}
        unknown = set(series) - set(known)
        if unknown:
            raise KeyError(f"unknown series {sorted(unknown)}; expected any of {sorted(known)}")

        updates = {known[name]: source[name] for name in series}
        replaced = ", ".join(sorted(series))
        metadata = replace(
            self.metadata,
            notes=(self.metadata.notes + f" | actual {replaced} substituted").strip(" |"),
            extra={**self.metadata.extra, "actual_series": sorted(series)},
        )
        return replace(self, metadata=metadata, **updates)

    def slice(self, start: int, stop: int) -> ForecastResult:
        return replace(
            self,
            timestamps=self.timestamps[start:stop],
            load_forecast_mw=self.load_forecast_mw[start:stop],
            pv_forecast_mw=self.pv_forecast_mw[start:stop],
            price_forecast_eur_mwh=self.price_forecast_eur_mwh[start:stop],
        )


def forecast_horizon(
    forecasters: dict[str, Any],
    frame: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int = 24,
    training_window: tuple[str, str] | None = None,
) -> ForecastResult:
    """Run one fitted forecaster per series and assemble a `ForecastResult`.

    `forecasters` maps target column name to a fitted forecaster, e.g.
    `{"load_mw": ..., "pv_mw": ..., "price_eur_mwh": ...}`.
    """
    required = {"load_mw", "pv_mw", "price_eur_mwh"}
    missing = required - set(forecasters)
    if missing:
        raise KeyError(f"no forecaster supplied for {sorted(missing)}")

    predictions = {
        target: forecasters[target].predict(frame, issue_time, horizon)
        for target in required
    }
    index = predictions["load_mw"].index
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("UTC").tz_localize(None)

    return ForecastResult(
        timestamps=index.to_numpy(),
        load_forecast_mw=predictions["load_mw"].to_numpy(),
        pv_forecast_mw=predictions["pv_mw"].to_numpy(),
        price_forecast_eur_mwh=predictions["price_eur_mwh"].to_numpy(),
        metadata=ForecastMetadata(
            issue_time=pd.Timestamp(issue_time),
            models={target: forecasters[target].name for target in sorted(required)},
            training_window=training_window,
        ),
    )


def actuals_for(frame: pd.DataFrame, forecast: ForecastResult) -> TimeSeriesData:
    """The observed data covering the same horizon as `forecast`.

    Used to score a forecast-driven schedule against what actually happened.
    Raises if any period is missing, rather than scoring a shortened horizon
    against a full one and reporting the difference as a saving.
    """
    index = pd.DatetimeIndex(forecast.timestamps)
    if index.tz is None:
        index = index.tz_localize("UTC")
    window = frame.reindex(index)
    missing = window[["load_mw", "pv_mw", "price_eur_mwh"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} of {len(window)} periods in this horizon have no "
            "observed value, so the schedule cannot be scored against actuals"
        )
    return TimeSeriesData(
        timestamps=index.tz_convert("UTC").tz_localize(None).to_numpy(),
        load_mw=window["load_mw"].to_numpy(dtype=float),
        pv_mw=window["pv_mw"].to_numpy(dtype=float),
        price_eur_mwh=window["price_eur_mwh"].to_numpy(dtype=float),
    )
