"""Typed, validated data structures for the dispatch model.

Design intent, shared with the sibling repos: a `Battery`, `Tariff`,
`SiteConfig` or `TimeSeriesData` fails loudly at construction — an efficiency
outside (0, 1], an initial state of charge outside the usable band, a price
series whose length does not match the timestamps — rather than surfacing three
layers down as an opaque solver infeasibility.

One validation here is not a hygiene check but a modelling result. See
`Tariff._check_not_arbitrageable`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

_TOLERANCE = 1e-9


def _as_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        bad = int(np.argmax(~np.isfinite(array)))
        raise ValueError(f"{name} contains a non-finite value at index {bad}: {array[bad]}")
    return array


@dataclass(frozen=True)
class Battery:
    """A single battery energy storage system.

    Energy limits are expressed as **fractions of nameplate capacity**, because
    that is how depth-of-discharge policy is actually specified. The absolute
    MWh band is derived (`soc_min_mwh`, `soc_max_mwh`).

    Fidelity is deliberately modest: efficiency losses, power limits, an energy
    band, and a throughput-proportional degradation cost. That degradation term
    is an **economic proxy, not an electrochemical model** — it does not know
    about cycle depth, temperature, calendar ageing, or C-rate.
    """

    name: str
    energy_capacity_mwh: float
    p_charge_max_mw: float
    p_discharge_max_mw: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    initial_soc_frac: float = 0.50
    # EUR per MWh of throughput (charge + discharge), applied as a cost in the
    # objective variants that include degradation.
    degradation_cost_eur_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_capacity_mwh <= 0:
            raise ValueError(
                f"{self.name}: energy_capacity_mwh must be > 0, "
                f"got {self.energy_capacity_mwh}"
            )
        for attr in ("p_charge_max_mw", "p_discharge_max_mw"):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{self.name}: {attr} must be > 0, got {getattr(self, attr)}")
        for attr in ("charge_efficiency", "discharge_efficiency"):
            value = getattr(self, attr)
            if not 0 < value <= 1:
                raise ValueError(f"{self.name}: {attr} must be in (0, 1], got {value}")
        for attr in ("soc_min_frac", "soc_max_frac", "initial_soc_frac"):
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                raise ValueError(f"{self.name}: {attr} must be in [0, 1], got {value}")
        if self.soc_max_frac <= self.soc_min_frac:
            raise ValueError(
                f"{self.name}: soc_max_frac ({self.soc_max_frac}) must be greater than "
                f"soc_min_frac ({self.soc_min_frac})"
            )
        if not self.soc_min_frac <= self.initial_soc_frac <= self.soc_max_frac:
            raise ValueError(
                f"{self.name}: initial_soc_frac ({self.initial_soc_frac}) must lie within "
                f"the usable band [{self.soc_min_frac}, {self.soc_max_frac}]"
            )
        if self.degradation_cost_eur_mwh < 0:
            raise ValueError(
                f"{self.name}: degradation_cost_eur_mwh must be >= 0, "
                f"got {self.degradation_cost_eur_mwh}"
            )

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_capacity_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_capacity_mwh

    @property
    def initial_soc_mwh(self) -> float:
        return self.initial_soc_frac * self.energy_capacity_mwh

    @property
    def usable_energy_mwh(self) -> float:
        """Width of the usable band. Not the nameplate capacity."""
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency


@dataclass(frozen=True)
class Tariff:
    """Per-period import and export prices, plus an optional demand charge.

    Both price arrays are in EUR/MWh and are indexed by period, aligned with the
    horizon they were built for.
    """

    import_price_eur_mwh: np.ndarray
    export_price_eur_mwh: np.ndarray
    # EUR per MW of peak grid import over the horizon. Charged once, on the
    # single highest import period -- see docs/formulation.md.
    demand_charge_eur_mw: float = 0.0

    def __post_init__(self) -> None:
        buy = _as_array(self.import_price_eur_mwh, "import_price_eur_mwh")
        sell = _as_array(self.export_price_eur_mwh, "export_price_eur_mwh")
        object.__setattr__(self, "import_price_eur_mwh", buy)
        object.__setattr__(self, "export_price_eur_mwh", sell)

        if buy.size != sell.size:
            raise ValueError(
                f"import_price_eur_mwh has {buy.size} periods but "
                f"export_price_eur_mwh has {sell.size}"
            )
        if self.demand_charge_eur_mw < 0:
            raise ValueError(
                f"demand_charge_eur_mw must be >= 0, got {self.demand_charge_eur_mw}"
            )
        self._check_not_arbitrageable(buy, sell)

    @staticmethod
    def _check_not_arbitrageable(buy: np.ndarray, sell: np.ndarray) -> None:
        """Reject a tariff that pays more to export than it costs to import.

        This is the one validation here that came out of a measured result
        rather than defensive habit. In a probe run, a tariff paying 1.3x the
        import price for exports produced an optimum that imported and exported
        **simultaneously in all 24 hours, with no battery in the model at
        all** — the meter being gamed, not the battery being used.

        The tempting fix is a binary variable forbidding simultaneous import and
        export. That is the wrong fix: it would make an unphysical tariff solve
        slowly instead of failing, hiding a data error behind a MILP.

        The subtle case is negative prices, and this dataset has 484 of them. A
        tariff of the form `import = wholesale + markup`, `export = ratio x
        wholesale` is safe for positive prices whenever `ratio <= 1`, but
        inverts when the wholesale price goes negative: at -90 EUR/MWh with
        `ratio = 0.7`, importing pays you 90 while exporting costs you 63, so
        the round trip nets +27 EUR/MWh forever. A markup of at least
        `max(p * (ratio - 1))` is what closes it.
        """
        violation = sell - buy
        worst = int(np.argmax(violation))
        if violation[worst] > _TOLERANCE:
            raise ValueError(
                f"export price exceeds import price in period {worst} "
                f"(export {sell[worst]:.2f} > import {buy[worst]:.2f} EUR/MWh), "
                f"by up to {violation[worst]:.2f} EUR/MWh across the horizon.\n"
                "Such a tariff can be arbitraged by importing and exporting at the "
                "same time, with no battery involved, so any dispatch result would "
                "be meaningless.\n"
                "If your prices go negative, this is usually the cause: an export "
                "price defined as a fraction of wholesale inverts below zero. "
                "Raising the import markup by at least "
                f"{violation[worst]:.2f} EUR/MWh resolves it."
            )

    @property
    def n_periods(self) -> int:
        return int(self.import_price_eur_mwh.size)


@dataclass(frozen=True)
class TariffPolicy:
    """Rules for turning a wholesale price series into a `Tariff`.

    Separated from `Tariff` so that a scenario can vary the *policy* (a heavier
    demand charge, a stingier export rate) and re-derive prices for any horizon,
    rather than editing arrays.
    """

    # Added to wholesale to give the delivered import price: network charges,
    # levies, taxes, supplier margin. Everything that is not the energy itself.
    import_markup_eur_mwh: float = 60.0
    # Fraction of wholesale paid for exported energy.
    export_ratio: float = 0.70
    demand_charge_eur_mw: float = 0.0

    def __post_init__(self) -> None:
        if self.import_markup_eur_mwh < 0:
            raise ValueError(
                f"import_markup_eur_mwh must be >= 0, got {self.import_markup_eur_mwh}"
            )
        if self.export_ratio < 0:
            raise ValueError(f"export_ratio must be >= 0, got {self.export_ratio}")
        if self.demand_charge_eur_mw < 0:
            raise ValueError(
                f"demand_charge_eur_mw must be >= 0, got {self.demand_charge_eur_mw}"
            )

    def apply(self, wholesale_price_eur_mwh: Sequence[float] | np.ndarray) -> Tariff:
        """Build a `Tariff` for one horizon of wholesale prices."""
        wholesale = _as_array(wholesale_price_eur_mwh, "wholesale_price_eur_mwh")
        return Tariff(
            import_price_eur_mwh=wholesale + self.import_markup_eur_mwh,
            export_price_eur_mwh=wholesale * self.export_ratio,
            demand_charge_eur_mw=self.demand_charge_eur_mw,
        )

    def minimum_safe_markup(
        self, wholesale_price_eur_mwh: Sequence[float] | np.ndarray
    ) -> float:
        """Smallest markup that keeps this policy non-arbitrageable on these prices.

        Useful for diagnosing the negative-price case described in
        `Tariff._check_not_arbitrageable` without waiting for the exception.
        """
        wholesale = _as_array(wholesale_price_eur_mwh, "wholesale_price_eur_mwh")
        bound = float(max(0.0, np.max(wholesale * (self.export_ratio - 1.0))))
        if bound == 0.0:
            return 0.0
        # Return a markup that is actually safe, not one sitting exactly on the
        # boundary. At the exact bound, import and export prices are equal up to
        # floating-point error, and rounding can land either side of it -- so a
        # helper named "minimum_safe_markup" would otherwise return a value that
        # Tariff rejects. The margin is relative, so it scales with the numbers.
        return bound * (1.0 + 1e-9) + 1e-9


@dataclass(frozen=True)
class GridConnection:
    """Connection limits at the point of common coupling."""

    import_limit_mw: float
    export_limit_mw: float

    def __post_init__(self) -> None:
        if self.import_limit_mw <= 0:
            raise ValueError(f"import_limit_mw must be > 0, got {self.import_limit_mw}")
        if self.export_limit_mw < 0:
            raise ValueError(f"export_limit_mw must be >= 0, got {self.export_limit_mw}")


@dataclass(frozen=True)
class SiteConfig:
    """Everything about the site that does not change hour to hour."""

    battery: Battery
    grid: GridConnection
    tariff_policy: TariffPolicy = field(default_factory=TariffPolicy)
    dt_hours: float = 1.0
    # Require the horizon to end at the state of charge it started from. Without
    # it, a finite horizon ends by selling the battery empty, which flatters the
    # reported cost with energy that was never paid for.
    enforce_terminal_soc: bool = True

    def __post_init__(self) -> None:
        if self.dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {self.dt_hours}")


@dataclass(frozen=True)
class TimeSeriesData:
    """Aligned load, PV and price observations on a regular time grid.

    This is *observed* data. Forecasts travel in a `ForecastResult`
    (`bess_dispatch.forecasting.interface`) instead, so that the optimizer can
    never be handed actuals by accident.
    """

    timestamps: np.ndarray  # datetime64[ns, UTC] or comparable
    load_mw: np.ndarray
    pv_mw: np.ndarray
    price_eur_mwh: np.ndarray

    def __post_init__(self) -> None:
        stamps = np.asarray(self.timestamps)
        if stamps.ndim != 1 or stamps.size == 0:
            raise ValueError(f"timestamps must be a non-empty 1-D array, got {stamps.shape}")
        object.__setattr__(self, "timestamps", stamps)

        for name in ("load_mw", "pv_mw", "price_eur_mwh"):
            array = _as_array(getattr(self, name), name)
            if array.size != stamps.size:
                raise ValueError(
                    f"{name} has {array.size} values but there are {stamps.size} timestamps"
                )
            object.__setattr__(self, name, array)

        if (self.load_mw < 0).any():
            bad = int(np.argmax(self.load_mw < 0))
            raise ValueError(f"load_mw must be >= 0, got {self.load_mw[bad]} at index {bad}")
        if (self.pv_mw < 0).any():
            bad = int(np.argmax(self.pv_mw < 0))
            raise ValueError(f"pv_mw must be >= 0, got {self.pv_mw[bad]} at index {bad}")
        # price_eur_mwh is deliberately unconstrained: negative prices are real.

        self._check_regular_grid(stamps)

    @staticmethod
    def _to_epoch_seconds(stamps: np.ndarray) -> np.ndarray:
        """Seconds since the epoch, whatever flavour of timestamp came in.

        A timezone-aware pandas index arrives here as an object array of
        Timestamps; numpy has no timezone-aware dtype, so converting it with
        `.astype("datetime64[s]")` both warns and silently discards the offset.
        Timestamps are UTC by convention throughout this project (see
        data/DATA_DICTIONARY.md), so the offset carries no information -- but
        dropping it quietly is still the wrong way to arrive there.
        """
        if stamps.dtype == object:
            return np.array([t.timestamp() for t in stamps], dtype="int64")
        return stamps.astype("datetime64[s]").astype("int64")

    @classmethod
    def _check_regular_grid(cls, stamps: np.ndarray) -> None:
        if stamps.size < 2:
            return
        deltas = np.diff(cls._to_epoch_seconds(stamps))
        if (deltas <= 0).any():
            bad = int(np.argmax(deltas <= 0))
            raise ValueError(
                f"timestamps must be strictly increasing, but index {bad + 1} "
                f"({stamps[bad + 1]}) does not follow index {bad} ({stamps[bad]})"
            )
        if len(np.unique(deltas)) != 1:
            spacings = sorted({int(d) for d in deltas})
            raise ValueError(
                "timestamps must be evenly spaced -- lag features and the state-of-charge "
                f"recursion both assume it. Found {len(spacings)} distinct spacings "
                f"(seconds): {spacings[:5]}"
            )

    def __len__(self) -> int:
        return int(self.timestamps.size)

    @property
    def n_periods(self) -> int:
        return len(self)

    @property
    def net_load_mw(self) -> np.ndarray:
        """Load minus PV. Negative where the site is exporting before storage."""
        return self.load_mw - self.pv_mw

    def slice(self, start: int, stop: int) -> TimeSeriesData:
        """A contiguous sub-window, preserving validation."""
        return TimeSeriesData(
            timestamps=self.timestamps[start:stop],
            load_mw=self.load_mw[start:stop],
            pv_mw=self.pv_mw[start:stop],
            price_eur_mwh=self.price_eur_mwh[start:stop],
        )
