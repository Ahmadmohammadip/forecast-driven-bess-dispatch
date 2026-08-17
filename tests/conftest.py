"""Shared fixtures.

Everything here is built from the seeded synthetic generator, not the committed
dataset. Two reasons: the suite stays fast enough to run on every save, and CI
never needs the 1.5 MB CSV or the network. The handful of tests that genuinely
need real data are marked and skip cleanly when it is absent.

Fixtures return objects rather than module-level constants because several
tests mutate a copy, and a shared mutable default is a bug waiting for a
Friday afternoon.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Pin the backend for the test process only. The library deliberately does not
# do this -- see visualization/eda.py -- but a test run has no display and no
# reason to want one.
matplotlib.use("Agg")

from bess_dispatch.data.loaders import DEFAULT_DATASET, frame_to_timeseries  # noqa: E402
from bess_dispatch.data.schema import (  # noqa: E402
    Battery,
    GridConnection,
    SiteConfig,
    TariffPolicy,
)
from bess_dispatch.data.synthetic import make_synthetic_site  # noqa: E402
from bess_dispatch.forecasting.interface import ForecastResult  # noqa: E402


@pytest.fixture
def battery() -> Battery:
    """The reference battery from configs/base.yaml."""
    return Battery(
        name="Batt1",
        energy_capacity_mwh=1.0,
        p_charge_max_mw=0.5,
        p_discharge_max_mw=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        soc_min_frac=0.10,
        soc_max_frac=0.90,
        initial_soc_frac=0.50,
        degradation_cost_eur_mwh=2.0,
    )


@pytest.fixture
def tariff_policy() -> TariffPolicy:
    return TariffPolicy(
        import_markup_eur_mwh=60.0, export_ratio=0.70, demand_charge_eur_mw=5.0
    )


@pytest.fixture
def site(battery: Battery, tariff_policy: TariffPolicy) -> SiteConfig:
    return SiteConfig(
        battery=battery,
        grid=GridConnection(import_limit_mw=2.0, export_limit_mw=2.0),
        tariff_policy=tariff_policy,
    )


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """Ninety days of seeded synthetic site data."""
    return make_synthetic_site(periods=24 * 90)


@pytest.fixture
def synthetic_year() -> pd.DataFrame:
    """A full synthetic year, for tests that need seasonal range."""
    return make_synthetic_site(periods=8760)


@pytest.fixture
def one_day(synthetic_frame: pd.DataFrame):
    """A single 24-hour `TimeSeriesData`, mid-window."""
    window = synthetic_frame.iloc[24 * 40 : 24 * 41]
    return frame_to_timeseries(window)


@pytest.fixture
def perfect_forecast(one_day) -> ForecastResult:
    return ForecastResult.from_actuals(one_day)


@pytest.fixture
def spiky_day(one_day):
    """A day engineered so arbitrage is unambiguously worth doing.

    Prices are cheap for the first half and dear for the second, with a gap far
    wider than round-trip losses. Tests that need the battery to *do something*
    use this, so they assert on physics rather than on whether a marginal
    arbitrage happened to clear.
    """
    from bess_dispatch.data.schema import TimeSeriesData

    prices = np.concatenate([np.full(12, 10.0), np.full(12, 200.0)])
    return TimeSeriesData(
        timestamps=one_day.timestamps,
        load_mw=one_day.load_mw,
        pv_mw=one_day.pv_mw,
        price_eur_mwh=prices,
    )


@pytest.fixture
def real_frame():
    """The committed dataset, or skip.

    Only for tests that assert something about the real data specifically.
    """
    if not DEFAULT_DATASET.exists():
        pytest.skip(f"{DEFAULT_DATASET} not present")
    from bess_dispatch.data.loaders import load_site_frame

    return load_site_frame()
