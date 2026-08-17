"""Construction-time validation.

The design contract is that a bad configuration fails where it is written, not
three layers down as an opaque solver infeasibility. These tests pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.data.schema import (
    Battery,
    GridConnection,
    Tariff,
    TariffPolicy,
    TimeSeriesData,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"energy_capacity_mwh": 0}, "energy_capacity_mwh must be > 0"),
        ({"p_charge_max_mw": 0}, "p_charge_max_mw must be > 0"),
        ({"charge_efficiency": 1.5}, "charge_efficiency must be in"),
        ({"discharge_efficiency": 0}, "discharge_efficiency must be in"),
        ({"soc_min_frac": 0.9, "soc_max_frac": 0.1}, "must be greater than"),
        ({"initial_soc_frac": 0.99}, "must lie within the usable band"),
        ({"degradation_cost_eur_mwh": -1}, "must be >= 0"),
    ],
)
def test_battery_rejects_bad_input(kwargs, message):
    defaults = {
        "name": "B",
        "energy_capacity_mwh": 1.0,
        "p_charge_max_mw": 0.5,
        "p_discharge_max_mw": 0.5,
    }
    with pytest.raises(ValueError, match=message):
        Battery(**{**defaults, **kwargs})


def test_battery_derived_quantities(battery):
    assert battery.soc_min_mwh == pytest.approx(0.1)
    assert battery.soc_max_mwh == pytest.approx(0.9)
    # Usable energy is the band width, not the nameplate.
    assert battery.usable_energy_mwh == pytest.approx(0.8)
    assert battery.round_trip_efficiency == pytest.approx(0.9025)


def test_grid_rejects_bad_limits():
    with pytest.raises(ValueError, match="import_limit_mw must be > 0"):
        GridConnection(0.0, 1.0)
    with pytest.raises(ValueError, match="export_limit_mw must be >= 0"):
        GridConnection(1.0, -0.1)


# --- the arbitrage guard -------------------------------------------------
# This is the validation that came out of a measured result rather than
# defensive habit: a tariff paying more to export than to import can be
# arbitraged with no battery in the model at all.


def test_tariff_rejects_export_above_import():
    with pytest.raises(ValueError, match="export price exceeds import price"):
        Tariff(
            import_price_eur_mwh=np.array([50.0, 50.0]),
            export_price_eur_mwh=np.array([50.0, 65.0]),
        )


def test_tariff_error_names_the_consequence_and_the_fix():
    with pytest.raises(ValueError) as caught:
        Tariff(
            import_price_eur_mwh=np.array([10.0]),
            export_price_eur_mwh=np.array([25.0]),
        )
    message = str(caught.value)
    assert "importing and exporting at the same time" in message
    assert "15.00" in message  # the markup increase that would fix it


def test_negative_prices_invert_a_percentage_export_tariff():
    """The subtle case, and the reason the guard exists at all.

    `export = 0.7 x wholesale` is safe while wholesale is positive and inverts
    below zero: at -90 EUR/MWh importing pays 90 and exporting costs 63.
    """
    wholesale = np.array([50.0, -90.0])
    policy = TariffPolicy(import_markup_eur_mwh=0.0, export_ratio=0.7)
    with pytest.raises(ValueError, match="export price exceeds import price"):
        policy.apply(wholesale)


def test_minimum_safe_markup_is_actually_safe():
    """A helper that returns a value its own validator rejects is a bug.

    At the exact analytic bound, floating point lands on either side; the
    helper returns a hair above it.
    """
    wholesale = np.array([50.0, -90.0, 0.0, 120.0])
    for ratio in (0.0, 0.5, 0.7, 0.9, 1.0):
        policy = TariffPolicy(import_markup_eur_mwh=0.0, export_ratio=ratio)
        needed = policy.minimum_safe_markup(wholesale)
        safe = TariffPolicy(import_markup_eur_mwh=needed, export_ratio=ratio)
        safe.apply(wholesale)  # must not raise


def test_markup_just_below_the_bound_is_rejected():
    wholesale = np.array([50.0, -90.0])
    policy = TariffPolicy(import_markup_eur_mwh=0.0, export_ratio=0.7)
    needed = policy.minimum_safe_markup(wholesale)
    with pytest.raises(ValueError):
        TariffPolicy(
            import_markup_eur_mwh=needed - 0.5, export_ratio=0.7
        ).apply(wholesale)


def test_tariff_length_mismatch():
    with pytest.raises(ValueError, match="has 2 periods but"):
        Tariff(np.array([10.0, 10.0]), np.array([5.0]))


# --- time series ---------------------------------------------------------


def test_timeseries_allows_negative_prices_but_not_negative_load(one_day):
    """Negative prices are real market behaviour; negative load is not physical."""
    TimeSeriesData(
        timestamps=one_day.timestamps,
        load_mw=one_day.load_mw,
        pv_mw=one_day.pv_mw,
        price_eur_mwh=np.full(len(one_day), -75.0),
    )
    with pytest.raises(ValueError, match="load_mw must be >= 0"):
        TimeSeriesData(
            timestamps=one_day.timestamps,
            load_mw=one_day.load_mw - 10.0,
            pv_mw=one_day.pv_mw,
            price_eur_mwh=one_day.price_eur_mwh,
        )


def test_timeseries_rejects_an_irregular_grid(one_day):
    stamps = one_day.timestamps.copy()
    stamps[5] = stamps[5] + np.timedelta64(17, "m")
    with pytest.raises(ValueError, match="evenly spaced"):
        TimeSeriesData(
            timestamps=stamps,
            load_mw=one_day.load_mw,
            pv_mw=one_day.pv_mw,
            price_eur_mwh=one_day.price_eur_mwh,
        )


def test_timeseries_rejects_out_of_order_timestamps(one_day):
    stamps = one_day.timestamps.copy()
    stamps[3], stamps[4] = stamps[4], stamps[3]
    with pytest.raises(ValueError, match="strictly increasing|evenly spaced"):
        TimeSeriesData(
            timestamps=stamps,
            load_mw=one_day.load_mw,
            pv_mw=one_day.pv_mw,
            price_eur_mwh=one_day.price_eur_mwh,
        )


def test_timeseries_rejects_nan(one_day):
    values = one_day.load_mw.copy()
    values[7] = np.nan
    with pytest.raises(ValueError, match="non-finite value at index 7"):
        TimeSeriesData(
            timestamps=one_day.timestamps,
            load_mw=values,
            pv_mw=one_day.pv_mw,
            price_eur_mwh=one_day.price_eur_mwh,
        )


def test_net_load_and_slice(one_day):
    assert np.allclose(one_day.net_load_mw, one_day.load_mw - one_day.pv_mw)
    window = one_day.slice(4, 10)
    assert len(window) == 6
    assert np.allclose(window.load_mw, one_day.load_mw[4:10])
