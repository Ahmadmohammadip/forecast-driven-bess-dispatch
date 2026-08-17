"""Physics of the solved schedule: balance, state of charge, and the objective.

The rule that shaped these tests, learned the hard way across the four sibling
repos: **assert on physics and totals, not on individual dispatch decisions.**
An LP with flat price hours has many optima, and a test pinning "charges in hour
3" fails the day the solver picks an equally good hour 4. Where a specific
action matters, `spiky_day` makes it uniquely optimal by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from bess_dispatch.data.schema import TimeSeriesData
from bess_dispatch.forecasting.interface import ForecastResult
from bess_dispatch.optimization.builder import (
    OBJECTIVES,
    build_dispatch_model,
    build_no_battery_model,
)
from bess_dispatch.optimization.rules import rule_based_schedule
from bess_dispatch.optimization.solve import (
    evaluate_schedule,
    soc_settlement_eur,
    solve_dispatch,
)

TOL = 1e-6


def solve(forecast, site, objective="cost_degradation_demand"):
    return solve_dispatch(build_dispatch_model(forecast, site, objective=objective))


# --- power balance -------------------------------------------------------


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_power_balance_holds_every_period(perfect_forecast, one_day, site, objective):
    result = solve(perfect_forecast, site, objective)
    residual = (
        one_day.load_mw
        + result.charge_mw
        + result.grid_export_mw
        - (one_day.pv_mw - result.curtailment_mw)
        - result.discharge_mw
        - result.grid_import_mw
    )
    assert np.abs(residual).max() < TOL


def test_curtailment_never_exceeds_generation(perfect_forecast, one_day, site):
    result = solve(perfect_forecast, site)
    assert (result.curtailment_mw <= one_day.pv_mw + TOL).all()
    assert (result.curtailment_mw >= -TOL).all()


def test_grid_limits_respected(perfect_forecast, site):
    result = solve(perfect_forecast, site)
    assert result.grid_import_mw.max() <= site.grid.import_limit_mw + TOL
    assert result.grid_export_mw.max() <= site.grid.export_limit_mw + TOL


# --- state of charge -----------------------------------------------------


def test_soc_recursion_holds(perfect_forecast, site, battery):
    result = solve(perfect_forecast, site)
    previous = np.concatenate([[battery.initial_soc_mwh], result.soc_mwh[:-1]])
    expected = (
        previous
        + battery.charge_efficiency * result.charge_mw * site.dt_hours
        - result.discharge_mw * site.dt_hours / battery.discharge_efficiency
    )
    assert np.abs(expected - result.soc_mwh).max() < TOL


def test_soc_stays_inside_the_band(perfect_forecast, site, battery):
    result = solve(perfect_forecast, site)
    assert result.soc_mwh.min() >= battery.soc_min_mwh - TOL
    assert result.soc_mwh.max() <= battery.soc_max_mwh + TOL


def test_terminal_soc_returns_to_the_start(perfect_forecast, site, battery):
    result = solve(perfect_forecast, site)
    assert result.soc_mwh[-1] == pytest.approx(battery.initial_soc_mwh, abs=TOL)


def test_without_the_terminal_constraint_the_battery_ends_empty(
    spiky_day, site, battery
):
    """Why `enforce_terminal_soc` exists.

    Released, a finite horizon ends by selling the battery down, and the
    reported cost is flattered by energy that was never paid for.
    """
    from dataclasses import replace

    free = replace(site, enforce_terminal_soc=False)
    forecast = ForecastResult.from_actuals(spiky_day)

    pinned = solve(forecast, site)
    released = solve(forecast, free)

    assert released.soc_mwh[-1] < pinned.soc_mwh[-1] - 0.05
    assert released.total_cost_eur < pinned.total_cost_eur


def test_efficiency_losses_run_the_right_way(spiky_day, site):
    """A full cycle returns less energy than it took in."""
    result = solve(ForecastResult.from_actuals(spiky_day), site)
    charged = result.charge_mw.sum() * site.dt_hours
    discharged = result.discharge_mw.sum() * site.dt_hours
    assert charged > 0, "the spiky day should provoke some arbitrage"
    assert discharged < charged


def test_power_limits_respected(perfect_forecast, site, battery):
    result = solve(perfect_forecast, site)
    assert result.charge_mw.max() <= battery.p_charge_max_mw + TOL
    assert result.discharge_mw.max() <= battery.p_discharge_max_mw + TOL


def test_no_simultaneous_charge_and_discharge(spiky_day, site):
    """The claim that justifies leaving binaries out of the model.

    Round-trip losses make it strictly wasteful, so the optimum never does it
    and the constraint forbidding it would never bind.
    """
    result = solve(ForecastResult.from_actuals(spiky_day), site)
    both = (result.charge_mw > 1e-6) & (result.discharge_mw > 1e-6)
    assert not both.any()


def test_no_simultaneous_import_and_export(spiky_day, site):
    result = solve(ForecastResult.from_actuals(spiky_day), site)
    both = (result.grid_import_mw > 1e-6) & (result.grid_export_mw > 1e-6)
    assert not both.any()


# --- objective -----------------------------------------------------------


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_cost_breakdown_sums_to_the_objective(perfect_forecast, site, objective):
    result = solve(perfect_forecast, site, objective)
    assert sum(result.cost_breakdown().values()) == pytest.approx(
        result.total_cost_eur, abs=1e-6
    )


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_breakdown_lists_only_optimised_terms(perfect_forecast, site, objective):
    result = solve(perfect_forecast, site, objective)
    assert tuple(result.cost_breakdown()) == OBJECTIVES[objective]


def test_adding_cost_terms_never_lowers_that_measure(spiky_day, site):
    """Each variant must be optimal for its own objective.

    Evaluating the cost-only schedule under the degradation objective can never
    beat the schedule that actually optimised it.
    """
    forecast = ForecastResult.from_actuals(spiky_day)
    cost_only = solve(forecast, site, "cost")
    with_degradation = solve(forecast, site, "cost_degradation")

    as_scored = cost_only.energy_cost_eur + cost_only.degradation_cost_eur
    assert with_degradation.total_cost_eur <= as_scored + TOL


def test_degradation_cost_reduces_cycling(spiky_day, site):
    from dataclasses import replace

    forecast = ForecastResult.from_actuals(spiky_day)
    cheap = replace(site, battery=replace(site.battery, degradation_cost_eur_mwh=0.0))
    dear = replace(site, battery=replace(site.battery, degradation_cost_eur_mwh=60.0))

    assert (
        solve(forecast, dear, "cost_degradation").throughput_mwh
        <= solve(forecast, cheap, "cost_degradation").throughput_mwh + TOL
    )


def test_battery_never_costs_more_than_no_battery(perfect_forecast, site):
    """With perfect information the battery is an option, never an obligation."""
    with_battery = solve(perfect_forecast, site)
    without = solve_dispatch(
        build_no_battery_model(
            perfect_forecast, site, objective="cost_degradation_demand"
        )
    )
    assert with_battery.total_cost_eur <= without.total_cost_eur + TOL


def test_no_battery_model_holds_the_battery_still(perfect_forecast, site, battery):
    result = solve_dispatch(build_no_battery_model(perfect_forecast, site))
    assert np.allclose(result.charge_mw, 0.0)
    assert np.allclose(result.discharge_mw, 0.0)
    assert np.allclose(result.soc_mwh, battery.initial_soc_mwh)


def test_unknown_objective_is_refused(perfect_forecast, site):
    with pytest.raises(KeyError, match="unknown objective"):
        build_dispatch_model(perfect_forecast, site, objective="cost_of_living")


# --- scoring against actuals --------------------------------------------


def test_evaluate_schedule_reprices_against_reality(spiky_day, site):
    """A plan made on wrong prices must be scored on the right ones."""
    wrong = TimeSeriesData(
        timestamps=spiky_day.timestamps,
        load_mw=spiky_day.load_mw,
        pv_mw=spiky_day.pv_mw,
        price_eur_mwh=spiky_day.price_eur_mwh[::-1].copy(),
    )
    planned = solve(ForecastResult.from_actuals(wrong), site)
    scored = evaluate_schedule(planned, spiky_day, site)
    assert scored["realised_cost_eur"] > scored["planned_cost_eur"]


def test_evaluate_schedule_clips_to_the_soc_band(spiky_day, site, battery):
    scored = evaluate_schedule(solve(ForecastResult.from_actuals(spiky_day), site),
                               spiky_day, site)
    assert battery.soc_min_mwh - TOL <= scored["terminal_soc_mwh"]
    assert scored["terminal_soc_mwh"] <= battery.soc_max_mwh + TOL


def test_soc_settlement_charges_a_depleted_run(battery):
    """Ending short must cost; ending long must credit.

    Without this, a controller that finishes empty has been handed free energy
    — the defect that made the rule-based arm appear to beat perfect foresight.
    """
    short = soc_settlement_eur(0.5, 0.1, battery, 90.0)
    long = soc_settlement_eur(0.5, 0.9, battery, 90.0)
    neutral = soc_settlement_eur(0.5, 0.5, battery, 90.0)
    assert short > 0
    assert long < 0
    assert neutral == pytest.approx(0.0)


# --- the rule-based controller ------------------------------------------


def test_rule_based_never_violates_the_battery(one_day, site, battery):
    result = rule_based_schedule(one_day, site)
    assert result.soc_mwh.min() >= battery.soc_min_mwh - TOL
    assert result.soc_mwh.max() <= battery.soc_max_mwh + TOL
    assert result.charge_mw.max() <= battery.p_charge_max_mw + TOL
    assert result.discharge_mw.max() <= battery.p_discharge_max_mw + TOL


def test_rule_based_ignores_price(one_day, site):
    """Greedy self-consumption must be identical under any price series.

    If reversing prices changed its schedule, it would be looking at them.
    """
    reversed_prices = TimeSeriesData(
        timestamps=one_day.timestamps,
        load_mw=one_day.load_mw,
        pv_mw=one_day.pv_mw,
        price_eur_mwh=one_day.price_eur_mwh[::-1].copy(),
    )
    a = rule_based_schedule(one_day, site)
    b = rule_based_schedule(reversed_prices, site)
    assert np.allclose(a.charge_mw, b.charge_mw)
    assert np.allclose(a.discharge_mw, b.discharge_mw)


def test_optimized_beats_rule_based_on_a_spiky_day(spiky_day, site):
    optimized = solve(ForecastResult.from_actuals(spiky_day), site, "cost")
    naive = rule_based_schedule(spiky_day, site, ("energy",))
    assert optimized.total_cost_eur <= naive.total_cost_eur + TOL
