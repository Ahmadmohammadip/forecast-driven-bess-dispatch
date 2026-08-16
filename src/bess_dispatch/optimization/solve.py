"""Solver interface and result extraction.

The model is a pure LP, so HiGHS is the only solver needed — no conda, no
Ipopt, no MILP branch-and-bound. `solver_name` exists so the project is not
welded to one solver, per the brief's request for a solver abstraction; CBC and
GLPK work unchanged if their binaries are on PATH.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import TerminationCondition

from bess_dispatch.data.schema import SiteConfig, TimeSeriesData

DEFAULT_SOLVER = "appsi_highs"


@dataclass
class DispatchResult:
    """A solved schedule, plus everything needed to interpret it.

    Arrays are indexed by period. Costs are in EUR over the horizon.
    """

    grid_import_mw: np.ndarray
    grid_export_mw: np.ndarray
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    curtailment_mw: np.ndarray
    peak_import_mw: float

    total_cost_eur: float
    energy_cost_eur: float
    degradation_cost_eur: float
    demand_charge_eur: float

    objective_terms: tuple[str, ...]
    solve_time_s: float
    termination: str
    dt_hours: float = 1.0
    metadata: dict = field(default_factory=dict)

    def cost_breakdown(self) -> dict[str, float]:
        """Components of the objective. Sums to `total_cost_eur` by construction.

        Only the terms the chosen variant actually optimised are included --
        listing a term that was not in the objective would make the breakdown
        disagree with the total.
        """
        available = {
            "energy": self.energy_cost_eur,
            "degradation": self.degradation_cost_eur,
            "demand_charge": self.demand_charge_eur,
        }
        return {term: available[term] for term in self.objective_terms}

    @property
    def throughput_mwh(self) -> float:
        """Total energy through the battery, charge plus discharge."""
        return float((self.charge_mw.sum() + self.discharge_mw.sum()) * self.dt_hours)

    @property
    def equivalent_full_cycles(self) -> float | None:
        capacity = self.metadata.get("energy_capacity_mwh")
        if not capacity:
            return None
        return self.throughput_mwh / (2 * capacity)

    @property
    def net_battery_mw(self) -> np.ndarray:
        """Discharge minus charge: positive when the battery is supporting the site."""
        return self.discharge_mw - self.charge_mw

    def summary(self) -> str:
        return (
            f"{self.total_cost_eur:,.2f} EUR over {len(self.grid_import_mw)} periods "
            f"- peak import {self.peak_import_mw:.3f} MW, "
            f"throughput {self.throughput_mwh:.2f} MWh, "
            f"{self.termination}, {self.solve_time_s:.2f}s"
        )


def solve_dispatch(
    model: ConcreteModel,
    solver_name: str = DEFAULT_SOLVER,
    time_limit: float | None = None,
) -> DispatchResult:
    """Solve and extract. Raises `RuntimeError` unless the solve is optimal."""
    solver = SolverFactory(solver_name)
    if time_limit is not None:
        solver.options["time_limit"] = time_limit

    started = time.perf_counter()
    # Load solutions explicitly rather than letting the solve call do it: on an
    # infeasible model the implicit load raises a confusing error from inside
    # Pyomo before this code gets the chance to say what actually went wrong.
    results = solver.solve(model, load_solutions=False)
    elapsed = time.perf_counter() - started

    condition = results.solver.termination_condition
    if condition != TerminationCondition.optimal:
        raise RuntimeError(_explain_failure(condition, model, solver_name))
    model.solutions.load_from(results)

    periods = sorted(model.T)
    extract = lambda variable: np.array(  # noqa: E731
        [value(variable[t]) for t in periods], dtype=float
    )

    site: SiteConfig = model._site
    return DispatchResult(
        grid_import_mw=extract(model.g_imp),
        grid_export_mw=extract(model.g_exp),
        charge_mw=extract(model.p_ch),
        discharge_mw=extract(model.p_dis),
        soc_mwh=extract(model.soc),
        curtailment_mw=extract(model.curtail),
        peak_import_mw=float(value(model.peak)),
        total_cost_eur=float(value(model.objective)),
        energy_cost_eur=float(value(model.energy_cost)),
        degradation_cost_eur=float(value(model.degradation_cost)),
        demand_charge_eur=float(value(model.demand_charge_cost)),
        objective_terms=tuple(model._objective_terms),
        solve_time_s=elapsed,
        termination=str(condition),
        dt_hours=float(value(model.dt)),
        metadata={
            "energy_capacity_mwh": site.battery.energy_capacity_mwh,
            "battery": site.battery.name,
            "forecast": model._forecast_metadata.label(),
            "perfect_foresight": model._forecast_metadata.is_perfect_foresight,
        },
    )


def _explain_failure(condition, model: ConcreteModel, solver_name: str) -> str:
    """Name the likely causes rather than forwarding the solver's status code."""
    if condition == TerminationCondition.infeasible:
        site: SiteConfig = model._site
        return (
            f"the dispatch LP is infeasible. On this model that almost always means "
            f"one of:\n"
            f"  - net load exceeds the grid import limit "
            f"({site.grid.import_limit_mw} MW) in some period;\n"
            f"  - surplus PV exceeds the export limit "
            f"({site.grid.export_limit_mw} MW) plus what the battery can absorb, "
            f"and curtailment is somehow bounded;\n"
            f"  - the terminal state-of-charge constraint cannot be met within the "
            f"power limits over this horizon "
            f"(enforce_terminal_soc={site.enforce_terminal_soc}).\n"
            f"The first two are data problems; the third is usually a horizon that "
            f"is too short for the battery's C-rate."
        )
    if condition == TerminationCondition.maxTimeLimit:
        return (
            "the solver hit its time limit. That is surprising for an LP this size "
            "and usually means the horizon is far longer than intended."
        )
    return f"solver {solver_name!r} terminated with {condition}, not an optimal solution"


def evaluate_schedule(
    result: DispatchResult,
    actuals: TimeSeriesData,
    site: SiteConfig,
    tariff=None,
) -> dict[str, float]:
    """Re-price a schedule against what actually happened.

    This is the honest way to score a forecast-driven plan. The controller
    committed to charging and discharging based on a forecast; those battery
    actions are what it did. The grid import that *resulted* is then whatever
    the real load and PV required, not what the plan assumed — so import is
    recomputed from the balance rather than taken from the plan.

    Any shortfall the battery cannot cover is met by the grid, which is what
    would physically happen. Where the committed schedule would drive the state
    of charge outside its band under real conditions, the battery action is
    clipped first, exactly as a real controller's limits would clip it.
    """
    tariff = tariff or site.tariff_policy.apply(actuals.price_eur_mwh)
    dt = site.dt_hours
    battery = site.battery

    charge, discharge, soc = _apply_with_soc_limits(result, battery, dt)

    # Whatever the site could not self-supply comes from the grid.
    net = actuals.load_mw + charge - actuals.pv_mw - discharge
    grid_import = np.clip(net, 0, None)
    grid_export = np.clip(-net, 0, None)
    curtailed = np.clip(grid_export - site.grid.export_limit_mw, 0, None)
    grid_export = np.minimum(grid_export, site.grid.export_limit_mw)

    energy_cost = float(
        np.sum(
            tariff.import_price_eur_mwh * grid_import
            - tariff.export_price_eur_mwh * grid_export
        )
        * dt
    )
    # The same import volume priced at bare wholesale, with no markup. The
    # battery can only move the wholesale component -- network charges and
    # levies ride along with every MWh regardless of when it is taken -- so
    # savings expressed against the whole bill understate what the controller
    # is actually doing. Both denominators are reported.
    wholesale_cost = float(np.sum(actuals.price_eur_mwh * grid_import) * dt)
    degradation = float(
        battery.degradation_cost_eur_mwh * np.sum(charge + discharge) * dt
    )
    peak = float(grid_import.max()) if grid_import.size else 0.0
    demand_charge = float(tariff.demand_charge_eur_mw * peak)

    components = {
        "energy": energy_cost,
        "degradation": degradation,
        "demand_charge": demand_charge,
    }
    realised = sum(components[term] for term in result.objective_terms)

    return {
        "realised_cost_eur": realised,
        "planned_cost_eur": result.total_cost_eur,
        "realised_energy_cost_eur": energy_cost,
        "realised_degradation_eur": degradation,
        "realised_demand_charge_eur": demand_charge,
        "wholesale_energy_cost_eur": wholesale_cost,
        "peak_import_mw": peak,
        "terminal_soc_mwh": float(soc[-1]) if soc.size else battery.initial_soc_mwh,
        "grid_import_mwh": float(grid_import.sum() * dt),
        "grid_export_mwh": float(grid_export.sum() * dt),
        "curtailed_mwh": float(curtailed.sum() * dt),
        "throughput_mwh": float(np.sum(charge + discharge) * dt),
        "clipped_periods": float(np.sum(~np.isclose(charge, result.charge_mw))
                                 + np.sum(~np.isclose(discharge, result.discharge_mw))),
    }


def _apply_with_soc_limits(result: DispatchResult, battery, dt: float):
    """Replay the committed charge/discharge, clipping at the energy limits.

    A schedule built on a forecast can be infeasible under real conditions only
    through the state of charge, since the power limits were respected when it
    was built. Clipping there is what a real controller does; letting the state
    of charge drift outside its band would silently invent storage capacity.
    """
    charge = result.charge_mw.copy()
    discharge = result.discharge_mw.copy()
    soc = np.empty_like(charge)

    level = battery.initial_soc_mwh
    for t in range(charge.size):
        headroom = (battery.soc_max_mwh - level) / (battery.charge_efficiency * dt)
        charge[t] = min(charge[t], max(headroom, 0.0))
        available = (level - battery.soc_min_mwh) * battery.discharge_efficiency / dt
        discharge[t] = min(discharge[t], max(available, 0.0))

        level = (
            level
            + battery.charge_efficiency * charge[t] * dt
            - discharge[t] * dt / battery.discharge_efficiency
        )
        soc[t] = level

    return charge, discharge, soc
