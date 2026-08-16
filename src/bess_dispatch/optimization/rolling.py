"""Rolling-horizon control: what the system would actually do in operation.

The day-ahead benchmark answers "how good is a plan made once a day?". This
answers the operational question: a controller that re-forecasts and re-solves
every hour, commits only the next interval, and then does it again with one
more hour of observed data.

The discipline the brief insists on — *never allow future actual data to enter
the forecast at the current time* — is not enforced here at all. It is enforced
one level down, by `build_features` discarding every row at or after the issue
time. This module simply cannot violate it: it has no path to the future that
does not go through that function.

Two design choices worth stating:

**Only the first `commit_periods` intervals are executed.** The rest of each
solve is discarded. That is the point of receding-horizon control — the tail of
the plan exists to stop the controller behaving myopically at the boundary, not
to be carried out.

**Terminal state of charge is enforced on each lookahead window.** Without it,
every solve ends by dumping the battery in its final period, and since only the
first interval is committed the controller would drain a little more each hour
and never refill. It is a stand-in for a terminal value function, which is the
more principled fix and is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from bess_dispatch.data.schema import SiteConfig
from bess_dispatch.forecasting.interface import actuals_for, forecast_horizon
from bess_dispatch.optimization.builder import OBJECTIVES, build_dispatch_model
from bess_dispatch.optimization.solve import soc_settlement_eur, solve_dispatch


@dataclass
class RollingResult:
    """What the controller actually did, hour by hour, over the whole run."""

    timestamps: pd.DatetimeIndex
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    grid_import_mw: np.ndarray
    grid_export_mw: np.ndarray
    curtailment_mw: np.ndarray

    energy_cost_eur: float
    degradation_cost_eur: float
    demand_charge_eur: float
    total_cost_eur: float
    wholesale_energy_cost_eur: float

    soc_settlement_eur: float
    terminal_soc_mwh: float
    initial_soc_mwh: float

    objective_terms: tuple[str, ...]
    n_solves: int
    total_solve_time_s: float
    dt_hours: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def peak_import_mw(self) -> float:
        return float(self.grid_import_mw.max()) if self.grid_import_mw.size else 0.0

    @property
    def throughput_mwh(self) -> float:
        return float((self.charge_mw.sum() + self.discharge_mw.sum()) * self.dt_hours)

    def cost_breakdown(self) -> dict[str, float]:
        available = {
            "energy": self.energy_cost_eur,
            "degradation": self.degradation_cost_eur,
            "demand_charge": self.demand_charge_eur,
        }
        return {term: available[term] for term in self.objective_terms}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "charge_mw": self.charge_mw,
                "discharge_mw": self.discharge_mw,
                "soc_mwh": self.soc_mwh,
                "grid_import_mw": self.grid_import_mw,
                "grid_export_mw": self.grid_export_mw,
                "curtailment_mw": self.curtailment_mw,
            },
            index=self.timestamps,
        )

    def summary(self) -> str:
        return (
            f"{self.total_cost_eur:,.2f} EUR over {len(self.timestamps)} hours "
            f"from {self.n_solves} solves - peak {self.peak_import_mw:.3f} MW, "
            f"throughput {self.throughput_mwh:.2f} MWh, "
            f"{self.total_solve_time_s:.1f}s total"
        )


def run_rolling_horizon(
    frame: pd.DataFrame,
    site: SiteConfig,
    fitted: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookahead: int = 24,
    commit_periods: int = 1,
    objective: str = "cost_degradation_demand",
    solver_name: str = "appsi_highs",
    perfect_foresight: bool = False,
) -> RollingResult:
    """Run the receding-horizon controller from `start` to `end` (exclusive).

    Set `perfect_foresight=True` to run the identical loop with actuals in place
    of forecasts. That isolates the cost of forecast error from the cost of the
    rolling structure itself, which are different things and are often
    conflated.
    """
    if commit_periods < 1:
        raise ValueError(f"commit_periods must be >= 1, got {commit_periods}")
    if commit_periods > lookahead:
        raise ValueError(
            f"commit_periods ({commit_periods}) cannot exceed lookahead ({lookahead})"
        )
    terms = OBJECTIVES[objective]

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tz is None:
        start = start.tz_localize("UTC")
    if end.tz is None:
        end = end.tz_localize("UTC")

    dt = site.dt_hours
    battery = site.battery
    step = pd.Timedelta(hours=1)

    executed: list[dict] = []
    level = battery.initial_soc_mwh
    n_solves = 0
    solve_time = 0.0

    # Commit inside [start, end); look ahead beyond it. The lookahead window is
    # allowed to run past `end` as long as the data exists, because a real
    # controller on 28 February genuinely can see into March. Stopping the loop
    # a lookahead short of `end` instead would silently cover fewer hours than
    # the day-ahead arms and make the totals incomparable.
    last_usable = frame.index[-1] - step * (lookahead - 1)
    decision_time = start
    while decision_time < end and decision_time <= last_usable:
        forecast = forecast_horizon(fitted, frame, decision_time, lookahead)
        truth = actuals_for(frame, forecast)
        if perfect_foresight:
            from bess_dispatch.forecasting.interface import ForecastResult

            forecast = ForecastResult.from_actuals(truth)

        window_site = replace(
            site,
            battery=replace(
                battery,
                initial_soc_frac=float(
                    np.clip(
                        level / battery.energy_capacity_mwh,
                        battery.soc_min_frac,
                        battery.soc_max_frac,
                    )
                ),
            ),
        )
        schedule = solve_dispatch(
            build_dispatch_model(forecast, window_site, objective=objective),
            solver_name,
        )
        n_solves += 1
        solve_time += schedule.solve_time_s

        # Execute only the committed head of the plan, against real conditions.
        for k in range(commit_periods):
            charge, discharge, level = _apply_one_period(
                schedule.charge_mw[k], schedule.discharge_mw[k], level, battery, dt
            )
            net = truth.load_mw[k] + charge - truth.pv_mw[k] - discharge
            grid_import = max(net, 0.0)
            surplus = max(-net, 0.0)
            grid_export = min(surplus, site.grid.export_limit_mw)
            executed.append(
                {
                    "timestamp": decision_time + step * k,
                    "charge_mw": charge,
                    "discharge_mw": discharge,
                    "soc_mwh": level,
                    "grid_import_mw": grid_import,
                    "grid_export_mw": grid_export,
                    "curtailment_mw": surplus - grid_export,
                    "price_eur_mwh": truth.price_eur_mwh[k],
                }
            )

        decision_time = decision_time + step * commit_periods

    if not executed:
        raise ValueError(
            f"no decision times between {start} and {end} fit a {lookahead}-period "
            "lookahead; widen the window or shorten the lookahead"
        )

    log = pd.DataFrame(executed).set_index("timestamp")
    tariff = site.tariff_policy.apply(log["price_eur_mwh"].to_numpy())

    energy_cost = float(
        np.sum(
            tariff.import_price_eur_mwh * log["grid_import_mw"].to_numpy()
            - tariff.export_price_eur_mwh * log["grid_export_mw"].to_numpy()
        )
        * dt
    )
    wholesale = float(
        np.sum(log["price_eur_mwh"].to_numpy() * log["grid_import_mw"].to_numpy()) * dt
    )
    degradation = float(
        battery.degradation_cost_eur_mwh
        * np.sum(log["charge_mw"].to_numpy() + log["discharge_mw"].to_numpy())
        * dt
    )
    peak = float(log["grid_import_mw"].max())
    demand_charge = float(tariff.demand_charge_eur_mw * peak)
    # Settle the state of charge the run ends on. A controller that finishes
    # depleted has taken energy it never paid for, and over a long run that
    # silently inflates its saving.
    reference_price = float(tariff.import_price_eur_mwh.mean())
    settlement = soc_settlement_eur(
        battery.initial_soc_mwh, level, battery, reference_price
    )

    components = {
        "energy": energy_cost + settlement,
        "degradation": degradation,
        "demand_charge": demand_charge,
    }

    return RollingResult(
        timestamps=log.index,
        charge_mw=log["charge_mw"].to_numpy(),
        discharge_mw=log["discharge_mw"].to_numpy(),
        soc_mwh=log["soc_mwh"].to_numpy(),
        grid_import_mw=log["grid_import_mw"].to_numpy(),
        grid_export_mw=log["grid_export_mw"].to_numpy(),
        curtailment_mw=log["curtailment_mw"].to_numpy(),
        energy_cost_eur=energy_cost,
        degradation_cost_eur=degradation,
        demand_charge_eur=demand_charge,
        total_cost_eur=sum(components[term] for term in terms),
        wholesale_energy_cost_eur=wholesale,
        soc_settlement_eur=settlement,
        terminal_soc_mwh=float(level),
        initial_soc_mwh=battery.initial_soc_mwh,
        objective_terms=terms,
        n_solves=n_solves,
        total_solve_time_s=solve_time,
        dt_hours=dt,
        metadata={
            "lookahead": lookahead,
            "commit_periods": commit_periods,
            "perfect_foresight": perfect_foresight,
            "energy_capacity_mwh": battery.energy_capacity_mwh,
        },
    )


def _apply_one_period(charge: float, discharge: float, level: float, battery, dt: float):
    """Clip a committed action to what the state of charge allows, then apply it.

    The plan was feasible against forecast conditions; under real ones the only
    way it can fail is through the energy band, since the power limits were
    already respected when it was built. Clipping there is what a real
    controller does — letting the level drift outside the band would invent
    storage that does not exist.
    """
    headroom = (battery.soc_max_mwh - level) / (battery.charge_efficiency * dt)
    charge = min(charge, max(headroom, 0.0))
    available = (level - battery.soc_min_mwh) * battery.discharge_efficiency / dt
    discharge = min(discharge, max(available, 0.0))

    level = (
        level
        + battery.charge_efficiency * charge * dt
        - discharge * dt / battery.discharge_efficiency
    )
    return charge, discharge, level
