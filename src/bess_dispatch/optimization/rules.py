"""A rule-based controller: a battery operated without optimization.

The brief asks for "battery without optimization" as a scenario, and it is the
most important baseline in the study — more informative than the no-battery
case. Anyone can show that a battery beats no battery. The question a buyer
actually has is whether *optimizing* it beats the obvious controller that ships
in the inverter.

The rule here is greedy self-consumption, which is what most behind-the-meter
systems really do:

* surplus PV charges the battery instead of being exported;
* any shortfall discharges the battery instead of being imported;
* prices are never consulted.

It respects exactly the same power and energy limits as the optimized model, so
the comparison isolates the decision rule rather than the hardware.
"""

from __future__ import annotations

import numpy as np

from bess_dispatch.data.schema import SiteConfig, TimeSeriesData
from bess_dispatch.optimization.solve import DispatchResult


def rule_based_schedule(
    data: TimeSeriesData,
    site: SiteConfig,
    objective_terms: tuple[str, ...] = ("energy",),
) -> DispatchResult:
    """Greedy self-consumption over `data`, returned in the same shape a solve gives.

    Returning a `DispatchResult` means this arm flows through `evaluate_schedule`
    and the results tables unchanged, so it is priced by identical code.

    Note it makes no attempt to end at the starting state of charge. That is the
    point: a controller that does not look ahead cannot plan to, and forcing it
    to would be lending it foresight it does not have. The energy difference is
    reported rather than hidden.
    """
    battery = site.battery
    dt = site.dt_hours
    horizon = len(data)

    charge = np.zeros(horizon)
    discharge = np.zeros(horizon)
    soc = np.zeros(horizon)

    level = battery.initial_soc_mwh
    for t in range(horizon):
        surplus = data.pv_mw[t] - data.load_mw[t]
        if surplus > 0:
            headroom_mwh = battery.soc_max_mwh - level
            charge[t] = min(
                surplus,
                battery.p_charge_max_mw,
                headroom_mwh / (battery.charge_efficiency * dt),
            )
            charge[t] = max(charge[t], 0.0)
        elif surplus < 0:
            available_mwh = level - battery.soc_min_mwh
            discharge[t] = min(
                -surplus,
                battery.p_discharge_max_mw,
                available_mwh * battery.discharge_efficiency / dt,
            )
            discharge[t] = max(discharge[t], 0.0)

        level = (
            level
            + battery.charge_efficiency * charge[t] * dt
            - discharge[t] * dt / battery.discharge_efficiency
        )
        soc[t] = level

    net = data.load_mw + charge - data.pv_mw - discharge
    grid_import = np.clip(net, 0, None)
    grid_export = np.minimum(np.clip(-net, 0, None), site.grid.export_limit_mw)
    curtailment = np.clip(np.clip(-net, 0, None) - site.grid.export_limit_mw, 0, None)

    tariff = site.tariff_policy.apply(data.price_eur_mwh)
    energy_cost = float(
        np.sum(
            tariff.import_price_eur_mwh * grid_import
            - tariff.export_price_eur_mwh * grid_export
        )
        * dt
    )
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
    total = sum(components[term] for term in objective_terms)

    return DispatchResult(
        grid_import_mw=grid_import,
        grid_export_mw=grid_export,
        charge_mw=charge,
        discharge_mw=discharge,
        soc_mwh=soc,
        curtailment_mw=curtailment,
        peak_import_mw=peak,
        total_cost_eur=total,
        energy_cost_eur=energy_cost,
        degradation_cost_eur=degradation,
        demand_charge_eur=demand_charge,
        objective_terms=tuple(objective_terms),
        solve_time_s=0.0,
        termination="rule-based (no solve)",
        dt_hours=dt,
        metadata={
            "energy_capacity_mwh": battery.energy_capacity_mwh,
            "battery": battery.name,
            "forecast": "none (greedy self-consumption)",
            "perfect_foresight": False,
            "terminal_soc_mwh": float(soc[-1]) if horizon else None,
        },
    )
