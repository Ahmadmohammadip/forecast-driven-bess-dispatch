"""The dispatch model: a pure linear program, built from a forecast.

Takes a `ForecastResult` and a `SiteConfig` and returns a Pyomo
`ConcreteModel`. It never reads a file, never sees a DataFrame, and cannot tell
a forecast from actuals — which is what lets the perfect-foresight benchmark run
through exactly this code.

**No binary variables anywhere**, and that is a measured claim rather than a
simplification. Probe runs on real data across export ratios 0.7x, 1.0x and
1.3x, with and without degradation cost and with and without a demand charge,
never once produced an optimum that charged and discharged in the same period.
Round-trip losses already make simultaneous charge/discharge strictly wasteful,
so the constraint forbidding it would never bind.

What the probe *did* find was a failure mode that looks similar and is not: with
export compensation above the import tariff, the optimum imports and exports
simultaneously in every hour, with no battery in the model at all. That is the
meter being gamed, and the fix belongs in `Tariff` validation, not here. Adding
a binary would have made an unphysical tariff solve slowly instead of failing.
"""

from __future__ import annotations

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Var,
    minimize,
)

from bess_dispatch.data.schema import SiteConfig, Tariff
from bess_dispatch.forecasting.interface import ForecastResult

# Which cost terms each objective variant includes. The brief asks for all
# three to be run, not for one to be chosen.
OBJECTIVES: dict[str, tuple[str, ...]] = {
    "cost": ("energy",),
    "cost_degradation": ("energy", "degradation"),
    "cost_degradation_demand": ("energy", "degradation", "demand_charge"),
}


def build_dispatch_model(
    forecast: ForecastResult,
    site: SiteConfig,
    tariff: Tariff | None = None,
    objective: str = "cost",
) -> ConcreteModel:
    """Build the dispatch LP for one horizon.

    `tariff` defaults to applying the site's `TariffPolicy` to the forecast
    prices. Passing one explicitly is how the rolling-horizon controller keeps
    the tariff fixed while the price forecast moves.
    """
    if objective not in OBJECTIVES:
        raise KeyError(
            f"unknown objective {objective!r}; expected one of {sorted(OBJECTIVES)}"
        )
    terms = OBJECTIVES[objective]

    tariff = tariff or site.tariff_policy.apply(forecast.price_forecast_eur_mwh)
    if tariff.n_periods != len(forecast):
        raise ValueError(
            f"tariff covers {tariff.n_periods} periods but the forecast covers "
            f"{len(forecast)}"
        )

    battery = site.battery
    horizon = len(forecast)
    dt = site.dt_hours

    model = ConcreteModel(name=f"bess_dispatch[{objective}]")
    model.T = RangeSet(0, horizon - 1)

    # --- data -----------------------------------------------------------
    model.p_load = Param(model.T, initialize=dict(enumerate(forecast.load_forecast_mw)))
    model.p_pv = Param(model.T, initialize=dict(enumerate(forecast.pv_forecast_mw)))
    model.buy = Param(model.T, initialize=dict(enumerate(tariff.import_price_eur_mwh)))
    model.sell = Param(model.T, initialize=dict(enumerate(tariff.export_price_eur_mwh)))
    model.dt = Param(initialize=dt)

    # --- decisions ------------------------------------------------------
    model.g_imp = Var(model.T, bounds=(0, site.grid.import_limit_mw))
    model.g_exp = Var(model.T, bounds=(0, site.grid.export_limit_mw))
    model.p_ch = Var(model.T, bounds=(0, battery.p_charge_max_mw))
    model.p_dis = Var(model.T, bounds=(0, battery.p_discharge_max_mw))
    model.soc = Var(model.T, bounds=(battery.soc_min_mwh, battery.soc_max_mwh))
    # Curtailment is a real decision, not a slack. Without it, a period whose PV
    # exceeds load plus charging plus the export limit would be infeasible --
    # the model would report "impossible" where reality just spills the surplus.
    model.curtail = Var(model.T, bounds=(0, None))
    # Peak grid import over the horizon, for the demand charge. Kept as a
    # variable with a >= constraint rather than a max(), which keeps it linear.
    model.peak = Var(within=NonNegativeReals)

    # --- constraints ----------------------------------------------------
    def power_balance(m, t):
        """Load + charging + export == usable PV + discharge + import."""
        return (
            m.p_load[t] + m.p_ch[t] + m.g_exp[t]
            == (m.p_pv[t] - m.curtail[t]) + m.p_dis[t] + m.g_imp[t]
        )

    model.power_balance = Constraint(model.T, rule=power_balance)

    def curtailment_limit(m, t):
        return m.curtail[t] <= m.p_pv[t]

    model.curtailment_limit = Constraint(model.T, rule=curtailment_limit)

    def soc_dynamics(m, t):
        """soc[t] = soc[t-1] + eta_c * charge * dt - discharge * dt / eta_d.

        Charging is derated on the way in and discharging on the way out, so a
        full cycle returns eta_c * eta_d of what went in.
        """
        previous = battery.initial_soc_mwh if t == m.T.first() else m.soc[t - 1]
        return m.soc[t] == (
            previous
            + battery.charge_efficiency * m.p_ch[t] * m.dt
            - m.p_dis[t] * m.dt / battery.discharge_efficiency
        )

    model.soc_dynamics = Constraint(model.T, rule=soc_dynamics)

    if site.enforce_terminal_soc:
        # Without this a finite horizon ends by selling the battery empty, and
        # the reported cost is flattered by energy that was never paid for.
        model.terminal_soc = Constraint(
            expr=model.soc[model.T.last()] == battery.initial_soc_mwh
        )

    def peak_definition(m, t):
        return m.peak >= m.g_imp[t]

    model.peak_definition = Constraint(model.T, rule=peak_definition)

    # --- costs, as named expressions ------------------------------------
    # Built as Expressions so the reported breakdown is read back from the same
    # objects the objective is made of, rather than recomputed by a second
    # calculation that could disagree with it.
    model.energy_cost = Expression(
        expr=sum(
            (model.buy[t] * model.g_imp[t] - model.sell[t] * model.g_exp[t]) * model.dt
            for t in model.T
        )
    )
    model.degradation_cost = Expression(
        expr=battery.degradation_cost_eur_mwh
        * sum((model.p_ch[t] + model.p_dis[t]) * model.dt for t in model.T)
    )
    model.demand_charge_cost = Expression(expr=tariff.demand_charge_eur_mw * model.peak)

    available = {
        "energy": model.energy_cost,
        "degradation": model.degradation_cost,
        "demand_charge": model.demand_charge_cost,
    }
    model.objective = Objective(
        expr=sum(available[term] for term in terms), sense=minimize
    )

    # Stashed for solve.py, so it can report a breakdown without being told
    # again which variant was built.
    model._objective_terms = terms
    model._tariff = tariff
    model._site = site
    model._forecast_metadata = forecast.metadata
    return model


def build_no_battery_model(
    forecast: ForecastResult,
    site: SiteConfig,
    tariff: Tariff | None = None,
    objective: str = "cost",
) -> ConcreteModel:
    """The same site with the battery pinned at zero — the do-nothing baseline.

    Built by fixing the battery variables rather than by writing a second model,
    so the baseline cost is computed by identical code and any modelling error
    affects both arms equally.
    """
    model = build_dispatch_model(forecast, site, tariff, objective)
    for t in model.T:
        model.p_ch[t].fix(0.0)
        model.p_dis[t].fix(0.0)
        model.soc[t].fix(site.battery.initial_soc_mwh)
    return model
