"""Scenarios and sensitivity sweeps — the brief's section 16.

Two kinds of variation, kept apart because they mean different things:

* **Scenarios** change the *world*: dearer electricity, a worse solar year, a
  heavier load. Implemented by scaling the input series, which is honest as long
  as it is labelled — scaling prices by 1.5 preserves their shape and therefore
  keeps the arbitrage opportunity structurally identical while changing its
  size.
* **Sensitivities** change the *system*: capacity, power rating, round-trip
  efficiency, degradation cost. These are procurement questions.

Both run the day-ahead benchmark rather than the rolling controller. A sweep of
20 variants at 1,440 solves each is an hour of compute to answer a question the
cheaper arm already answers; the rolling controller is run once, in
`controllers.py`, on the reference case.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from bess_dispatch.data.loaders import load_site_frame
from bess_dispatch.data.schema import SiteConfig
from bess_dispatch.evaluation.benchmark import Arm, run_benchmark, summarise_benchmark

# A trimmed arm set: the sweeps ask how the *system* responds, so the ablation
# arms would multiply runtime without changing the answer.
SWEEP_ARMS = (
    Arm("no battery", "no_battery"),
    Arm("rule-based", "rule"),
    Arm("forecast", "optimized"),
    Arm("perfect foresight", "optimized", perfect=True),
)


def scale_frame(
    frame: pd.DataFrame,
    price: float = 1.0,
    pv: float = 1.0,
    load: float = 1.0,
) -> pd.DataFrame:
    """Scale the input series multiplicatively.

    Shape-preserving by construction, which is the point and also the
    limitation: a 1.5x price scenario is "the same market, dearer", not "a more
    volatile market". Volatility is varied separately by
    `scale_price_volatility`, because multiplying every price by a constant
    leaves the within-day spread proportional and the arbitrage structure
    untouched.
    """
    scaled = frame.copy()
    scaled["price_eur_mwh"] = scaled["price_eur_mwh"] * price
    scaled["pv_mw"] = scaled["pv_mw"] * pv
    scaled["load_mw"] = scaled["load_mw"] * load
    if "tso_load_forecast_mw" in scaled.columns:
        scaled["tso_load_forecast_mw"] = scaled["tso_load_forecast_mw"] * load
    return scaled


def scale_price_volatility(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Stretch prices about each day's own mean, holding the daily level fixed.

    This is the variation a battery actually cares about. Doubling volatility
    doubles the within-day spread while leaving the average price paid roughly
    unchanged, so any change in savings is attributable to the spread rather
    than to the bill simply getting bigger.
    """
    scaled = frame.copy()
    price = scaled["price_eur_mwh"]
    daily_mean = price.groupby(scaled.index.floor("D")).transform("mean")
    scaled["price_eur_mwh"] = daily_mean + (price - daily_mean) * factor
    return scaled


def _run_variant(
    label: str,
    site: SiteConfig,
    frame: pd.DataFrame,
    objective: str,
    split: str,
    max_days: int | None,
) -> pd.DataFrame:
    results = run_benchmark(
        site,
        split=split,
        objective=objective,
        arms=SWEEP_ARMS,
        frame=frame,
        max_days=max_days,
    )
    summary = summarise_benchmark(results).reset_index()
    summary.insert(0, "variant", label)
    return summary


def run_scenarios(
    site: SiteConfig,
    objective: str = "cost_degradation_demand",
    split: str = "test",
    frame: pd.DataFrame | None = None,
    max_days: int | None = None,
) -> pd.DataFrame:
    """The brief's world scenarios: dearer power, worse sun, heavier load."""
    frame = frame if frame is not None else load_site_frame()
    variants = {
        "reference": frame,
        "prices +50%": scale_frame(frame, price=1.5),
        "prices -30%": scale_frame(frame, price=0.7),
        "PV -50%": scale_frame(frame, pv=0.5),
        "load +30%": scale_frame(frame, load=1.3),
        "volatility x2": scale_price_volatility(frame, 2.0),
        "volatility x0.5": scale_price_volatility(frame, 0.5),
    }
    blocks = [
        _run_variant(label, site, variant_frame, objective, split, max_days)
        for label, variant_frame in variants.items()
    ]
    return pd.concat(blocks, ignore_index=True).set_index(["variant", "arm"])


def run_sensitivities(
    site: SiteConfig,
    objective: str = "cost_degradation_demand",
    split: str = "test",
    frame: pd.DataFrame | None = None,
    max_days: int | None = None,
) -> pd.DataFrame:
    """The brief's system sensitivities: sizing, efficiency, degradation cost."""
    frame = frame if frame is not None else load_site_frame()
    battery = site.battery
    variants: dict[str, SiteConfig] = {}

    # Sizing: energy and power scaled together, which is how batteries are
    # actually procured -- a 2 MWh unit does not ship with a 0.5 MW inverter.
    for energy, power in ((0.5, 0.25), (1.0, 0.5), (2.0, 1.0), (4.0, 2.0)):
        variants[f"size {energy} MWh / {power} MW"] = replace(
            site,
            battery=replace(
                battery,
                energy_capacity_mwh=energy,
                p_charge_max_mw=power,
                p_discharge_max_mw=power,
            ),
        )
    # Power alone, energy held fixed: how much does the C-rate matter?
    for power in (0.25, 1.0):
        variants[f"power {power} MW at 1.0 MWh"] = replace(
            site,
            battery=replace(battery, p_charge_max_mw=power, p_discharge_max_mw=power),
        )
    for eta in (0.85, 0.90, 0.95, 1.00):
        variants[f"round-trip {eta * eta:.2f}"] = replace(
            site,
            battery=replace(battery, charge_efficiency=eta, discharge_efficiency=eta),
        )
    for cost in (0.0, 2.0, 5.0, 10.0):
        variants[f"degradation {cost:.0f} EUR/MWh"] = replace(
            site, battery=replace(battery, degradation_cost_eur_mwh=cost)
        )
    for charge in (0.0, 5.0, 50.0):
        variants[f"demand charge {charge:.0f} EUR/MW"] = replace(
            site,
            tariff_policy=replace(site.tariff_policy, demand_charge_eur_mw=charge),
        )

    blocks = [
        _run_variant(label, variant_site, frame, objective, split, max_days)
        for label, variant_site in variants.items()
    ]
    return pd.concat(blocks, ignore_index=True).set_index(["variant", "arm"])


def sizing_curve(
    sensitivities: pd.DataFrame, arm: str = "perfect foresight"
) -> pd.DataFrame:
    """Pull the sizing rows out of a sensitivity sweep, for the README."""
    sizes = [
        variant
        for variant in sensitivities.index.get_level_values("variant").unique()
        if variant.startswith("size ")
    ]
    rows = sensitivities.loc[(sizes, arm), :].reset_index()
    return rows.set_index("variant")[
        ["total_cost_eur", "saving_eur", "saving_pct", "saving_pct_wholesale"]
    ]


def main() -> int:
    from bess_dispatch.config import load_config

    config = load_config()
    site = config.site()
    out_dir = Path(__file__).resolve().parents[3] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = run_scenarios(site, objective=config.objective)
    scenarios.to_csv(out_dir / "scenarios.csv")
    print("=== scenarios: saving vs no battery, EUR over 60 days ===")
    print(scenarios["saving_eur"].unstack("arm").round(2).to_string())

    sensitivities = run_sensitivities(site, objective=config.objective)
    sensitivities.to_csv(out_dir / "sensitivities.csv")
    print("\n=== sensitivities: saving vs no battery, EUR over 60 days ===")
    print(sensitivities["saving_eur"].unstack("arm").round(2).to_string())

    print("\n=== sizing curve (perfect foresight) ===")
    print(sizing_curve(sensitivities).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
