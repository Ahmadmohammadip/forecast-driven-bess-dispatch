"""Interactive dashboard.

Per the brief: a **visualization layer, not the optimization engine**. Every
number on screen comes from the same functions the command-line pipeline calls,
so the app cannot drift from the reported results — if it disagrees with
`results/`, one of them is broken and both are worth checking.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bess_dispatch.config import load_config  # noqa: E402
from bess_dispatch.data.loaders import load_site_frame, split_frame  # noqa: E402
from bess_dispatch.evaluation.benchmark import (  # noqa: E402
    DEFAULT_ARMS,
    ablation_table,
    fit_forecasters,
    run_benchmark,
    summarise_benchmark,
)
from bess_dispatch.forecasting.features import FeatureSpec  # noqa: E402
from bess_dispatch.forecasting.interface import (  # noqa: E402
    ForecastResult,
    actuals_for,
    forecast_horizon,
)
from bess_dispatch.optimization.builder import OBJECTIVES, build_dispatch_model  # noqa: E402
from bess_dispatch.optimization.solve import solve_dispatch  # noqa: E402
from bess_dispatch.visualization import plots  # noqa: E402

st.set_page_config(page_title="Forecast-driven BESS dispatch", layout="wide")


@st.cache_data(show_spinner=False)
def cached_frame() -> pd.DataFrame:
    return load_site_frame()


@st.cache_resource(show_spinner="Fitting forecasters (once per session)…")
def cached_forecasters():
    frame = cached_frame()
    return fit_forecasters(split_frame(frame, "train"), 24, FeatureSpec())


@st.cache_data(show_spinner="Solving every arm over the test window…")
def cached_benchmark(_site, objective: str) -> pd.DataFrame:
    return run_benchmark(
        _site,
        objective=objective,
        frame=cached_frame(),
        fitted=cached_forecasters(),
    )


frame = cached_frame()
base = load_config()

st.title("Forecast-driven BESS dispatch")
st.caption(
    "ML forecasts of load, PV and wholesale price feeding a Pyomo cost-minimising "
    "dispatch LP. Real DE/LU market data, 2018-10 to 2020-09."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("System configuration")
    energy = st.slider("Energy capacity (MWh)", 0.25, 4.0, 1.0, 0.25)
    power = st.slider("Power rating (MW)", 0.1, 2.0, 0.5, 0.05)
    eta = st.slider(
        "One-way efficiency", 0.80, 1.0, 0.95, 0.01,
        help="Round-trip efficiency is this squared.",
    )
    degradation = st.slider(
        "Degradation cost (EUR/MWh throughput)", 0.0, 12.0, 2.0, 0.5,
        help="An economic proxy for wear, not an electrochemical model.",
    )

    st.header("Tariff")
    markup = st.slider("Import markup (EUR/MWh)", 0.0, 120.0, 60.0, 5.0)
    export_ratio = st.slider("Export ratio (x wholesale)", 0.0, 1.0, 0.7, 0.05)
    demand_charge = st.slider("Demand charge (EUR/MW of peak)", 0.0, 100.0, 5.0, 5.0)

    st.header("Objective")
    objective = st.selectbox(
        "Cost terms", list(OBJECTIVES), index=list(OBJECTIVES).index(base.objective)
    )

    st.caption(
        f"Round-trip efficiency {eta * eta:.3f}. "
        "Every figure below is recomputed from these settings."
    )

site = replace(
    base.site(),
    battery=replace(
        base.site().battery,
        energy_capacity_mwh=energy,
        p_charge_max_mw=power,
        p_discharge_max_mw=power,
        charge_efficiency=eta,
        discharge_efficiency=eta,
        degradation_cost_eur_mwh=degradation,
    ),
    tariff_policy=replace(
        base.site().tariff_policy,
        import_markup_eur_mwh=markup,
        export_ratio=export_ratio,
        demand_charge_eur_mw=demand_charge,
    ),
)

# The tariff guard is a real constraint, not a warning to click through: an
# export price above the import price can be arbitraged with no battery at all.
safe_markup = site.tariff_policy.minimum_safe_markup(
    frame["price_eur_mwh"].dropna().to_numpy()
)
if markup < safe_markup:
    st.error(
        f"This tariff is arbitrageable. With export paid at {export_ratio:.0%} of "
        f"wholesale, an import markup of at least **{safe_markup:.1f} EUR/MWh** is "
        f"needed; you have set {markup:.1f}. Below that threshold the "
        f"{int((frame['price_eur_mwh'] < 0).sum())} negative-price hours in this "
        "record let the site import and export simultaneously for a risk-free "
        "profit, with no battery involved — so any dispatch result would be "
        "meaningless. Raise the markup or the export ratio."
    )
    st.stop()

single_day, benchmark_tab, forecasts_tab, about_tab = st.tabs(
    ["Single day", "Benchmark & KPIs", "Forecasts", "About"]
)

# ------------------------------------------------------------- single day
with single_day:
    test = split_frame(frame, "test")
    days = sorted({str(stamp.date()) for stamp in test.index})
    chosen = st.select_slider("Day", options=days, value=days[15])
    issue = pd.Timestamp(chosen, tz="UTC")

    fitted = cached_forecasters()
    forecast = forecast_horizon(fitted, frame, issue, 24)
    truth = actuals_for(frame, forecast)

    solved = solve_dispatch(build_dispatch_model(forecast, site, objective=objective))
    perfect = solve_dispatch(
        build_dispatch_model(
            ForecastResult.from_actuals(truth), site, objective=objective
        )
    )

    left, middle, right = st.columns(3)
    left.metric("Forecast-driven cost", f"{solved.total_cost_eur:,.2f} EUR")
    middle.metric(
        "Perfect foresight",
        f"{perfect.total_cost_eur:,.2f} EUR",
        delta=f"{perfect.total_cost_eur - solved.total_cost_eur:,.2f}",
        delta_color="inverse",
    )
    right.metric("Cycles", f"{solved.throughput_mwh / (2 * energy):.2f}")

    st.pyplot(
        plots.plot_dispatch_day(
            solved,
            truth,
            timestamps=pd.DatetimeIndex(forecast.timestamps),
            title=f"Forecast-driven dispatch, {chosen}",
        )
    )
    st.pyplot(
        plots.plot_forecast_vs_perfect(
            solved,
            perfect,
            truth,
            forecast,
            timestamps=pd.DatetimeIndex(forecast.timestamps),
        )
    )
    st.caption(
        "A discharge in the wrong hour is not the optimizer failing — it is the "
        "optimizer being right about the wrong prices."
    )

# --------------------------------------------------------- benchmark tab
with benchmark_tab:
    st.write(
        "Every arm solved over the same 60 test days and priced against the same "
        "actuals. Only what the controller knew when it decided varies."
    )
    if st.button("Run the full benchmark", type="primary"):
        st.session_state["ran_benchmark"] = True

    if st.session_state.get("ran_benchmark"):
        results = cached_benchmark(site, objective)
        summary = summarise_benchmark(results)

        cols = st.columns(3)
        cols[0].metric(
            "Forecast-driven saving",
            f"{summary.loc['forecast', 'saving_eur']:,.0f} EUR",
            f"{summary.loc['forecast', 'saving_pct_wholesale']:.2f}% of wholesale",
        )
        cols[1].metric(
            "Perfect foresight",
            f"{summary.loc['perfect foresight', 'saving_eur']:,.0f} EUR",
        )
        cols[2].metric(
            "Value captured",
            f"{summary.loc['forecast', 'value_captured_pct']:.1f}%",
        )

        st.pyplot(plots.plot_arm_comparison(summary))
        st.pyplot(plots.plot_ablation(ablation_table(summary)))
        st.dataframe(summary.round(3), width="stretch")
    else:
        st.info(
            f"Solves {len(DEFAULT_ARMS)} arms over 60 days — a few seconds. "
            "Results are cached per configuration."
        )

# --------------------------------------------------------- forecasts tab
with forecasts_tab:
    st.write(
        "Models were chosen per series by validation MAE, not by reputation. "
        "Gradient boosting wins on load and **loses to a naive baseline on PV**."
    )
    selection_path = Path(__file__).resolve().parents[1] / "results" / "forecast_selection.csv"
    if selection_path.exists():
        table = pd.read_csv(selection_path)
        for target in table["target"].unique():
            st.subheader(target)
            block = table[table["target"] == target].set_index("model")
            columns = [
                column
                for column in ("MAE", "RMSE", "R2", "skill_vs_persistence", "selected")
                if column in block.columns
            ]
            st.dataframe(block[columns].round(4), width="stretch")
    else:
        st.info("Run `python -m bess_dispatch.forecasting.selection` to generate this table.")

# ------------------------------------------------------------- about tab
with about_tab:
    st.markdown(
        """
### What this is

A behind-the-meter battery dispatched against **forecasts**, not against known
prices. The forecasting exists because its output drives an optimization
decision — the whole project measures what that forecast is worth.

### The result

Forecast-driven day-ahead dispatch captures about **41%** of what perfect
information would be worth. Re-planning hourly instead lifts that to roughly the
same value perfect information buys in a day-ahead framing.

The ablation says where the rest goes, and it is not evenly split: making the
**price** forecast perfect closes 98.7% of the gap; load and PV together account
for 2.5%.

### What to be careful about

- Load and PV are national aggregates rescaled to one site. The shapes are real;
  the magnitudes are assumptions. A national load curve is smoother, and easier
  to forecast, than a single building's.
- The demand charge is levied on the **daily** peak here because the horizon is a
  day. A real one is monthly, and minimising each day's peak is not the same as
  minimising the month's.
- Degradation is a throughput proxy, not electrochemistry.

Full detail in `docs/formulation.md` and `docs/forecasting.md`.
"""
    )
