"""Generate notebooks/02_walkthrough.ipynb.

The end-to-end story in one place: fit, forecast, solve, price, compare. Like
the EDA notebook it holds no logic of its own — it narrates over the package,
so it cannot drift from what the pipeline actually does.

    python notebooks/build_walkthrough_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# 2 — Walkthrough: what is a forecast worth?\n"
        "\n"
        "The whole pipeline on one day, then on sixty.\n"
        "\n"
        "The question is deliberately not *how accurate are the models*. It is\n"
        "**how much money does the accuracy buy**, and **which of the three\n"
        "forecasts the money depends on** — which turns out not to be an even\n"
        "split.",
    ),
    (
        "code",
        "%matplotlib inline\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from bess_dispatch.config import load_config\n"
        "from bess_dispatch.data.loaders import load_site_frame, split_frame\n"
        "from bess_dispatch.evaluation.benchmark import (\n"
        "    ablation_table,\n"
        "    fit_forecasters,\n"
        "    run_benchmark,\n"
        "    summarise_benchmark,\n"
        ")\n"
        "from bess_dispatch.forecasting.interface import (\n"
        "    ForecastResult,\n"
        "    actuals_for,\n"
        "    forecast_horizon,\n"
        ")\n"
        "from bess_dispatch.optimization.builder import build_dispatch_model\n"
        "from bess_dispatch.optimization.solve import solve_dispatch\n"
        "from bess_dispatch.visualization import plots\n"
        "\n"
        "config = load_config()\n"
        "site = config.site()\n"
        "frame = load_site_frame()\n"
        "print(f'{site.battery.energy_capacity_mwh} MWh / "
        "{site.battery.p_charge_max_mw} MW, '\n"
        "      f'round-trip {site.battery.round_trip_efficiency:.3f}')",
    ),
    (
        "markdown",
        "## Fit the forecasters\n"
        "\n"
        "One model per series, each chosen by validation MAE rather than by\n"
        "reputation. The choice is not uniform — gradient boosting wins on load\n"
        "and **loses to a naive baseline on PV**. See `docs/forecasting.md`.",
    ),
    (
        "code",
        "fitted = fit_forecasters(split_frame(frame, 'train'), horizon=24)\n"
        "for target, model in fitted.items():\n"
        "    print(f'{target:<16} {model.name}')",
    ),
    (
        "markdown",
        "## One day\n"
        "\n"
        "Forecast the next 24 hours, solve against the forecast, and pull the\n"
        "actuals for the same window so the plan can be scored on reality.",
    ),
    (
        "code",
        "issue = pd.Timestamp('2020-01-16', tz='UTC')\n"
        "forecast = forecast_horizon(fitted, frame, issue, 24)\n"
        "truth = actuals_for(frame, forecast)\n"
        "\n"
        "planned = solve_dispatch(build_dispatch_model(forecast, site, "
        "objective=config.objective))\n"
        "print(planned.summary())\n"
        "print(planned.cost_breakdown())",
    ),
    (
        "code",
        "plots.plot_dispatch_day(\n"
        "    planned, truth,\n"
        "    timestamps=pd.DatetimeIndex(forecast.timestamps),\n"
        "    title='Forecast-driven dispatch, 2020-01-16',\n"
        ");",
    ),
    (
        "markdown",
        "## Perfect foresight, through the identical code\n"
        "\n"
        "`ForecastResult.from_actuals` wraps observed data in the same type a\n"
        "model produces, so it goes through the same builder and the same\n"
        "solver. That is what makes the comparison a measurement of\n"
        "*information* rather than of two implementations.",
    ),
    (
        "code",
        "perfect = solve_dispatch(\n"
        "    build_dispatch_model(\n"
        "        ForecastResult.from_actuals(truth), site, objective=config.objective\n"
        "    )\n"
        ")\n"
        "print(f'forecast-driven {planned.total_cost_eur:>10,.2f} EUR')\n"
        "print(f'perfect         {perfect.total_cost_eur:>10,.2f} EUR')\n"
        "gap = planned.total_cost_eur - perfect.total_cost_eur\n"
        "print(f'cost of not knowing {gap:>10,.2f} EUR')",
    ),
    (
        "code",
        "plots.plot_forecast_vs_perfect(\n"
        "    planned, perfect, truth, forecast,\n"
        "    timestamps=pd.DatetimeIndex(forecast.timestamps),\n"
        ");",
    ),
    (
        "markdown",
        "The forecast under-called the true morning peak and over-called the\n"
        "afternoon, so the two controllers discharge nine hours apart. **Both\n"
        "schedules are optimal for the prices they were given** — a discharge in\n"
        "the wrong hour is not the optimizer failing.",
    ),
    (
        "markdown",
        "## Sixty days, seven arms\n"
        "\n"
        "Every arm solved over the same days and priced against the same\n"
        "actuals by the same function. Only what the controller knew varies.\n"
        "\n"
        "The three `+ actual` arms are the ablation: each hands the optimizer\n"
        "one series' true values and leaves the other two forecast.",
    ),
    (
        "code",
        "results = run_benchmark(site, objective=config.objective, frame=frame, "
        "fitted=fitted)\n"
        "summary = summarise_benchmark(results)\n"
        "summary[['total_cost_eur', 'saving_eur', 'saving_pct_wholesale', "
        "'value_captured_pct']].round(3)",
    ),
    ("code", "plots.plot_arm_comparison(summary);"),
    (
        "markdown",
        "Two things worth pausing on.\n"
        "\n"
        "**The rule-based controller is nearly worthless here.** Greedy\n"
        "self-consumption can only charge from surplus PV, and in a German\n"
        "January there essentially never is any. That is the controller that\n"
        "ships in the inverter, and it is the comparison a buyer actually cares\n"
        "about — not the no-battery case.\n"
        "\n"
        "**Savings are quoted against two denominators.** The battery can only\n"
        "move the wholesale energy component; network charges and levies ride\n"
        "along with every MWh whenever it is taken.",
    ),
    (
        "markdown",
        "## Which forecast does the money depend on?",
    ),
    (
        "code",
        "ablation = ablation_table(summary)\n"
        "print(f'total gap to perfect foresight: "
        "{ablation.attrs[\"total_gap_eur\"]:,.2f} EUR')\n"
        "ablation.round(2)",
    ),
    ("code", "plots.plot_ablation(ablation);"),
    (
        "markdown",
        "**Price error is nearly the whole story.** Making the price forecast\n"
        "perfect closes 98.7% of the gap; load and PV together account for\n"
        "2.5%.\n"
        "\n"
        "This is the result that changes what you would do next. Effort spent\n"
        "improving load or PV accuracy would buy almost nothing on this site;\n"
        "probabilistic *price* forecasting feeding a risk-aware dispatch is\n"
        "where the remaining value is.\n"
        "\n"
        "It also puts the load-data caveat in proportion. Load here is a\n"
        "national aggregate rescaled to one site, and therefore smoother and\n"
        "easier to forecast than a real building — but since load accuracy is\n"
        "worth 0.8% of the gap, even a far worse load forecaster would barely\n"
        "move the answer.",
    ),
    (
        "markdown",
        "## Where this goes next\n"
        "\n"
        "- `python -m bess_dispatch.run --stage rolling` — the hourly\n"
        "  receding-horizon controller, which recovers most of what perfect\n"
        "  information is worth in the day-ahead framing.\n"
        "- `python -m bess_dispatch.run --stage scenarios` — sizing,\n"
        "  efficiency, degradation and volatility sweeps, including the regime\n"
        "  where the forecast-driven controller **loses money**.\n"
        "- `streamlit run app/streamlit_app.py` — the same functions behind\n"
        "  sliders.",
    ),
]


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "markdown" else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = HERE / "02_walkthrough.ipynb"
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(f"wrote {build()}")
