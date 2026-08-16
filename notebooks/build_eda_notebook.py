"""Generate notebooks/01_eda.ipynb.

The notebook is a narrative over `bess_dispatch.visualization.eda`; it holds no
analysis logic of its own, per the brief's "keep business logic out of
notebooks". Writing it from a script keeps it diffable and means the prose and
the code cannot drift apart in a merge.

    python notebooks/build_eda_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# 1 — Exploratory analysis\n"
        "\n"
        "What this notebook is for: deciding whether a battery on this site has "
        "anything worth optimizing, **before** any model is built.\n"
        "\n"
        "Every figure answers a stated question. The plotting code lives in "
        "`bess_dispatch.visualization.eda` so it is testable and reusable; this "
        "notebook only narrates.\n"
        "\n"
        "Data provenance, units, splits and the rescaling caveat are in "
        "`data/DATA_DICTIONARY.md`. The short version: real DE/LU wholesale "
        "prices, and national load and solar shapes rescaled to a hypothetical "
        "1 MW / 0.8 MWp site.",
    ),
    (
        "code",
        "%matplotlib inline\n"
        "\n"
        "from bess_dispatch.data.loaders import (\n"
        "    daily_price_spread,\n"
        "    describe,\n"
        "    load_site_frame,\n"
        "    train_test_report,\n"
        ")\n"
        "from bess_dispatch.visualization import eda\n"
        "\n"
        "frame = load_site_frame()\n"
        "print(f'{len(frame):,} hourly rows, {frame.index[0]} to {frame.index[-1]}')",
    ),
    (
        "markdown",
        "## The splits\n"
        "\n"
        "Chronological, never random — a random split over a time series leaks "
        "the future into the past.\n"
        "\n"
        "The test window stops before March 2020 on purpose. European load and "
        "prices break structurally that month, and folding a once-in-a-decade "
        "shock into the headline would confound forecast quality with COVID. "
        "The `shift` split holds that period back to be looked at on its own.",
    ),
    ("code", "train_test_report().round(2)"),
    (
        "markdown",
        "## Summary statistics\n"
        "\n"
        "Note the price row: 484 negative hours, a minimum of −90 EUR/MWh and a "
        "maximum of 200. **No outliers are removed anywhere in this project.** "
        "Those extremes are real market events, and negative prices are "
        "precisely when a battery earns by charging — discarding them would "
        "delete the phenomenon under study.",
    ),
    ("code", "describe(frame).round(2)"),
    (
        "markdown",
        "## Q1 — Is demand in phase with expensive hours?\n"
        "\n"
        "If load, price and PV all peaked together there would be little for a "
        "battery to do beyond self-consumption.",
    ),
    # Trailing semicolon: with %matplotlib inline the figure is auto-displayed,
    # so returning it as well renders every chart twice.
    ("code", "eda.plot_daily_profiles(frame);"),
    (
        "markdown",
        "They are not in phase, and that is the whole opportunity. Price is "
        "twin-peaked — morning and evening — with a **midday trough that PV "
        "helps dig**. Load is comparatively flat. So the battery has a standing "
        "invitation: charge into the cheap midday hours, discharge into the "
        "evening peak.",
    ),
    (
        "markdown",
        "## Q2 — How much is one perfect cycle worth?\n"
        "\n"
        "The within-day price spread is an upper bound on arbitrage revenue per "
        "MWh cycled, before round-trip losses. If the median day's spread were "
        "small relative to those losses, the battery would be better off idle.",
    ),
    ("code", "eda.plot_price_spread(frame);"),
    (
        "code",
        "spread = daily_price_spread(frame)\n"
        "rte = 0.95 * 0.95\n"
        "print(f'median daily spread   {spread.median():.1f} EUR/MWh')\n"
        "print(f'90th percentile       {spread.quantile(0.9):.1f} EUR/MWh')\n"
        "print(f'largest               {spread.max():.1f} EUR/MWh')\n"
        "print()\n"
        "print(f'round-trip efficiency {rte:.3f}: charging at price p and')\n"
        "print('discharging at price q nets q*eta_d - p/eta_c per MWh stored,')\n"
        "print('so the spread has to clear the loss before a cycle pays.')",
    ),
    (
        "markdown",
        "The median day offers a spread around 29 EUR/MWh. That is comfortably "
        "above what round-trip losses cost at these price levels, so arbitrage "
        "is worth doing — but it is not enormous, and it sets expectations for "
        "how large the savings can be. The distribution has a long right tail: "
        "a handful of days offer more than 100 EUR/MWh, and those days will "
        "dominate the annual result.",
    ),
    (
        "markdown",
        "## Q3 — Does PV push the site into export, and how peaky is import?\n"
        "\n"
        "The duration curve answers both. Where it crosses zero is how many "
        "hours the site exports with no storage at all; its left-hand end is "
        "what a demand charge would be levied on.",
    ),
    ("code", "eda.plot_net_load_duration(frame);"),
    (
        "markdown",
        "The site exports in only 117 hours out of 17,531 — **0.7% of the "
        "time**. This matters for interpreting later results: export "
        "compensation is nearly irrelevant on this site, so the battery's value "
        "has to come from shifting *when* energy is imported, not from selling "
        "surplus. A sunnier site or a smaller load would tell a different "
        "story.",
    ),
    (
        "markdown",
        "## Q4 — What intraday shape does the battery actually trade?\n"
        "\n"
        "A raw hour-by-month heatmap of price would mislead here, and it is "
        "worth being explicit about why. The record runs 2018-10 to 2020-09, so "
        "months 10–12 come only from 2018 and 2019 while months 1–9 mix 2019 "
        "and 2020 — and 2019 averaged 37.7 EUR/MWh against 2020's 27.7. A raw "
        "heatmap shows that level shift between years and invites reading it as "
        "a seasonal effect.\n"
        "\n"
        "Subtracting each day's own mean removes the level and leaves the "
        "within-day shape, which is also the only thing a daily-cycling battery "
        "can monetise.",
    ),
    ("code", "eda.plot_price_by_hour_and_month(frame);"),
    (
        "markdown",
        "The midday dip is **seasonal, not year-round**. From March to "
        "September midday sits well below the daily mean; from November to "
        "January it sits *above* it, and only the overnight trough is left to "
        "charge into. The evening peak around 18:00–20:00 is the one feature "
        "present all year.\n"
        "\n"
        "This is why the feature set includes month-of-year: a forecaster "
        "without it cannot represent a signal that inverts between seasons.",
    ),
    (
        "markdown",
        "## Q5 — Is price just a function of load and PV?\n"
        "\n"
        "If it were, forecasting price would reduce to forecasting the other "
        "two, and it would not need its own model.",
    ),
    ("code", "eda.plot_correlations(frame);"),
    (
        "markdown",
        "It is not. Price correlates with net load at only 0.51 — a real "
        "relationship, but far from determinative. Price carries information "
        "from fuel costs, cross-border flows, outages and wind that this site "
        "cannot see at all.\n"
        "\n"
        "That has a direct consequence for the rest of the project: **price is "
        "the hardest of the three series to forecast, and it is also the one "
        "the dispatch decision is most sensitive to.** The ablation in the "
        "evaluation phase tests exactly that claim rather than assuming it.",
    ),
    (
        "markdown",
        "## What this establishes\n"
        "\n"
        "1. Price and load are out of phase, so there is arbitrage to capture "
        "beyond self-consumption.\n"
        "2. The median day offers roughly 29 EUR/MWh of spread — worth "
        "capturing, but modest, so expect single-digit percentage savings from "
        "a modestly-sized battery.\n"
        "3. The site almost never exports, so value must come from shifting "
        "import timing.\n"
        "4. The tradeable intraday shape inverts between summer and winter.\n"
        "5. Price is only loosely tied to load and PV, so it needs its own "
        "model and will likely be the binding constraint on performance.\n"
        "\n"
        "Next: `bess_dispatch.forecasting` builds models for all three series, "
        "with a leakage test that is the single most important test in the "
        "repository.",
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
    path = HERE / "01_eda.ipynb"
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(f"wrote {build()}")
