"""Figures build, are non-trivial, and do not leak axes.

These tests cannot judge whether a chart is *readable* — every defect found in
this project's figures was found by rendering them and looking. What they can do
is stop a figure silently becoming blank, and stop a long run exhausting memory
because nothing closes its figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bess_dispatch.forecasting.interface import ForecastResult
from bess_dispatch.optimization.builder import build_dispatch_model
from bess_dispatch.optimization.solve import solve_dispatch
from bess_dispatch.visualization import eda, plots


@pytest.fixture
def solved(perfect_forecast, site):
    return solve_dispatch(build_dispatch_model(perfect_forecast, site))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("name", sorted(eda.EDA_FIGURES))
def test_eda_figures_build_and_draw_something(synthetic_year, name):
    figure = eda.EDA_FIGURES[name](synthetic_year)
    assert figure.get_axes(), f"{name} produced no axes"
    # A blank figure has axes but nothing on them.
    drawn = sum(
        len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.images)
        for ax in figure.get_axes()
    )
    assert drawn > 0, f"{name} drew nothing"


def test_dispatch_day_builds(solved, one_day):
    figure = plots.plot_dispatch_day(
        solved, one_day, timestamps=pd.DatetimeIndex(one_day.timestamps)
    )
    assert len(figure.get_axes()) >= 2  # power + soc, plus the price twin


def test_forecast_vs_perfect_builds(solved, one_day, perfect_forecast, site):
    noisy = ForecastResult(
        timestamps=one_day.timestamps,
        load_forecast_mw=one_day.load_mw,
        pv_forecast_mw=one_day.pv_mw,
        price_forecast_eur_mwh=one_day.price_eur_mwh * 0.9,
    )
    other = solve_dispatch(build_dispatch_model(noisy, site))
    figure = plots.plot_forecast_vs_perfect(other, solved, one_day, noisy)
    assert len(figure.get_axes()) == 2


def test_arm_comparison_excludes_the_baseline():
    summary = pd.DataFrame(
        {"saving_eur": [0.0, 10.0, 25.0]},
        index=pd.Index(["no battery", "forecast", "perfect foresight"], name="arm"),
    )
    figure = plots.plot_arm_comparison(summary)
    labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
    assert "no battery" not in labels


def test_ablation_figure_builds():
    ablation = pd.DataFrame(
        {"gap_closed_pct": [98.7, 0.8, 1.7], "gap_closed_eur": [330.0, 3.0, 6.0]},
        index=pd.Index(["price", "load", "pv"], name="series made perfect"),
    )
    ablation.attrs["total_gap_eur"] = 337.0
    figure = plots.plot_ablation(ablation)
    assert figure.axes[0].get_xlim()[1] >= 100


def test_sizing_curve_builds():
    curve = pd.DataFrame(
        {
            "total_cost_eur": [1.0, 2.0],
            "saving_eur": [300.0, 570.0],
            "saving_pct": [0.32, 0.60],
            "saving_pct_wholesale": [0.96, 1.80],
        },
        index=pd.Index(["size 0.5 MWh / 0.25 MW", "size 1.0 MWh / 0.5 MW"], name="variant"),
    )
    figure = plots.plot_sizing_curve(curve)
    assert len(figure.axes[0].lines) == 2


def test_forecast_error_by_hour_builds():
    by_hour = pd.DataFrame(
        {
            "MAE": np.linspace(1, 10, 24),
            "bias": np.linspace(-1, 1, 24),
            "n": np.full(24, 60),
        },
        index=pd.Index(range(24), name="hour"),
    )
    figure = plots.plot_forecast_error_by_hour(by_hour, "price")
    assert figure.axes[0].get_title(loc="left")


def test_render_all_writes_and_closes(tmp_path, solved, one_day):
    figures = {
        "a": plots.plot_dispatch_day(solved, one_day),
        "b": plots.plot_dispatch_day(solved, one_day),
    }
    written = plots.render_all(figures, tmp_path)
    assert len(written) == 2
    assert all(path.exists() and path.stat().st_size > 1000 for path in written)
    # Every figure must be closed, or a long sweep leaks them all.
    assert not plt.get_fignums()


def test_eda_render_all_writes_every_figure(tmp_path, synthetic_year):
    written = eda.render_all(synthetic_year, tmp_path)
    assert len(written) == len(eda.EDA_FIGURES)
    assert all(path.stat().st_size > 1000 for path in written)
