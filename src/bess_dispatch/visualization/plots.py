"""Result figures: what the controller did, and what it was worth.

Data figures live in `eda.py`. Everything here needs a solved schedule.

As in `eda.py`, the global matplotlib backend is deliberately not set: a library
that pins Agg at import makes every figure in a notebook render as the text
"<Figure size ...>". matplotlib falls back to Agg by itself when no display
exists, which covers CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_LOAD = "#1f4e79"
COLOR_PV = "#e8a33d"
COLOR_PRICE = "#a83232"
COLOR_CHARGE = "#4a7fb5"
COLOR_DISCHARGE = "#3d7a5a"
COLOR_SOC = "#6b4c8a"
COLOR_IMPORT = "#8a8f94"
COLOR_MUTED = "#9aa0a6"


def plot_dispatch_day(
    result,
    actuals,
    timestamps=None,
    title: str | None = None,
) -> plt.Figure:
    """One day: battery power against price, with state of charge beneath.

    The two panels share an x-axis because the whole claim is that the battery
    charges when the price is low and discharges when it is high. Split across
    two figures, a reader has to take that on trust.
    """
    periods = np.arange(len(result.charge_mw))
    hours = (
        pd.DatetimeIndex(timestamps).hour if timestamps is not None else periods
    )

    fig, (ax_power, ax_soc) = plt.subplots(
        2,
        1,
        figsize=(10, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.12},
    )

    ax_power.bar(
        periods, result.discharge_mw, color=COLOR_DISCHARGE, label="Discharge", width=0.8
    )
    ax_power.bar(
        periods, -result.charge_mw, color=COLOR_CHARGE, label="Charge", width=0.8
    )
    ax_power.plot(
        periods, actuals.load_mw, color=COLOR_LOAD, lw=1.8, label="Load", zorder=4
    )
    ax_power.plot(
        periods, actuals.pv_mw, color=COLOR_PV, lw=1.8, label="PV", zorder=4
    )
    ax_power.axhline(0, color="black", lw=0.8)
    ax_power.set_ylabel("Power (MW)")
    ax_power.grid(alpha=0.25)

    ax_price = ax_power.twinx()
    ax_price.plot(
        periods,
        actuals.price_eur_mwh,
        color=COLOR_PRICE,
        lw=2,
        ls="--",
        label="Wholesale price",
        zorder=5,
    )
    ax_price.set_ylabel("Wholesale price (EUR/MWh)", color=COLOR_PRICE)
    ax_price.tick_params(axis="y", colors=COLOR_PRICE)

    handles, labels = ax_power.get_legend_handles_labels()
    ph, pl = ax_price.get_legend_handles_labels()
    ax_power.legend(
        handles + ph, labels + pl, loc="upper left", ncol=3, frameon=False, fontsize=9
    )
    ax_power.set_title(
        title or "Dispatch against price — charge below the axis, discharge above",
        loc="left",
        fontsize=11,
    )

    ax_soc.fill_between(periods, result.soc_mwh, color=COLOR_SOC, alpha=0.25, lw=0)
    ax_soc.plot(periods, result.soc_mwh, color=COLOR_SOC, lw=2)
    ax_soc.set_ylabel("State of\ncharge (MWh)")
    ax_soc.set_xlabel("Hour of day (UTC)")
    ax_soc.grid(alpha=0.25)
    ax_soc.set_xticks(periods[::3])
    ax_soc.set_xticklabels([str(h) for h in np.asarray(hours)[::3]])

    fig.tight_layout()
    return fig


def plot_forecast_vs_perfect(
    forecast_result,
    perfect_result,
    actuals,
    forecast,
    timestamps=None,
    title: str | None = None,
) -> plt.Figure:
    """Why forecast error costs money, on a single day.

    The upper panel shows the forecast price against the price that actually
    happened; the lower panel shows what each controller did about it. Reading
    the two together is the point — a discharge in the wrong hour is not a bug
    in the optimizer, it is the optimizer being right about the wrong prices.
    """
    periods = np.arange(len(actuals.price_eur_mwh))
    hours = pd.DatetimeIndex(timestamps).hour if timestamps is not None else periods

    fig, (ax_price, ax_power) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.14},
    )

    ax_price.plot(
        periods, actuals.price_eur_mwh, color=COLOR_PRICE, lw=2.2, label="Actual price"
    )
    ax_price.plot(
        periods,
        forecast.price_forecast_eur_mwh,
        color=COLOR_MUTED,
        lw=2,
        ls="--",
        label="Forecast price",
    )
    ax_price.fill_between(
        periods,
        actuals.price_eur_mwh,
        forecast.price_forecast_eur_mwh,
        color=COLOR_PRICE,
        alpha=0.12,
        lw=0,
    )
    peak = int(np.argmax(actuals.price_eur_mwh))
    miss = actuals.price_eur_mwh[peak] - forecast.price_forecast_eur_mwh[peak]
    ax_price.annotate(
        f"true peak missed by\n{miss:.1f} EUR/MWh",
        xy=(peak, actuals.price_eur_mwh[peak]),
        # Down and to the right: the peak sits at the top of the axes, so an
        # annotation placed above it collides with the title.
        xytext=(26, -34),
        textcoords="offset points",
        va="top",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "lw": 1},
    )
    ax_price.set_ylabel("Price (EUR/MWh)")
    ax_price.grid(alpha=0.25)
    ax_price.legend(loc="lower right", frameon=False, fontsize=9)
    ax_price.set_title(
        title or "Forecast error moves the discharge to the wrong hour",
        loc="left",
        fontsize=11,
    )

    width = 0.4
    ax_power.bar(
        periods - width / 2,
        perfect_result.discharge_mw - perfect_result.charge_mw,
        width=width,
        color=COLOR_DISCHARGE,
        label="Perfect foresight",
    )
    ax_power.bar(
        periods + width / 2,
        forecast_result.discharge_mw - forecast_result.charge_mw,
        width=width,
        color=COLOR_LOAD,
        label="Forecast-driven",
    )
    ax_power.axhline(0, color="black", lw=0.8)
    ax_power.set_ylabel("Net battery power (MW)\ndischarge positive")
    ax_power.set_xlabel("Hour of day (UTC)")
    ax_power.grid(alpha=0.25)
    ax_power.legend(loc="upper left", frameon=False, fontsize=9)
    ax_power.set_xticks(periods[::3])
    ax_power.set_xticklabels([str(h) for h in np.asarray(hours)[::3]])

    fig.tight_layout()
    return fig


def plot_arm_comparison(summary: pd.DataFrame, value: str = "saving_eur") -> plt.Figure:
    """Saving by arm, ordered as the arms were defined rather than by size.

    Definition order tells a story — floor, naive controller, real system,
    ablations, ceiling — that sorting by magnitude destroys.
    """
    working = summary[summary.index != "no battery"]
    labels = list(working.index)
    values = working[value].to_numpy()

    colors = []
    for label in labels:
        if label == "perfect foresight":
            colors.append(COLOR_DISCHARGE)
        elif label.startswith("forecast + actual"):
            colors.append(COLOR_MUTED)
        elif label == "forecast":
            colors.append(COLOR_LOAD)
        else:
            colors.append(COLOR_IMPORT)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Saving against no battery (EUR over the window)")
    ax.grid(alpha=0.25, axis="x")
    ax.axvline(0, color="black", lw=0.9)

    span = max(abs(values.max()), abs(values.min()), 1.0)
    for bar, val in zip(bars, values, strict=True):
        offset = span * 0.015
        ax.text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )
    ax.set_xlim(min(0, values.min()) - span * 0.12, values.max() + span * 0.16)
    ax.set_title(
        "What each controller was worth", loc="left", fontsize=11
    )
    fig.tight_layout()
    return fig


def plot_ablation(ablation: pd.DataFrame) -> plt.Figure:
    """Share of the forecast-driven shortfall attributable to each series."""
    labels = list(ablation.index)
    shares = ablation["gap_closed_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.barh(labels, shares, color=[COLOR_PRICE, COLOR_LOAD, COLOR_PV])
    ax.invert_yaxis()
    ax.set_xlabel("Share of the gap to perfect foresight that closes (%)")
    ax.set_xlim(0, max(105, shares.max() * 1.15))
    ax.grid(alpha=0.25, axis="x")
    for bar, share in zip(bars, shares, strict=True):
        ax.text(
            share + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1f}%",
            va="center",
            fontsize=9,
        )
    gap = ablation.attrs.get("total_gap_eur")
    suffix = f" (total gap {gap:,.0f} EUR)" if gap else ""
    ax.set_title(
        f"Making one forecast perfect{suffix}\nPrice error is nearly the whole story",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_sizing_curve(curve: pd.DataFrame) -> plt.Figure:
    """Saving against battery size, on both denominators."""
    labels = [label.replace("size ", "") for label in curve.index]
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(
        positions,
        curve["saving_pct_wholesale"],
        marker="o",
        color=COLOR_DISCHARGE,
        lw=2,
        label="of wholesale energy cost",
    )
    ax.plot(
        positions,
        curve["saving_pct"],
        marker="s",
        color=COLOR_MUTED,
        lw=2,
        ls="--",
        label="of the full delivered bill",
    )
    for x, y in zip(positions, curve["saving_pct_wholesale"], strict=True):
        ax.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Saving (%)")
    ax.set_xlabel("Battery size")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, max(curve["saving_pct_wholesale"]) * 1.28)
    ax.set_title(
        "Value scales with sizing — and depends which bill you divide by",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_forecast_error_by_hour(by_hour: pd.DataFrame, target: str) -> plt.Figure:
    """MAE and bias by local hour, for one series."""
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(by_hour.index, by_hour["MAE"], color=COLOR_LOAD, alpha=0.75, label="MAE")
    ax.plot(
        by_hour.index,
        by_hour["bias"],
        color=COLOR_PRICE,
        lw=2,
        marker="o",
        ms=3.5,
        label="bias (signed)",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Error")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Forecast error by hour — {target}", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_rolling_trace(rolling, hours: int = 168) -> plt.Figure:
    """The first `hours` of a rolling run: what the controller actually executed."""
    n = min(hours, len(rolling.timestamps))
    index = rolling.timestamps[:n]

    fig, (ax_power, ax_soc) = plt.subplots(
        2, 1, figsize=(11, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12},
    )
    ax_power.fill_between(
        index, 0, rolling.discharge_mw[:n], color=COLOR_DISCHARGE, alpha=0.85,
        label="Discharge", lw=0,
    )
    ax_power.fill_between(
        index, 0, -rolling.charge_mw[:n], color=COLOR_CHARGE, alpha=0.85,
        label="Charge", lw=0,
    )
    ax_power.plot(
        index, rolling.grid_import_mw[:n], color=COLOR_IMPORT, lw=1.2,
        label="Grid import",
    )
    ax_power.axhline(0, color="black", lw=0.8)
    ax_power.set_ylabel("Power (MW)")
    ax_power.grid(alpha=0.25)
    ax_power.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    ax_power.set_title(
        f"Rolling controller, first {n} hours executed "
        f"({rolling.n_solves} solves over the full run)",
        loc="left",
        fontsize=11,
    )

    ax_soc.plot(index, rolling.soc_mwh[:n], color=COLOR_SOC, lw=1.8)
    ax_soc.fill_between(index, rolling.soc_mwh[:n], color=COLOR_SOC, alpha=0.2, lw=0)
    ax_soc.set_ylabel("State of\ncharge (MWh)")
    ax_soc.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def render_all(figures: dict[str, plt.Figure], out_dir, dpi: int = 130) -> list[Path]:
    """Save a dict of named figures, closing each as it goes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in figures.items():
        path = out_dir / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written
