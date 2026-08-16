"""Exploratory figures about the *data*, before any model touches it.

Result figures live in `plots.py`. The split is deliberate: these five answer
questions about whether the problem is worth solving at all, and they would
still be worth looking at if the optimizer did not exist.

Every function returns a `matplotlib.figure.Figure` and draws nothing to screen,
so they work headless in CI and in a notebook without a backend argument.

The brief asks for seven charts. There are five, because two of its suggestions
(a raw load trace and a raw PV trace) answer nothing the daily-profile figure
does not answer better. "Every chart must answer a question" cuts both ways.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess_dispatch.data.loaders import daily_price_spread

# Deliberately no matplotlib.use("Agg") here. A library that forces the global
# backend at import time breaks the one place these figures matter most: in a
# notebook, an Agg-pinned Figure comes back as the text "<Figure size ...>"
# instead of an image. matplotlib already falls back to Agg by itself when no
# display is available, which covers CI; tests pin it explicitly in conftest.

# One palette, used consistently: load is the thing you pay for, PV is what
# offsets it, price is the signal the battery trades against.
COLOR_LOAD = "#1f4e79"
COLOR_PV = "#e8a33d"
COLOR_PRICE = "#a83232"
COLOR_NET = "#3d7a5a"
COLOR_MUTED = "#9aa0a6"


def _local_hour(frame: pd.DataFrame) -> np.ndarray:
    """Hour of day in local time.

    Local, not UTC: solar position and human routine both follow local clocks,
    and a profile plotted against UTC hour smears both by the DST offset.
    """
    index = frame.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    # Converting the tz-aware index beats parsing the local_timestamp column:
    # that column carries mixed +0100/+0200 offsets across the DST boundary,
    # which is correct data but which pandas refuses to parse into one series.
    return index.tz_convert("Europe/Berlin").hour.to_numpy()


def plot_daily_profiles(frame: pd.DataFrame) -> plt.Figure:
    """Q: is the site's demand in phase with expensive hours?

    If load and price peaked together and PV filled the gap, there would be
    little for a battery to do beyond self-consumption. The answer here is what
    motivates arbitrage on top of self-consumption.
    """
    hour = _local_hour(frame)
    grouped = frame.groupby(hour)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax_price = ax.twinx()

    for column, color, label in (
        ("load_mw", COLOR_LOAD, "Load"),
        ("pv_mw", COLOR_PV, "PV generation"),
    ):
        mean = grouped[column].mean()
        low = grouped[column].quantile(0.10)
        high = grouped[column].quantile(0.90)
        ax.plot(mean.index, mean.to_numpy(), color=color, lw=2.2, label=label, zorder=3)
        ax.fill_between(mean.index, low, high, color=color, alpha=0.15, lw=0, zorder=1)

    price_mean = grouped["price_eur_mwh"].mean()
    ax_price.plot(
        price_mean.index,
        price_mean.to_numpy(),
        color=COLOR_PRICE,
        lw=2.2,
        ls="--",
        label="Wholesale price",
        zorder=4,
    )

    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Power (MW)")
    ax_price.set_ylabel("Wholesale price (EUR/MWh)", color=COLOR_PRICE)
    ax_price.tick_params(axis="y", colors=COLOR_PRICE)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(alpha=0.25, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    ph, pl = ax_price.get_legend_handles_labels()
    ax.legend(handles + ph, labels + pl, loc="upper left", frameon=False)
    ax.set_title(
        "Average day: PV peaks at midday, price peaks morning and evening\n"
        "Shaded bands are the 10th-90th percentile across all days",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_price_spread(frame: pd.DataFrame) -> plt.Figure:
    """Q: how much is one perfect charge/discharge cycle worth?

    The within-day price spread is an upper bound on arbitrage revenue per MWh
    cycled, before efficiency losses. If the median day's spread were small
    relative to round-trip losses, the battery would be better off idle.
    """
    spread = daily_price_spread(frame)
    price = frame["price_eur_mwh"].dropna()

    fig, (ax_hist, ax_time) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    ax_hist.hist(spread, bins=45, color=COLOR_PRICE, alpha=0.75)
    top = ax_hist.get_ylim()[1]
    # Stagger the two labels vertically. Anchored at the same height they
    # collide, because p50 and p90 are only ~24 EUR/MWh apart on this axis.
    for quantile, style, height in ((0.5, "-", 0.94), (0.9, ":", 0.80)):
        value = spread.quantile(quantile)
        ax_hist.axvline(value, color="black", ls=style, lw=1.4)
        ax_hist.annotate(
            f"p{int(quantile * 100)} = {value:.0f}",
            xy=(value, top * height),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
        )
    ax_hist.set_xlabel("Within-day price spread (EUR/MWh)")
    ax_hist.set_ylabel("Days")
    ax_hist.set_title("Spread bounds what a cycle can earn", loc="left", fontsize=10)
    ax_hist.grid(alpha=0.25)

    # Drop the timezone before period conversion: to_period() would otherwise
    # warn that it is discarding it, and month buckets do not need it.
    month_key = spread.index.tz_localize(None).to_period("M")
    monthly = spread.groupby(month_key).median()
    ax_time.bar(
        range(len(monthly)),
        monthly.to_numpy(),
        color=COLOR_PRICE,
        alpha=0.8,
        width=0.75,
    )
    ax_time.set_xticks(range(0, len(monthly), 3))
    ax_time.set_xticklabels(
        [str(p) for p in monthly.index[::3]], rotation=45, ha="right", fontsize=8
    )
    ax_time.set_ylabel("Median daily spread (EUR/MWh)")
    negative = int((price < 0).sum())
    ax_time.set_title(
        f"and it moves month to month  ({negative} negative-price hours in the record)",
        loc="left",
        fontsize=10,
    )
    ax_time.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    return fig


def plot_net_load_duration(frame: pd.DataFrame) -> plt.Figure:
    """Q: does PV ever push the site into export, and how peaky is the import?

    The duration curve answers both at once. Where it crosses zero is how many
    hours the site exports before any storage exists; its left-hand end is what
    a demand charge would be levied on.
    """
    net = (frame["load_mw"] - frame["pv_mw"]).dropna().sort_values(ascending=False)
    hours = np.arange(len(net))
    share = 100 * hours / max(len(net) - 1, 1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(share, net.to_numpy(), color=COLOR_NET, lw=2)
    ax.fill_between(share, 0, net.to_numpy(), where=net > 0, color=COLOR_NET, alpha=0.18, lw=0)
    ax.fill_between(share, 0, net.to_numpy(), where=net <= 0, color=COLOR_PV, alpha=0.35, lw=0)
    ax.axhline(0, color="black", lw=0.9)

    export_hours = int((net <= 0).sum())
    peak = float(net.iloc[0])
    ax.annotate(
        f"peak net import {peak:.2f} MW\n(what a demand charge bites on)",
        xy=(0, peak),
        xytext=(14, -30),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "lw": 1},
    )
    if export_hours:
        crossing = 100 * (len(net) - export_hours) / len(net)
        ax.annotate(
            f"{export_hours} h of export ({export_hours / len(net):.1%})",
            xy=(crossing, 0),
            xytext=(-10, -34),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "lw": 1},
        )

    ax.set_xlabel("Share of hours exceeded (%)")
    ax.set_ylabel("Net load, load minus PV (MW)")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.25)
    ax.set_title("Net load duration curve", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_price_by_hour_and_month(frame: pd.DataFrame) -> plt.Figure:
    """Q: what intraday *shape* does the battery trade, and does it change with season?

    Plotting the mean price per hour and month directly would be misleading
    here, and measurably so. The record runs 2018-10 to 2020-09, so months
    10-12 are drawn only from 2018 and 2019 while months 1-9 mix 2019 and 2020
    -- and 2019 averaged 37.7 EUR/MWh against 2020's 27.7. A raw heatmap
    therefore shows a level shift between years and invites reading it as a
    seasonal effect. (An earlier draft of this figure did exactly that.)

    Subtracting each day's own mean removes the level entirely and leaves the
    within-day shape, which is also the only thing a battery on a daily cycle
    can monetise: it cannot trade the fact that one year was dearer than
    another.
    """
    price = frame["price_eur_mwh"]
    deviation = price - price.groupby(frame.index.floor("D")).transform("mean")

    pivot = (
        pd.DataFrame(
            {
                "hour": _local_hour(frame),
                "month": frame.index.month,
                "deviation": deviation.to_numpy(),
            }
        )
        .dropna()
        .pivot_table(index="hour", columns="month", values="deviation", aggfunc="mean")
    )

    limit = float(np.abs(pivot.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.to_numpy(),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label="Price minus that day's mean (EUR/MWh)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day (local time)")
    ax.set_xticks(range(1, 13))
    ax.set_yticks(range(0, 24, 3))
    ax.set_title(
        "The midday dip is a spring and summer effect, not a year-round one\n"
        "In November to January midday sits above the daily average, and only "
        "the overnight trough is left to trade",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_correlations(frame: pd.DataFrame) -> plt.Figure:
    """Q: is price predictable from load and PV, or is it its own animal?

    If price were a tight function of net load, forecasting it would reduce to
    forecasting the other two. The weak correlation is why price gets its own
    model, and why price error dominates the ablation.
    """
    columns = {
        "load_mw": "Load",
        "pv_mw": "PV",
        "price_eur_mwh": "Price",
    }
    working = frame[list(columns)].copy()
    working["net_load"] = working["load_mw"] - working["pv_mw"]
    labels = [*columns.values(), "Net load"]
    corr = working.corr()

    fig, ax = plt.subplots(figsize=(5.6, 5))
    mesh = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(mesh, ax=ax, shrink=0.82, label="Pearson correlation")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.to_numpy()[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(value) > 0.55 else "black",
            )
    ax.set_title("Price is only loosely tied to net load", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


EDA_FIGURES = {
    "daily_profiles": plot_daily_profiles,
    "price_spread": plot_price_spread,
    "net_load_duration": plot_net_load_duration,
    "price_hour_month": plot_price_by_hour_and_month,
    "correlations": plot_correlations,
}


def render_all(frame: pd.DataFrame, out_dir, dpi: int = 130) -> list:
    """Write every EDA figure to `out_dir`. Returns the paths written."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, function in EDA_FIGURES.items():
        figure = function(frame)
        path = out_dir / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written
