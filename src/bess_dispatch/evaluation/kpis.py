"""The brief's section 15 metrics, computed from a benchmark run.

One note on **renewable self-consumption**, because it is the metric most often
quoted and most often defined loosely. Here it is the share of generated PV that
the site used itself — directly, or via the battery — rather than exporting or
curtailing. On this site it starts near 100% before any battery exists, because
the site exports in only 0.7% of hours. So it is a nearly useless headline
number *for this site*, and saying so is more useful than reporting a number
that looks impressive and means nothing. It is kept because it would matter on a
sunnier site, and the code should not need changing to find that out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KPI_ORDER = (
    "total_cost_eur",
    "saving_eur",
    "saving_pct",
    "saving_pct_wholesale",
    "peak_import_mw",
    "peak_reduction_pct",
    "throughput_mwh",
    "equivalent_full_cycles",
    "mean_soc_mwh",
    "self_consumption_pct",
    "curtailed_mwh",
    "grid_import_mwh",
    "grid_export_mwh",
    "mean_solve_s",
)


def kpi_table(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    energy_capacity_mwh: float,
    pv_generated_mwh: float | None = None,
    baseline_arm: str = "no battery",
) -> pd.DataFrame:
    """Assemble the KPI table from a benchmark run and its summary."""
    grouped = results.groupby("arm")

    table = pd.DataFrame(index=summary.index)
    table["total_cost_eur"] = summary["total_cost_eur"]
    table["saving_eur"] = summary["saving_eur"]
    table["saving_pct"] = summary["saving_pct"]
    table["saving_pct_wholesale"] = summary["saving_pct_wholesale"]

    table["peak_import_mw"] = summary["peak_import_mw"]
    baseline_peak = summary.loc[baseline_arm, "peak_import_mw"]
    table["peak_reduction_pct"] = 100 * (baseline_peak - summary["peak_import_mw"]) / baseline_peak

    table["throughput_mwh"] = summary["throughput_mwh"]
    # Two half-cycles make one full cycle, so throughput is halved.
    table["equivalent_full_cycles"] = summary["throughput_mwh"] / (2 * energy_capacity_mwh)
    table["mean_soc_mwh"] = grouped["mean_soc_mwh"].mean()

    table["curtailed_mwh"] = summary["curtailed_mwh"]
    table["grid_import_mwh"] = grouped["grid_import_mwh"].sum()
    table["grid_export_mwh"] = grouped["grid_export_mwh"].sum()

    if pv_generated_mwh:
        used = pv_generated_mwh - table["grid_export_mwh"] - table["curtailed_mwh"]
        table["self_consumption_pct"] = 100 * used / pv_generated_mwh
    else:
        table["self_consumption_pct"] = np.nan

    table["mean_solve_s"] = summary["mean_solve_s"]
    return table[list(KPI_ORDER)]


def forecast_error_row(errors: dict[str, float]) -> pd.Series:
    """The brief asks for forecast errors in the KPI report; this formats them."""
    return pd.Series(errors, name="forecast MAE")


def describe_kpis() -> pd.DataFrame:
    """What each KPI means, so a table is readable without the source."""
    definitions = {
        "total_cost_eur": "Realised cost over the window, priced against actuals.",
        "saving_eur": "Cost reduction against the no-battery arm.",
        "saving_pct": "Saving as a share of the full delivered bill.",
        "saving_pct_wholesale": (
            "Saving as a share of the wholesale energy component only. The "
            "battery cannot touch network charges or levies, so this is the "
            "denominator it actually competes against."
        ),
        "peak_import_mw": "Highest grid import in any hour of the window.",
        "peak_reduction_pct": (
            "Peak cut against the no-battery arm. Can be negative: a "
            "cost-only objective happily raises the peak by charging into it."
        ),
        "throughput_mwh": "Total energy through the battery, charge plus discharge.",
        "equivalent_full_cycles": "Throughput expressed in full charge/discharge cycles.",
        "mean_soc_mwh": "Average state of charge.",
        "self_consumption_pct": (
            "Share of generated PV used on site rather than exported or "
            "curtailed. Near 100% here even with no battery, because this site "
            "exports in only 0.7% of hours."
        ),
        "curtailed_mwh": "PV generation spilled because it could not be used or exported.",
        "grid_import_mwh": "Total energy drawn from the grid.",
        "grid_export_mwh": "Total energy sent to the grid.",
        "mean_solve_s": "Mean solver wall time per horizon.",
    }
    return pd.DataFrame(
        {"definition": definitions}, index=pd.Index(KPI_ORDER, name="kpi")
    )
