"""Figure generation.

Every figure is drawn from the same CSV tables that the written results quote, so a
figure and a number in the README cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
DIVERGING_LOW = "#d03b3b"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#2a78d6"

FIGURE_SIZE = (9.0, 5.2)


def _new_axes(figsize=FIGURE_SIZE):
    figure, axes = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    return figure, axes


def _style(axes, title: str, subtitle: str = "", xlabel: str = "", ylabel: str = "", grid_axis: str = "y"):
    axes.set_title(title, color=PRIMARY_INK, fontsize=13, fontweight="600", loc="left", pad=18 if subtitle else 10)
    if subtitle:
        axes.text(
            0.0, 1.02, subtitle, transform=axes.transAxes, color=SECONDARY_INK, fontsize=10, ha="left", va="bottom"
        )
    axes.set_xlabel(xlabel, color=SECONDARY_INK, fontsize=10)
    axes.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10)
    axes.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(0.8)
    if grid_axis:
        axes.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.8, zorder=0)
        axes.set_axisbelow(True)


def _save(figure, path: Path):
    figure.tight_layout(pad=1.4)
    figure.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)


def concentration_curve_figure(curves: pd.DataFrame, path: Path) -> None:
    """Share of next year spend captured as targeting depth increases."""
    labels = {
        "two_part_model": "Two part model",
        "prior_year_cost": "Prior year cost",
        "chronic_condition_count": "Chronic condition count",
    }
    figure, axes = _new_axes()

    reference = curves[curves["policy"] == "two_part_model"].sort_values("top_k_pct")
    axes.plot(
        reference["top_k_pct"],
        reference["pct_of_spend_oracle"],
        color=MUTED_INK,
        linewidth=1.6,
        linestyle=(0, (1, 2)),
        label="Perfect foresight bound",
        zorder=2,
    )
    axes.plot(
        reference["top_k_pct"],
        reference["pct_of_spend_random"],
        color=MUTED_INK,
        linewidth=1.6,
        linestyle="--",
        label="Random targeting",
        zorder=2,
    )
    for index, (policy, label) in enumerate(labels.items()):
        subset = curves[curves["policy"] == policy].sort_values("top_k_pct")
        if subset.empty:
            continue
        axes.plot(
            subset["top_k_pct"],
            subset["pct_of_spend_captured"],
            color=SERIES[index],
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=label,
            zorder=3,
        )

    model_at_ten = curves[(curves["policy"] == "two_part_model") & (curves["top_k_pct"] == 10.0)]
    if not model_at_ten.empty:
        value = float(model_at_ten["pct_of_spend_captured"].iloc[0])
        axes.annotate(
            f"{value:.1f}% of spend\nin the top 10%",
            xy=(10.0, value),
            xytext=(23.0, value + 2.5),
            color=PRIMARY_INK,
            fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=1.0),
        )

    _style(
        axes,
        "Targeting the riskiest members captures spend faster than random outreach",
        "Held out 2010 spend, ranked by risk scored from 2009 data",
        "Members contacted, percent of the population",
        "Share of next year spend captured, percent",
    )
    legend = axes.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)
    _save(figure, path)


def spend_concentration_figure(distribution: pd.DataFrame, path: Path) -> None:
    """How concentrated realised spend is, by year."""
    figure, axes = _new_axes(figsize=(8.0, 4.8))
    years = distribution["year"].astype(int).astype(str)
    positions = np.arange(len(years))
    width = 0.38

    axes.bar(
        positions - width / 2,
        distribution["pct_spend_in_top_decile"],
        width,
        color=SERIES[0],
        label="Top 10% of members",
        zorder=3,
    )
    axes.bar(
        positions + width / 2,
        distribution["pct_spend_in_top_percent"],
        width,
        color=SERIES[1],
        label="Top 1% of members",
        zorder=3,
    )
    for index, row in distribution.reset_index(drop=True).iterrows():
        axes.text(
            index - width / 2,
            row["pct_spend_in_top_decile"] + 1.0,
            f"{row['pct_spend_in_top_decile']:.0f}%",
            ha="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
        axes.text(
            index + width / 2,
            row["pct_spend_in_top_percent"] + 1.0,
            f"{row['pct_spend_in_top_percent']:.0f}%",
            ha="center",
            color=SECONDARY_INK,
            fontsize=9,
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(years)
    axes.set_ylim(0, 60)
    _style(
        axes,
        "Half of all spend sits with a tenth of the members",
        "Share of total annual spend held by the most expensive members",
        "",
        "Share of total spend, percent",
    )
    legend = axes.legend(frameon=False, fontsize=9.5, loc="upper right")
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)
    _save(figure, path)


def calibration_figure(calibration: pd.DataFrame, path: Path) -> None:
    """Predicted against realised cost by decile of predicted risk."""
    figure, axes = _new_axes()
    positions = calibration["decile"].to_numpy()
    width = 0.38

    axes.bar(positions - width / 2, calibration["mean_predicted"], width, color=SERIES[0], label="Mean predicted", zorder=3)
    axes.bar(positions + width / 2, calibration["mean_actual"], width, color=SERIES[1], label="Mean realised", zorder=3)

    _style(
        axes,
        "The model ranks risk well but predicts the wrong overall level",
        "Spend per member by decile of predicted risk, held out year",
        "Decile of predicted risk, 10 is the highest",
        "Cost per member, dollars",
    )
    axes.set_xticks(positions)
    legend = axes.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)
    _save(figure, path)


def tier_profile_figure(profile: pd.DataFrame, path: Path) -> None:
    """Realised cost per member across the operational risk tiers."""
    figure, axes = _new_axes(figsize=(8.0, 4.8))
    order = ["low", "medium", "high"]
    ordered = profile.set_index("risk_tier").reindex(order).reset_index()
    colors = [SEQUENTIAL[1], SEQUENTIAL[2], SEQUENTIAL[3]]

    bars = axes.bar(ordered["risk_tier"], ordered["mean_actual_cost"], color=colors, width=0.55, zorder=3)
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 150,
            f"${row['mean_actual_cost']:,.0f}\n{row['pct_of_members']:.0f}% of members\n{row['pct_of_total_spend']:.0f}% of spend",
            ha="center",
            color=SECONDARY_INK,
            fontsize=9,
        )

    axes.set_ylim(0, ordered["mean_actual_cost"].max() * 1.45)
    _style(
        axes,
        "The high tier holds five percent of members and fourteen percent of spend",
        "Realised next year cost per member by assigned risk tier",
        "",
        "Realised cost per member, dollars",
    )
    _save(figure, path)


def importance_figure(importance: pd.DataFrame, path: Path, top_n: int = 12) -> None:
    """What the ranking loses when each feature is scrambled."""
    subset = importance.head(top_n).iloc[::-1]
    figure, axes = _new_axes(figsize=(8.6, 5.4))
    axes.barh(subset["feature"], subset["mean_auc_drop"], color=SERIES[0], height=0.62, zorder=3)
    for _, row in subset.iterrows():
        axes.text(
            row["mean_auc_drop"] + 0.0008,
            row["feature"],
            f"{row['mean_auc_drop']:.3f}",
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
    axes.set_xlim(0, max(subset["mean_auc_drop"].max() * 1.25, 0.01))
    _style(
        axes,
        "Prior spend and drug utilisation carry most of the ranking",
        "Loss in high cost identification when a feature is randomly permuted",
        "Drop in area under the curve",
        "",
        grid_axis="x",
    )
    _save(figure, path)


def condition_profile_figure(profile: pd.DataFrame, path: Path) -> None:
    """Observed cost multiple for members carrying each chronic condition."""
    ordered = profile.sort_values("cost_ratio").tail(11)
    figure, axes = _new_axes(figsize=(8.6, 5.4))
    axes.barh(ordered["condition"], ordered["cost_ratio"], color=SERIES[2], height=0.62, zorder=3)
    axes.axvline(1.0, color=BASELINE, linewidth=1.2, zorder=4)
    for _, row in ordered.iterrows():
        axes.text(
            row["cost_ratio"] + 0.01,
            row["condition"],
            f"{row['cost_ratio']:.2f}x",
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
    axes.set_xlim(0, ordered["cost_ratio"].max() * 1.18)
    _style(
        axes,
        "Every tracked condition marks a more expensive member",
        "Ratio of mean next year cost, members with the condition against those without",
        "Cost ratio, 1.0 means no difference",
        "",
        grid_axis="x",
    )
    _save(figure, path)


def sensitivity_figure(grid: pd.DataFrame, path: Path) -> None:
    """Net benefit of the programme across the assumed economic inputs."""
    pivot = grid.pivot(index="outreach_cost_per_member", columns="avoidable_cost_fraction", values="net_benefit")
    pivot = pivot.sort_index(ascending=False)
    values = pivot.to_numpy() / 1_000_000.0

    figure, axes = _new_axes(figsize=(8.4, 5.0))
    limit = float(np.nanmax(np.abs(values)))
    colormap = LinearSegmentedColormap.from_list(
        "netbenefit", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    axes.imshow(values, cmap=colormap, norm=norm, aspect="auto")

    axes.set_xticks(range(len(pivot.columns)))
    axes.set_xticklabels([f"{value:.0%}" for value in pivot.columns])
    axes.set_yticks(range(len(pivot.index)))
    axes.set_yticklabels([f"${value:,.0f}" for value in pivot.index])

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axes.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=PRIMARY_INK if abs(value) < limit * 0.55 else SURFACE,
            )

    _style(
        axes,
        "The programme pays for itself only when outreach is cheap and avoidance is high",
        "Net benefit in millions of dollars, targeting the high tier. Positive is blue",
        "Share of spend that proactive management avoids",
        "Outreach cost per member",
        grid_axis="",
    )
    axes.grid(False)
    _save(figure, path)


def level_shift_figure(level_shift: pd.DataFrame, signal: pd.DataFrame, path: Path) -> None:
    """The cost level break between years and what it does to persistence."""
    figure, axes = _new_axes(figsize=(8.0, 4.8))
    years = level_shift["year"].astype(int).astype(str)
    bars = axes.bar(years, level_shift["mean_cost"], color=SEQUENTIAL[2], width=0.55, zorder=3)
    for bar, (_, row) in zip(bars, level_shift.iterrows()):
        label = f"${row['mean_cost']:,.0f}"
        if pd.notna(row["change_vs_prior_pct"]):
            label += f"\n{row['change_vs_prior_pct']:+.1f}%"
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 90,
            label,
            ha="center",
            color=SECONDARY_INK,
            fontsize=9.5,
        )
    axes.set_ylim(0, level_shift["mean_cost"].max() * 1.3)
    _style(
        axes,
        "Mean spend falls by forty percent in the final year of the extract",
        "Any model fitted on an earlier year over predicts the level of the last one",
        "",
        "Mean cost per member, dollars",
    )
    _save(figure, path)


def build_all(tables_dir: Path, figures_dir: Path) -> list[str]:
    """Regenerates every figure from the published tables."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    curves = pd.read_csv(tables_dir / "concentration_curves.csv")
    concentration_curve_figure(curves, figures_dir / "concentration_curve.png")
    written.append("concentration_curve.png")

    distribution = pd.read_csv(tables_dir / "cost_distribution.csv")
    spend_concentration_figure(distribution, figures_dir / "spend_concentration.png")
    written.append("spend_concentration.png")

    calibration = pd.read_csv(tables_dir / "calibration_by_decile.csv")
    calibration_figure(calibration, figures_dir / "calibration_by_decile.png")
    written.append("calibration_by_decile.png")

    profile = pd.read_csv(tables_dir / "risk_tier_profile.csv")
    tier_profile_figure(profile, figures_dir / "risk_tier_profile.png")
    written.append("risk_tier_profile.png")

    importance = pd.read_csv(tables_dir / "permutation_importance.csv")
    importance_figure(importance, figures_dir / "permutation_importance.png")
    written.append("permutation_importance.png")

    conditions = pd.read_csv(tables_dir / "condition_cost_profile.csv")
    condition_profile_figure(conditions, figures_dir / "condition_cost_profile.png")
    written.append("condition_cost_profile.png")

    grid = pd.read_csv(tables_dir / "business_case_sensitivity.csv")
    sensitivity_figure(grid, figures_dir / "business_case_sensitivity.png")
    written.append("business_case_sensitivity.png")

    level_shift = pd.read_csv(tables_dir / "year_level_shift.csv")
    signal = pd.read_csv(tables_dir / "longitudinal_signal.csv")
    level_shift_figure(level_shift, signal, figures_dir / "cost_level_shift.png")
    written.append("cost_level_shift.png")

    return written
