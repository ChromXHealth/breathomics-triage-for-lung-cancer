"""Manuscript-style performance curve figures.

All statistics shown in the figures are computed from the supplied evaluation
results; this module never invents annotation text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_FONT = {"family": "Arial", "size": 20, "color": "black"}
TRANSPARENT_BACKGROUND = "rgba(0,0,0,0)"
LEGEND_BOTTOM_RIGHT = {"xanchor": "right", "yanchor": "bottom", "x": 1, "y": 0.0}
DEFAULT_TWO_CLASS_COLORS = ("#4871B3", "#E73C36")


def _load_plotly_graph_objects() -> Any:
    """Import plotly lazily so CLI help works without it installed."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for figure generation. Install it via the "
            "project environment (see environment.yml)."
        ) from exc
    return go


def _apply_axis_style(fig: Any, show_grid: bool = False) -> None:
    grid_color = "rgba(153, 153, 153, 0.3)" if show_grid else "rgba(153, 153, 153, 0)"
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        showgrid=show_grid,
        gridcolor=grid_color,
        ticks="inside",
        tickwidth=2,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        showgrid=show_grid,
        gridcolor=grid_color,
        ticks="inside",
        tickwidth=2,
    )


def roc_curve_payload(result: dict[str, Any], model: str) -> dict[str, Any]:
    """Build a ROC drawing payload from an evaluation result."""
    roc = result["roc"]
    return {
        "model": model,
        "auc": result["AUROC"],
        "fpr": roc["fpr"],
        "tpr": roc["tpr"],
    }


def draw_model_roc(
    *payloads: dict[str, Any],
    color_scheme: Sequence[str],
    show_diagonal: bool = True,
    fig_size: tuple[int, int] = (700, 700),
    auc_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> Any:
    """Draw ROC curves for one or more evaluated prediction sets.

    Each payload carries ``model``, ``auc``, ``fpr``, and ``tpr`` keys, as
    produced by :func:`roc_curve_payload`. When ``auc_bounds`` is supplied it
    maps model names to ``(lower, upper)`` intervals shown in the legend.
    """
    go = _load_plotly_graph_objects()
    fig = go.Figure()

    for payload, color in zip(payloads, color_scheme):
        label = f"{payload['model']} AUC={payload['auc']:.2f}"
        if auc_bounds and payload["model"] in auc_bounds:
            lower, upper = auc_bounds[payload["model"]]
            label += f", 95% CI: {lower:.2f} - {upper:.2f}"
        fig.add_trace(
            go.Scatter(
                x=payload["fpr"],
                y=payload["tpr"],
                name=label,
                mode="lines",
                line_color=color,
            )
        )

    if show_diagonal:
        diagonal = np.linspace(0, 1, 11)
        fig.add_trace(
            go.Scatter(
                x=diagonal,
                y=diagonal,
                mode="lines",
                line_color="black",
                line_dash="dash",
                showlegend=False,
            )
        )

    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis_title="1-Specificity",
        yaxis_title="Sensitivity",
        legend=LEGEND_BOTTOM_RIGHT.copy(),
        showlegend=True,
        font=DEFAULT_FONT.copy(),
        plot_bgcolor=TRANSPARENT_BACKGROUND,
        yaxis_range=[-0.01, 1.01],
        xaxis_range=[-0.01, 1.01],
    )
    _apply_axis_style(fig, show_grid=False)
    return fig


def write_figure(fig: Any, output_path: Any, width: int = 800, height: int = 800) -> None:
    """Write a plotly figure to PNG and SVG with the manuscript margins."""
    fig.update_layout(
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
        width=width,
        height=height,
    )
    fig.update_xaxes(linewidth=3)
    fig.update_yaxes(linewidth=3)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(destination))
    svg_path = destination.with_suffix(".svg")
    fig.write_image(str(svg_path))


BOX_COLOR_CYCLE = ("#4871B3", "#E73C36", "#B88640", "#168676", "#27447C", "#991F22")


def draw_boxplot(
    frame: Any,
    label_column: str,
    group_order: Sequence[str],
    y_column: str = "pred",
    colors: Sequence[str] = BOX_COLOR_CYCLE,
    y_title: str = "Model Prediction Score",
) -> Any:
    """Draw the manuscript prediction-score boxplot for ordered groups."""
    go = _load_plotly_graph_objects()
    fig = go.Figure()
    for index, group in enumerate(group_order):
        values = frame.loc[frame[label_column] == group, y_column]
        if len(values) == 0:
            continue
        color = colors[index % len(colors)]
        fig.add_trace(
            go.Box(
                y=values,
                name=f"{group} (n={len(values)})",
                boxpoints=False,
                line_color=color,
                fillcolor=color,
            )
        )
    fig.update_layout(
        width=750,
        height=700,
        showlegend=False,
        yaxis_title=y_title,
        font={"family": "Arial", "size": 24, "color": "black"},
        plot_bgcolor=TRANSPARENT_BACKGROUND,
    )
    fig.update_xaxes(tickangle=30)
    _apply_axis_style(fig, show_grid=False)
    return fig


def draw_metric_bars(
    rows: Sequence[dict[str, Any]],
    metric: str = "Sensitivity",
    colors: Sequence[str] = BOX_COLOR_CYCLE,
    y_title: Optional[str] = None,
) -> Any:
    """Draw a bar chart of one subgroup metric with Wilson error bars."""
    go = _load_plotly_graph_objects()
    fig = go.Figure()
    for index, row in enumerate(rows):
        color = colors[index % len(colors)]
        lower = row[metric] - row[f"{metric}_lower"]
        upper = row[f"{metric}_upper"] - row[metric]
        fig.add_trace(
            go.Bar(
                x=[row["group"]],
                y=[row[metric]],
                name=row["group"],
                marker_color=color,
                error_y={"type": "data", "array": [upper], "arrayminus": [lower]},
                showlegend=False,
            )
        )
    fig.update_layout(
        width=800,
        height=700,
        yaxis_title=y_title or metric,
        font={"family": "Arial", "size": 24, "color": "black"},
        plot_bgcolor=TRANSPARENT_BACKGROUND,
        yaxis_range=[0, 1.05],
    )
    _apply_axis_style(fig, show_grid=False)
    return fig


def draw_dca_curve(
    curves: Sequence[tuple[str, Any, Any]],
) -> Any:
    """Draw a decision-curve analysis figure.

    ``curves`` contains ``(name, net_benefit, thresholds)`` tuples; every
    value is computed by the caller.
    """
    go = _load_plotly_graph_objects()
    fig = go.Figure()
    palette = ("#E73C36", "#4871B3", "#B88640", "#168676")
    for index, (name, net_benefit, thresholds) in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=net_benefit,
                name=name,
                mode="lines",
                line_color=palette[index % len(palette)],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode="lines",
            line_color="black",
            line_dash="dash",
            name="Treat none",
            showlegend=False,
        )
    )
    fig.update_layout(
        width=800,
        height=700,
        xaxis_title="Threshold Probability",
        yaxis_title="Standardized Net Benefit",
        font={"family": "Arial", "size": 24, "color": "black"},
        plot_bgcolor=TRANSPARENT_BACKGROUND,
    )
    _apply_axis_style(fig, show_grid=True)
    return fig


def pr_curve_payload(pr_result: dict[str, Any], model: str) -> dict[str, Any]:
    """Build a PR drawing payload from a precision-recall result."""
    return {
        "model": model,
        "ap": pr_result["ap"],
        "ap_lower": pr_result.get("ap_lower"),
        "ap_upper": pr_result.get("ap_upper"),
        "precision": pr_result["precision"],
        "recall": pr_result["recall"],
    }


def draw_model_prc(
    *payloads: dict[str, Any],
    color_scheme: Sequence[str],
    show_prevalence: Optional[float] = None,
    fig_size: tuple[int, int] = (700, 700),
) -> Any:
    """Draw precision-recall curves for one or more prediction sets.

    Each payload carries ``model``, ``ap``, ``precision``, and ``recall``
    keys, as produced by :func:`pr_curve_payload`. When ``ap_lower`` and
    ``ap_upper`` are present they are shown in the legend. ``show_prevalence``
    draws the random-guess horizontal baseline at the observed prevalence.
    """
    go = _load_plotly_graph_objects()
    fig = go.Figure()

    for payload, color in zip(payloads, color_scheme):
        label = f"{payload['model']} AP={payload['ap']:.2f}"
        if payload.get("ap_lower") is not None and payload.get("ap_upper") is not None:
            label += f", 95% CI: {payload['ap_lower']:.2f} - {payload['ap_upper']:.2f}"
        fig.add_trace(
            go.Scatter(
                x=payload["recall"],
                y=payload["precision"],
                name=label,
                mode="lines",
                line_color=color,
            )
        )

    if show_prevalence is not None:
        fig.add_hline(
            y=show_prevalence,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Prevalence = {show_prevalence:.2f}",
            annotation_position="bottom right",
            annotation_font_color="gray",
            annotation_font_size=24,
        )

    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=LEGEND_BOTTOM_RIGHT.copy(),
        showlegend=True,
        font=DEFAULT_FONT.copy(),
        plot_bgcolor=TRANSPARENT_BACKGROUND,
        yaxis_range=[0, 1.01],
        xaxis_range=[0, 1.01],
    )
    _apply_axis_style(fig, show_grid=False)
    return fig


def draw_calibration_curve(
    calibration: dict[str, Any],
    model_name: str = "Model",
    brier: Optional[dict[str, Any]] = None,
    color_scheme: Sequence[str] = ("#1f5582", "#C8DCE6"),
    fig_size: tuple[int, int] = (800, 800),
) -> Any:
    """Draw a calibration curve with optional CI band, histogram, and Brier.

    ``calibration`` is the dict returned by
    :func:`src.calibration.compute_calibration_curve`; ``brier`` is the dict
    returned by :func:`src.calibration.compute_brier_score`. All annotation
    values are computed from these inputs.
    """
    go = _load_plotly_graph_objects()

    bin_centers = calibration["bin_centers"]
    fraction_positives = calibration["fraction_positives"]
    counts = calibration["counts"]

    valid_mask = ~np.isnan(fraction_positives)
    x_valid = bin_centers[valid_mask]
    y_valid = fraction_positives[valid_mask]
    max_count = float(np.max(counts)) if len(counts) > 0 else 1.0

    fig = go.Figure()

    # Sample-count histogram on a secondary y-axis.
    n_bins = calibration["n_bins"]
    bin_width = 1.0 / n_bins
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_width * 0.9,
            marker={"color": color_scheme[1], "opacity": 0.5},
            name="Sample Distribution",
            hovertemplate="Count: %{y}<extra></extra>",
            yaxis="y2",
            showlegend=True,
        )
    )

    # Perfectly calibrated diagonal.
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "gray", "width": 2, "dash": "dash"},
            name="Perfectly Calibrated",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Bootstrap confidence band.
    if "ci_lower" in calibration and "ci_upper" in calibration:
        ci_lower = calibration["ci_lower"][valid_mask]
        ci_upper = calibration["ci_upper"][valid_mask]
        ci_valid = ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
        x_ci = np.concatenate([x_valid[ci_valid], x_valid[ci_valid][::-1]])
        y_ci = np.concatenate([ci_upper[ci_valid], ci_lower[ci_valid][::-1]])
        fig.add_trace(
            go.Scatter(
                x=x_ci,
                y=y_ci,
                fill="toself",
                fillcolor=_hex_to_rgba(color_scheme[0], 0.15),
                line={"color": "rgba(0,0,0,0)"},
                name="95% CI",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Model calibration curve, annotated with the Brier score when supplied.
    legend_name = model_name
    if brier is not None:
        legend_name += (
            f" (Brier={brier['brier_score']:.3f}, "
            f"95% CI: {brier['brier_ci_lower']:.3f}-{brier['brier_ci_upper']:.3f})"
        )
    fig.add_trace(
        go.Scatter(
            x=x_valid,
            y=y_valid,
            mode="lines+markers",
            line={"color": color_scheme[0], "width": 3},
            marker={"size": 12, "color": "white", "line": {"color": color_scheme[0], "width": 2}},
            name=legend_name,
            hovertemplate="Pred: %{x:.3f}<br>Obs: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis={"title": "Predicted Probability", "range": [0, 1]},
        yaxis={"title": "Observed Fraction", "range": [0, 1]},
        yaxis2={
            "title": "Sample Count",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "showline": True,
            "linewidth": 2,
            "linecolor": "gray",
            "range": [0, max_count * 1.1],
        },
        legend={
            "xanchor": "center",
            "yanchor": "bottom",
            "x": 0.5,
            "y": 1.0,
            "orientation": "h",
        },
        showlegend=True,
        font=DEFAULT_FONT.copy(),
        plot_bgcolor=TRANSPARENT_BACKGROUND,
        barmode="overlay",
    )
    _apply_axis_style(fig, show_grid=True)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to an rgba string with the given alpha."""
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def draw_mean_abs_shap_bar(
    frame: Any,
    feature_column: str,
    value_columns: Sequence[str],
    color_scheme: Sequence[str] = ("#4871B3", "#E73C36"),
    fig_size: tuple[int, int] = (1000, 600),
) -> Any:
    """Draw grouped bars of mean absolute SHAP values per feature.

    ``frame`` holds one row per feature with a name column
    (``feature_column``) and one numeric column per dataset
    (``value_columns``, used as the legend labels). Bar labels show the
    values with three decimals, matching the manuscript style.
    """
    go = _load_plotly_graph_objects()
    fig = go.Figure()

    for column, color in zip(value_columns, color_scheme):
        fig.add_trace(
            go.Bar(
                x=frame[feature_column],
                y=frame[column],
                name=column,
                marker_color=color,
                width=0.35,
                text=frame[column],
                textposition="outside",
                texttemplate="%{text:.3f}",
            )
        )

    y_max = float(frame[list(value_columns)].max().max())
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP|",
        legend={"xanchor": "right", "yanchor": "top", "x": 1, "y": 1.0},
        showlegend=True,
        font=DEFAULT_FONT.copy(),
        plot_bgcolor=TRANSPARENT_BACKGROUND,
        barmode="group",
        yaxis_range=[0, y_max * 1.15 if y_max > 0 else 1],
    )
    _apply_axis_style(fig, show_grid=False)
    return fig
