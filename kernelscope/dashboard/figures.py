"""Plotly builders; units and evidence categories stay attached to the data."""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from kernelscope.dashboard import data, style


def variants_bar(predicted, measured, heuristic, theme="light", title="Kernel time · µs"):
    fig = go.Figure()
    plugins = list(predicted.plugin) if not predicted.empty else list(measured.index)
    labels = [p + ("<br>← library heuristic" if p == heuristic else "") for p in plugins]
    if not predicted.empty:
        fig.add_bar(name="Model prediction", x=predicted.time_us, y=labels, orientation="h",
                    marker_color=style.CATEGORICAL[theme][1],
                    customdata=predicted[["splits", "slots_per_sm", "limiter"]],
                    hovertemplate="%{y}<br>%{x:.1f} µs<br>splits %{customdata[0]} · slots/SM %{customdata[1]}"
                                  "<br>limiter %{customdata[2]}<extra>Predicted</extra>")
    if measured is not None and not measured.empty:
        values = measured.reindex(plugins)
        fig.add_bar(name="GPU measured", x=values, y=labels, orientation="h",
                    marker_color=style.CATEGORICAL[theme][0], text=values.round(1), textposition="auto",
                    hovertemplate="%{y}<br>%{x:.1f} µs<extra>Measured</extra>")
    fig.update_layout(barmode="group", height=max(350, len(plugins) * 44 + 100), xaxis_title="Kernel time (µs)")
    fig.update_yaxes(autorange="reversed")
    return style.layout(fig, theme, title, left=185)


def whatif_compare(base, scaled, theme="light", title="Resource sensitivity · model prediction", overlay=None):
    fig = go.Figure()
    for frame, label, slot in ((base, "Base machine · predicted", 0), (scaled, "Scaled machine · predicted", 1)):
        fig.add_bar(x=frame.time_us, y=frame.plugin, orientation="h", name=label,
                    marker_color=style.CATEGORICAL[theme][slot], text=frame.time_us.round(1), textposition="auto",
                    hovertemplate="%{y}<br>%{x:.1f} µs<extra>%{fullData.name}</extra>")
    if overlay is not None and not overlay.empty:
        for source, rows in overlay.groupby("source"):
            fig.add_scatter(x=rows.time_us, y=rows.plugin, name=source, mode="markers",
                            marker={"color": style.CATEGORICAL[theme][2], "size": 10, "symbol": "diamond"},
                            hovertemplate="%{y}<br>%{x:.1f} µs<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="group", height=510, xaxis_title="Kernel time (µs)")
    fig.update_yaxes(autorange="reversed")
    return style.layout(fig, theme, title, left=160)


def split_label(plugin):
    match = re.search(r"fd_s(\d+)", str(plugin))
    return f"s{match[1]}" if match else "heur" if str(plugin).startswith("flashdecoding") else str(plugin).replace("_paged", "")


def regret_heatmap(table, theme="light", title="Measured · library heuristic overhead"):
    t = table[~table.ragged].copy()
    fig = go.Figure()
    if not t.empty:
        # Head geometry must be selected upstream; avoid silently merging geometries.
        if t.duplicated(["B", "L_kv"]).any():
            raise ValueError("Select one head geometry before drawing a (B, L) heatmap")
        xs, ys = sorted(t.L_kv.unique()), sorted(t.B.unique())
        z = np.full((len(ys), len(xs)), np.nan)
        text = np.full(z.shape, "", dtype=object)
        details = np.full((*z.shape, 3), None, dtype=object)
        for row in t.itertuples():
            i, j = ys.index(row.B), xs.index(row.L_kv)
            z[i, j] = row.heuristic_regret
            text[i, j] = split_label(row.best_kernel)
            details[i, j] = [row.best_kernel, row.best_us, row.heuristic_us]
        finite = z[np.isfinite(z)]
        fig.add_trace(go.Heatmap(
            x=[str(x) for x in xs], y=[str(y) for y in ys], z=z, text=text,
            texttemplate="%{text}", customdata=details, colorscale=style.SEQUENTIAL,
            zmin=0, zmax=max(0.5, float(finite.max()) if finite.size else 0.5),
            colorbar={"title": "Overhead", "tickformat": ".0%", "thickness": 12},
            xgap=3, ygap=3,
            hovertemplate="B=%{y} · L=%{x}<br>Best %{customdata[0]} · %{customdata[1]:.1f} µs"
                          "<br>Heuristic %{customdata[2]:.1f} µs<br>Overhead %{z:.1%}<extra></extra>",
        ))
    fig.update_layout(height=380, xaxis_title="KV length (tokens)", yaxis_title="Batch size B")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return style.layout(fig, theme, title)


def ragged_bars(table, theme="light", title="Mixed lengths · measured kernel speedup opportunity", top=10):
    t = table[table.ragged].dropna(subset=["heuristic_us", "best_us"]).nlargest(top, "heuristic_regret")
    labels = [f"B{r.B} · {r.lens}" for r in t.itertuples()]
    fig = go.Figure(go.Bar(
        x=t.heuristic_us / t.best_us, y=labels, orientation="h", name="Heuristic / best measured",
        marker_color=style.CATEGORICAL[theme][0], text=(t.heuristic_us / t.best_us).map(lambda x: f"{x:.2f}×"),
        textposition="auto", customdata=t[["best_kernel", "heuristic_us", "best_us"]],
        hovertemplate="%{y}<br>%{x:.2f}×<br>Best %{customdata[0]} · %{customdata[2]:.1f} µs"
                      "<br>Heuristic %{customdata[1]:.1f} µs<extra>GPU measured</extra>",
    ))
    fig.add_vline(x=1, line_dash="dot", line_color=style.TEXT[theme][1])
    fig.update_layout(height=max(320, len(t) * 36 + 120), xaxis_title="Heuristic time / best measured time (×)")
    fig.update_yaxes(autorange="reversed")
    return style.layout(fig, theme, title, left=230)


def workload_lengths(workload, theme="light"):
    lens = list(workload.lens())
    fig = go.Figure(go.Bar(x=list(range(1, len(lens) + 1)), y=lens, name="KV tokens",
                          marker_color=style.CATEGORICAL[theme][0],
                          hovertemplate="Request %{x}<br>%{y:,} KV tokens<extra></extra>"))
    fig.update_layout(height=240, xaxis_title="Request in batch", yaxis_title="KV tokens")
    return style.layout(fig, theme, "Batch composition · 요청별 문맥 길이")


def serving_steps(serving, theme="light", title="Attention per step", metric="attn_us"):
    fig = go.Figure()
    endings, all_values = [], []
    for policy, frames in serving.items():
        steps = frames.get("steps", pd.DataFrame())
        if steps.empty or metric not in steps:
            continue
        grouped = steps.groupby("step")[metric].median() / 1000
        fig.add_scatter(x=grouped.index, y=grouped, name=policy, mode="lines+markers",
                        line={"color": style.policy_color(policy, theme), "dash": style.policy_dash(policy), "width": 2},
                        marker={"size": 8},
                        hovertemplate="Step %{x}<br>%{y:.3f} ms<extra>%{fullData.name}</extra>")
        if len(serving) <= 4 and not grouped.empty:
            endings.append((float(grouped.iloc[-1]), grouped.index[-1], policy))
            all_values.extend(grouped.tolist())
    if endings:
        span = max(max(all_values) - min(all_values), .001)
        previous = -1000.
        for value, step, policy in sorted(endings):
            position = (value - min(all_values)) / span * 215
            label_position = max(position, previous + 16)
            fig.add_annotation(x=step, y=value, text=policy, ax=14, ay=-(label_position - position),
                               showarrow=True, arrowhead=0, arrowcolor=style.GRID[theme], arrowwidth=1,
                               xanchor="left", font={"color": style.TEXT[theme][0]})
            previous = label_position
    fig.update_layout(height=350, xaxis_title="Decode step", yaxis_title="Time (ms)", showlegend=True)
    return style.layout(fig, theme, title, right=90)


def serving_batch(steps, theme="light", title="Batch composition across decode steps"):
    fig = go.Figure()
    if {"B", "n_long", "step"}.issubset(steps):
        steps = steps.groupby("step")[["B", "n_long"]].median()
        for values, label, slot in ((steps.n_long, "Long ≥ 4096 tokens", 0), (steps.B - steps.n_long, "Short", 1)):
            fig.add_scatter(x=steps.index, y=values, name=label, stackgroup="one", mode="lines",
                            line={"color": style.CATEGORICAL[theme][slot], "width": 2},
                            hovertemplate="Step %{x}<br>%{y:.0f} sequences<extra>%{fullData.name}</extra>")
    fig.update_layout(height=300, xaxis_title="Decode step", yaxis_title="Sequences")
    return style.layout(fig, theme, title)


def tpot_box(serving, theme="light", title="Time per output token · inter-token intervals"):
    fig = go.Figure()
    for policy, frames in serving.items():
        samples = data.tpot_samples(frames.get("tokens", pd.DataFrame())) / 1000
        fig.add_trace(go.Box(y=samples, name=policy, marker_color=style.policy_color(policy, theme),
                             boxpoints=False, hovertemplate="%{y:.3f} ms<extra>%{fullData.name}</extra>"))
    fig.update_layout(height=330, yaxis_title="TPOT (ms)")
    return style.layout(fig, theme, title)


def amdahl_curve(kernel_speedup, attention_fraction, overhead_fraction=0, theme="light"):
    fractions = np.linspace(0, 1, 101)
    fig = go.Figure(go.Scatter(
        x=fractions * 100,
        y=[data.amdahl_speedup(float(f), kernel_speedup, overhead_fraction) for f in fractions],
        mode="lines", name="Amdahl estimate", line={"color": style.CATEGORICAL[theme][1], "width": 2},
        hovertemplate="Attention share %{x:.0f}%<br>Estimated speedup %{y:.2f}×<extra>Modeled</extra>",
    ))
    fig.add_scatter(x=[attention_fraction * 100], y=[data.amdahl_speedup(attention_fraction, kernel_speedup, overhead_fraction)],
                    mode="markers", marker={"size": 12, "color": style.CATEGORICAL[theme][0]}, name="Selected assumption",
                    hovertemplate="Attention share %{x:.0f}%<br>Estimated speedup %{y:.2f}×<extra>Modeled</extra>")
    fig.add_hline(y=1, line_dash="dot", line_color=style.TEXT[theme][1])
    fig.update_layout(height=330, xaxis_title="Assumed attention share of baseline time (%)",
                      yaxis_title="Estimated total speedup (×)")
    return style.layout(fig, theme, "Amdahl sensitivity · modeled, not measured")


def policy_overhead(costs, theme="light"):
    from kernelscope.dashboard.research import OVERHEAD_PHASES, OVERHEAD_LABELS

    fig = go.Figure()
    for policy, rows in costs.groupby("policy", sort=False):
        rows = rows.set_index("phase").reindex(OVERHEAD_PHASES).dropna(subset=["mean_us"])
        rows = rows[rows.mean_us > 0]
        fig.add_bar(x=[OVERHEAD_LABELS[p] for p in rows.index], y=rows.mean_us, name=policy,
                    marker_color=style.policy_color(policy, theme), customdata=rows[["calls", "p90_us"]],
                    hovertemplate="%{x}<br>Mean %{y:.2f} µs<br>p90 %{customdata[1]:.2f} µs"
                                  "<br>%{customdata[0]} measured calls<extra>%{fullData.name}</extra>")
    fig.update_layout(height=350, barmode="group", yaxis_title="Policy CPU time per call (µs, log scale)")
    fig.update_yaxes(type="log")
    return style.layout(fig, theme, "Policy CPU overhead · 실제 선택 비용")


def campaign_comparison(overview, theme="light"):
    from pathlib import Path

    fig = go.Figure()
    eligible = overview[overview.claim_eligible & overview.policy.ne("heuristic")].copy()
    if eligible.empty:
        return style.layout(fig, theme, "일치가 검증된 정책의 실험별 TPOT 가속")
    eligible["label"] = eligible.apply(lambda r: (
        f"{r.model.rsplit('/', 1)[-1]}<br>{r.scenario_family} · seed {r.seed}"), axis=1)
    # Duplicate human labels must not collapse separate experiments.
    labels = eligible[["experiment", "label"]].drop_duplicates()
    duplicates = set(labels[labels.label.duplicated(keep=False)].label)
    eligible["label"] = eligible.apply(lambda r: r.label + f"<br>{r.experiment}" if r.label in duplicates else r.label, axis=1)
    for policy, rows in eligible.groupby("policy", sort=False):
        fig.add_scatter(x=rows.speedup, y=rows.label, mode="markers", name=policy,
                        marker={"color": style.policy_color(policy, theme), "size": 10},
                        error_x={"type": "data", "symmetric": False,
                                 "array": rows.speedup_ci95_high - rows.speedup,
                                 "arrayminus": rows.speedup - rows.speedup_ci95_low},
                        customdata=rows[["baseline_tpot_ms", "tpot_ms", "repeats", "prompt_kind", "evaluation_split"]],
                        hovertemplate="%{y}<br>Speedup %{x:.3f}×<br>Heuristic %{customdata[0]:.3f} ms"
                                      " → policy %{customdata[1]:.3f} ms<br>%{customdata[2]} repetitions"
                                      "<br>%{customdata[3]} · %{customdata[4]}<extra>%{fullData.name}</extra>")
    fig.add_vline(x=1, line_dash="dot", line_color=style.TEXT[theme][1])
    fig.update_layout(height=max(350, eligible.label.nunique() * 70 + 150),
                      xaxis_title="Heuristic TPOT / policy TPOT (×)")
    fig.update_yaxes(autorange="reversed")
    style.layout(fig, theme, "모델·입력·seed별 독립 비교 · TPOT 가속", left=360)
    fig.update_layout(margin={"t": 100}, title={"y": .98, "yanchor": "top"},
                      legend={"y": 1.02, "yanchor": "bottom"})
    return fig
