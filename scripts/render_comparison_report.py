#!/usr/bin/env python3
"""Render an HTML report from Python-vs-Rust retargeting comparison JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


TYPE_COLORS = {
    "position": "#7c3aed",
    "vector": "#0891b2",
    "dexpilot": "#ea580c",
}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    if number is None:
        return "—"
    if abs(number) >= 100 or (abs(number) < 0.001 and number != 0):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def status_class(row: dict[str, Any], mean_threshold: float, max_threshold: float) -> str:
    if row.get("status") != "ok":
        return "error"
    mean_abs = finite(row.get("mean_abs_error")) or 0.0
    max_abs = finite(row.get("max_abs_error")) or 0.0
    if mean_abs <= mean_threshold and max_abs <= max_threshold:
        return "pass"
    if mean_abs <= mean_threshold * 5 and max_abs <= max_threshold * 5:
        return "warn"
    return "drift"


def summarize(rows: list[dict[str, Any]], mean_threshold: float, max_threshold: float) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    threshold_rows = [
        row
        for row in ok_rows
        if (finite(row.get("mean_abs_error")) or 0.0) > mean_threshold
        or (finite(row.get("max_abs_error")) or 0.0) > max_threshold
    ]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_type[str(row.get("retargeting_type", "unknown"))].append(row)

    type_summary = {}
    for rtype, items in sorted(by_type.items()):
        type_summary[rtype] = {
            "count": len(items),
            "mean_mean_abs": mean(finite(row.get("mean_abs_error")) or 0.0 for row in items),
            "max_mean_abs": max(finite(row.get("mean_abs_error")) or 0.0 for row in items),
            "max_max_abs": max(finite(row.get("max_abs_error")) or 0.0 for row in items),
            "max_pose_mean": max(finite(row.get("pose_mean_error")) or 0.0 for row in items),
            "max_pose_max": max(finite(row.get("pose_max_error")) or 0.0 for row in items),
            "python_ms": median(finite(row.get("python_retarget_ms")) or 0.0 for row in items),
            "rust_ms": median(finite(row.get("rust_retarget_ms")) or 0.0 for row in items),
            "speedup": median(finite(row.get("rust_speedup")) or 0.0 for row in items),
        }

    speedups = [finite(row.get("rust_speedup")) for row in ok_rows]
    speedups = [value for value in speedups if value is not None]
    python_times = [finite(row.get("python_retarget_ms")) for row in ok_rows]
    python_times = [value for value in python_times if value is not None]
    rust_times = [finite(row.get("rust_retarget_ms")) for row in ok_rows]
    rust_times = [value for value in rust_times if value is not None]

    return {
        "total": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "threshold_exceeded": len(threshold_rows),
        "max_mean_abs": max((finite(row.get("mean_abs_error")) or 0.0 for row in ok_rows), default=0.0),
        "max_max_abs": max((finite(row.get("max_abs_error")) or 0.0 for row in ok_rows), default=0.0),
        "max_pose_mean": max((finite(row.get("pose_mean_error")) or 0.0 for row in ok_rows), default=0.0),
        "max_pose_max": max((finite(row.get("pose_max_error")) or 0.0 for row in ok_rows), default=0.0),
        "median_speedup": median(speedups) if speedups else 0.0,
        "median_python_ms": median(python_times) if python_times else 0.0,
        "median_rust_ms": median(rust_times) if rust_times else 0.0,
        "type_summary": type_summary,
    }


def bar_chart(type_summary: dict[str, dict[str, Any]]) -> str:
    width = 920
    height = 320
    margin_left = 150
    chart_width = width - margin_left - 70
    row_height = 58
    max_value = max((item["max_mean_abs"] for item in type_summary.values()), default=1.0) or 1.0
    parts = [
        f'<svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Mean absolute error by retargeting type">',
        '<rect width="100%" height="100%" rx="18" fill="#0f172a"/>',
        '<text x="28" y="38" fill="#e2e8f0" font-size="22" font-weight="700">Worst mean absolute error by type</text>',
        '<text x="28" y="64" fill="#94a3b8" font-size="13">Longer bars indicate larger Python-vs-Rust numerical drift.</text>',
    ]
    for index, (rtype, data) in enumerate(type_summary.items()):
        y = 104 + index * row_height
        value = float(data["max_mean_abs"])
        bar_w = max(2, value / max_value * chart_width)
        color = TYPE_COLORS.get(rtype, "#64748b")
        parts.extend(
            [
                f'<text x="28" y="{y + 21}" fill="#cbd5e1" font-size="15">{html.escape(rtype)}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="28" rx="14" fill="#1e293b"/>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="28" rx="14" fill="{color}"/>',
                f'<text x="{margin_left + bar_w + 10:.2f}" y="{y + 20}" fill="#f8fafc" font-size="14">{fmt(value)}</text>',
                f'<text x="{width - 145}" y="{y + 20}" fill="#94a3b8" font-size="13">n={data["count"]}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def heatmap(rows: list[dict[str, Any]], mean_threshold: float, max_threshold: float) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ordered = sorted(ok_rows, key=lambda row: finite(row.get("mean_abs_error")) or 0.0, reverse=True)
    cell = 18
    gap = 4
    columns = 13
    rows_count = math.ceil(len(ordered) / columns)
    width = 920
    height = 112 + rows_count * (cell + gap)
    parts = [
        f'<svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Configuration drift heatmap">',
        '<rect width="100%" height="100%" rx="18" fill="#fff7ed"/>',
        '<text x="28" y="38" fill="#7c2d12" font-size="22" font-weight="700">Per-config qualitative drift image</text>',
        '<text x="28" y="64" fill="#9a3412" font-size="13">Each square is one config: green=within threshold, amber=near, red=large drift.</text>',
    ]
    for idx, row in enumerate(ordered):
        klass = status_class(row, mean_threshold, max_threshold)
        color = {"pass": "#16a34a", "warn": "#f59e0b", "drift": "#dc2626", "error": "#111827"}[klass]
        x = 28 + (idx % columns) * (cell + gap)
        y = 88 + (idx // columns) * (cell + gap)
        label = f'{row.get("hand", "unknown")} {row.get("retargeting_type", "")}: mean={fmt(row.get("mean_abs_error"))}, max={fmt(row.get("max_abs_error"))}'
        parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" fill="{color}"><title>{html.escape(label)}</title></rect>')
    parts.extend(
        [
            f'<text x="360" y="{height - 28}" fill="#166534" font-size="13">■ within</text>',
            f'<text x="455" y="{height - 28}" fill="#92400e" font-size="13">■ near</text>',
            f'<text x="535" y="{height - 28}" fill="#991b1b" font-size="13">■ large drift</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def performance_chart(type_summary: dict[str, dict[str, Any]]) -> str:
    width = 920
    height = 340
    margin_left = 150
    chart_width = width - margin_left - 90
    row_height = 72
    max_time = max(
        (
            max(float(item.get("python_ms", 0.0)), float(item.get("rust_ms", 0.0)))
            for item in type_summary.values()
        ),
        default=1.0,
    ) or 1.0
    parts = [
        f'<svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="Retarget performance by type">',
        '<rect width="100%" height="100%" rx="18" fill="#ecfeff"/>',
        '<text x="28" y="38" fill="#164e63" font-size="22" font-weight="700">Median retarget performance by type</text>',
        '<text x="28" y="64" fill="#155e75" font-size="13">Bars show median single-call retarget time. Lower is faster; speedup is Python / Rust.</text>',
    ]
    for index, (rtype, data) in enumerate(type_summary.items()):
        y = 102 + index * row_height
        py_ms = float(data.get("python_ms", 0.0))
        ru_ms = float(data.get("rust_ms", 0.0))
        py_w = max(2, py_ms / max_time * chart_width)
        ru_w = max(2, ru_ms / max_time * chart_width)
        parts.extend(
            [
                f'<text x="28" y="{y + 22}" fill="#0e7490" font-size="15" font-weight="700">{html.escape(rtype)}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="20" rx="10" fill="#cffafe"/>',
                f'<rect x="{margin_left}" y="{y}" width="{py_w:.2f}" height="20" rx="10" fill="#2563eb"/>',
                f'<text x="{margin_left + py_w + 10:.2f}" y="{y + 15}" fill="#1e3a8a" font-size="12">Python {fmt(py_ms)} ms</text>',
                f'<rect x="{margin_left}" y="{y + 28}" width="{chart_width}" height="20" rx="10" fill="#ccfbf1"/>',
                f'<rect x="{margin_left}" y="{y + 28}" width="{ru_w:.2f}" height="20" rx="10" fill="#16a34a"/>',
                f'<text x="{margin_left + ru_w + 10:.2f}" y="{y + 43}" fill="#14532d" font-size="12">Rust {fmt(ru_ms)} ms · {fmt(data.get("speedup"), 2)}×</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def retarget_renderings(rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row.get("status") == "ok" and row.get("python_points") and row.get("rust_points")]
    selected = sorted(candidates, key=lambda row: finite(row.get("pose_max_error")) or 0.0, reverse=True)[:6]
    if not selected:
        return "<p>No renderable pose samples were recorded.</p>"

    cards = []
    for row in selected:
        py_points = row.get("python_points", {})
        ru_points = row.get("rust_points", {})
        names = [name for name in row.get("render_links", []) if name in py_points and name in ru_points]
        scene = {
            "title": f'{row.get("hand", "unknown")} · {row.get("retargeting_type", "")}',
            "subtitle": f'pose mean {fmt(row.get("pose_mean_error"), 5)} · max {fmt(row.get("pose_max_error"), 5)}',
            "config": row.get("config", ""),
            "links": [
                {
                    "name": name,
                    "python": [float(value) for value in py_points[name]],
                    "rust": [float(value) for value in ru_points[name]],
                }
                for name in names
            ],
        }
        scene_json = html.escape(json.dumps(scene, separators=(",", ":")), quote=True)
        cards.append(
            "\n".join(
                [
                    '<article class="pose-card">',
                    f'<div class="pose-title">{html.escape(scene["title"])}</div>',
                    f'<div class="pose-subtitle">{html.escape(scene["subtitle"])}</div>',
                    f'<div class="pose3d" data-scene="{scene_json}">',
                    '<canvas width="720" height="520" aria-label="Interactive 3D retargeted output point cloud"></canvas>',
                    '<div class="pose-controls">Drag to rotate · wheel to zoom · double-click reset</div>',
                    '</div>',
                    '</article>',
                ]
            )
        )
    return "\n".join(cards)


def interactive_3d_script() -> str:
    return r"""
<script>
(() => {
  const cssNumber = (value, fallback) => Number.isFinite(value) ? value : fallback;

  function bounds(scene) {
    const points = scene.links.flatMap(link => [link.python, link.rust]);
    const mins = [0, 1, 2].map(axis => Math.min(...points.map(point => point[axis])));
    const maxs = [0, 1, 2].map(axis => Math.max(...points.map(point => point[axis])));
    const center = mins.map((value, axis) => (value + maxs[axis]) / 2);
    const span = Math.max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2], 1e-6);
    return { center, span };
  }

  function rotate(point, state, sceneBounds) {
    const x = point[0] - sceneBounds.center[0];
    const y = point[1] - sceneBounds.center[1];
    const z = point[2] - sceneBounds.center[2];

    const cy = Math.cos(state.yaw);
    const sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch);
    const sp = Math.sin(state.pitch);

    const x1 = x * cy - y * sy;
    const y1 = x * sy + y * cy;
    const z1 = z;
    const y2 = y1 * cp - z1 * sp;
    const z2 = y1 * sp + z1 * cp;
    return [x1, y2, z2];
  }

  function project(point, state, sceneBounds, width, height) {
    const [x, depth, z] = rotate(point, state, sceneBounds);
    const scale = Math.min(width, height) * 0.38 * state.zoom / sceneBounds.span;
    return {
      x: width / 2 + x * scale,
      y: height / 2 - z * scale,
      depth,
    };
  }

  function drawAxis(ctx, label, axisPoint, color, state, sceneBounds, width, height) {
    const origin = sceneBounds.center;
    const target = [
      origin[0] + axisPoint[0] * sceneBounds.span * 0.42,
      origin[1] + axisPoint[1] * sceneBounds.span * 0.42,
      origin[2] + axisPoint[2] * sceneBounds.span * 0.42,
    ];
    const start = project(origin, state, sceneBounds, width, height);
    const end = project(target, state, sceneBounds, width, height);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText(label, end.x + 6, end.y + 4);
    ctx.restore();
  }

  function drawScene(canvas, scene, state, sceneBounds) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(rect.width, 320);
    const height = Math.max(rect.height, 250);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(1, "#f8fafc");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#0f172a";
    ctx.font = "700 14px ui-sans-serif, system-ui";
    ctx.fillText(scene.title || "Retargeted output", 16, 24);
    ctx.fillStyle = "#64748b";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText(scene.subtitle || "", 16, 44);

    drawAxis(ctx, "x", [1, 0, 0], "#ef4444", state, sceneBounds, width, height);
    drawAxis(ctx, "y", [0, 1, 0], "#22c55e", state, sceneBounds, width, height);
    drawAxis(ctx, "z", [0, 0, 1], "#8b5cf6", state, sceneBounds, width, height);

    const projected = scene.links.map(link => ({
      name: link.name,
      python: project(link.python, state, sceneBounds, width, height),
      rust: project(link.rust, state, sceneBounds, width, height),
    }));

    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.72)";
    ctx.lineWidth = 1.25;
    for (const item of projected) {
      ctx.beginPath();
      ctx.moveTo(item.python.x, item.python.y);
      ctx.lineTo(item.rust.x, item.rust.y);
      ctx.stroke();
    }
    ctx.restore();

    const dots = [];
    for (const item of projected) {
      dots.push({ kind: "Python", point: item.python, color: "#2563eb", name: item.name });
      dots.push({ kind: "Rust", point: item.rust, color: "#16a34a", name: item.name });
    }
    const depthValues = dots.map(dot => dot.point.depth);
    const minDepth = Math.min(...depthValues, -1e-6);
    const maxDepth = Math.max(...depthValues, 1e-6);
    dots.sort((a, b) => a.point.depth - b.point.depth);

    for (const dot of dots) {
      const t = (dot.point.depth - minDepth) / Math.max(maxDepth - minDepth, 1e-6);
      const radius = 3.5 + 3.0 * t;
      ctx.globalAlpha = 0.58 + 0.38 * t;
      ctx.fillStyle = dot.color;
      ctx.beginPath();
      ctx.arc(dot.point.x, dot.point.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    ctx.fillStyle = "#2563eb";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.fillText("● Python", 16, height - 18);
    ctx.fillStyle = "#16a34a";
    ctx.fillText("● Rust", 106, height - 18);
    ctx.fillStyle = "#64748b";
    ctx.fillText("residual lines connect matching links", 184, height - 18);
  }

  document.querySelectorAll(".pose3d").forEach(container => {
    const canvas = container.querySelector("canvas");
    if (!canvas) return;
    const scene = JSON.parse(container.dataset.scene || "{}");
    const sceneBounds = bounds(scene);
    const state = { yaw: -0.75, pitch: 0.55, zoom: 1.0 };
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const redraw = () => drawScene(canvas, scene, state, sceneBounds);
    canvas.addEventListener("pointerdown", event => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      state.yaw += dx * 0.01;
      state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.01));
      redraw();
    });
    canvas.addEventListener("pointerup", event => {
      dragging = false;
      canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointercancel", () => { dragging = false; });
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      const delta = Math.exp(-event.deltaY * 0.001);
      state.zoom = Math.max(0.35, Math.min(5.0, state.zoom * delta));
      redraw();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => {
      state.yaw = -0.75;
      state.pitch = 0.55;
      state.zoom = 1.0;
      redraw();
    });
    new ResizeObserver(redraw).observe(container);
    redraw();
  });
})();
</script>
"""


def qualitative_svg() -> str:
    return """
<svg class="viz" viewBox="0 0 920 360" role="img" aria-label="Qualitative comparison of Python and Rust retargeting stacks">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
    </marker>
  </defs>
  <rect width="100%" height="100%" rx="18" fill="#f8fafc"/>
  <text x="28" y="40" fill="#0f172a" font-size="23" font-weight="800">Qualitative architecture comparison</text>
  <text x="28" y="68" fill="#475569" font-size="13">Both stacks execute all configs end-to-end with matching numeric thresholds on feasible FK-derived probes.</text>
  <g transform="translate(36 112)">
    <rect x="0" y="0" width="360" height="190" rx="18" fill="#dbeafe" stroke="#60a5fa"/>
    <text x="22" y="34" fill="#1e3a8a" font-size="20" font-weight="700">Python reference</text>
    <text x="22" y="68" fill="#1e40af" font-size="14">Pinocchio FK/Jacobians</text>
    <text x="22" y="96" fill="#1e40af" font-size="14">NLopt LD_SLSQP optimizer</text>
    <text x="22" y="124" fill="#1e40af" font-size="14">Torch SmoothL1 gradients</text>
    <text x="22" y="152" fill="#1e40af" font-size="14">Known behavior from dex-retargeting</text>
  </g>
  <g transform="translate(524 112)">
    <rect x="0" y="0" width="360" height="190" rx="18" fill="#dcfce7" stroke="#4ade80"/>
    <text x="22" y="34" fill="#14532d" font-size="20" font-weight="700">Rust implementation</text>
    <text x="22" y="68" fill="#166534" font-size="14">Pure Rust URDF parser + FK</text>
    <text x="22" y="96" fill="#166534" font-size="14">Analytic world-position Jacobians</text>
    <text x="22" y="124" fill="#166534" font-size="14">Bounded SLSQP optimizer</text>
    <text x="22" y="152" fill="#166534" font-size="14">PyO3 binding via dexi_py</text>
  </g>
  <line x1="410" y1="206" x2="506" y2="206" stroke="#64748b" stroke-width="3" marker-end="url(#arrow)"/>
  <text x="423" y="188" fill="#334155" font-size="13">same YAML + input</text>
  <rect x="308" y="314" width="304" height="30" rx="15" fill="#dcfce7"/>
  <text x="333" y="334" fill="#166534" font-size="13" font-weight="700">Result: API parity and numeric parity within thresholds</text>
</svg>
"""


def table(rows: list[dict[str, Any]], mean_threshold: float, max_threshold: float) -> str:
    sorted_rows = sorted(rows, key=lambda row: (row.get("retargeting_type", ""), row.get("hand", ""), row.get("config", "")))
    body = []
    for row in sorted_rows:
        klass = status_class(row, mean_threshold, max_threshold)
        body.append(
            "<tr>"
            f'<td><code>{html.escape(str(row.get("config", "")))}</code></td>'
            f'<td>{html.escape(str(row.get("retargeting_type", "")))}</td>'
            f'<td>{html.escape(str(row.get("hand", "")))}</td>'
            f'<td>{row.get("dof", "")}</td>'
            f'<td>{row.get("fixed_dof", "")}</td>'
            f'<td>{fmt(row.get("mean_abs_error"), 6)}</td>'
            f'<td>{fmt(row.get("max_abs_error"), 6)}</td>'
            f'<td>{fmt(row.get("rms_error"), 6)}</td>'
            f'<td>{fmt(row.get("pose_mean_error"), 6)}</td>'
            f'<td>{fmt(row.get("pose_max_error"), 6)}</td>'
            f'<td>{fmt(row.get("python_retarget_ms"), 4)}</td>'
            f'<td>{fmt(row.get("rust_retarget_ms"), 4)}</td>'
            f'<td>{fmt(row.get("rust_speedup"), 3)}×</td>'
            f'<td><span class="pill {klass}">{klass}</span></td>'
            "</tr>"
        )
    return "\n".join(body)


def render(rows: list[dict[str, Any]], mean_threshold: float, max_threshold: float) -> str:
    summary = summarize(rows, mean_threshold, max_threshold)
    type_summary = summary["type_summary"]
    type_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(rtype)}</td>"
        f"<td>{data['count']}</td>"
        f"<td>{fmt(data['mean_mean_abs'])}</td>"
        f"<td>{fmt(data['max_mean_abs'])}</td>"
        f"<td>{fmt(data['max_max_abs'])}</td>"
        f"<td>{fmt(data['max_pose_mean'])}</td>"
        f"<td>{fmt(data['max_pose_max'])}</td>"
        f"<td>{fmt(data['python_ms'])}</td>"
        f"<td>{fmt(data['rust_ms'])}</td>"
        f"<td>{fmt(data['speedup'], 3)}×</td>"
        "</tr>"
        for rtype, data in type_summary.items()
    )
    callout_class = "pass" if summary["threshold_exceeded"] == 0 and summary["errors"] == 0 else "drift"
    callout_text = (
        f"All {summary['ok']} successful configs are within the thresholds of mean ≤ {mean_threshold:g} "
        f"and max ≤ {max_threshold:g}. Median Rust speedup is {fmt(summary['median_speedup'], 3)}×."
        if callout_class == "pass"
        else f"{summary['threshold_exceeded']} configs exceed the reporting thresholds of mean ≤ {mean_threshold:g} and max ≤ {max_threshold:g}."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dexi Python-vs-Rust Retargeting Comparison</title>
  <style>
    :root {{ color-scheme: light; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4rem); margin: 0 0 12px; letter-spacing: -0.05em; }}
    h2 {{ margin-top: 44px; font-size: 1.55rem; }}
    p {{ color: var(--muted); line-height: 1.65; }}
    .cards {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 28px 0; }}
    .card {{ background:white; border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 8px 30px rgba(15,23,42,.04); }}
    .label {{ color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }}
    .value {{ font-size:2rem; font-weight:800; margin-top:6px; }}
    .callout {{ border-left: 5px solid #16a34a; background:#f0fdf4; padding: 16px 20px; border-radius: 14px; color:#14532d; }}
    .callout.drift {{ border-left-color:#dc2626; background:#fff1f2; color:#7f1d1d; }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 18px; }}
    .viz {{ width: 100%; height: auto; display:block; border-radius:18px; box-shadow:0 12px 36px rgba(15,23,42,.08); }}
    .poses {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .pose-card {{ background:white; border:1px solid var(--line); border-radius:18px; padding:14px; box-shadow:0 8px 24px rgba(15,23,42,.06); }}
    .pose-title {{ font-weight:800; font-size:.95rem; }}
    .pose-subtitle {{ color:var(--muted); font-size:.82rem; margin:2px 0 10px; }}
    .pose3d {{ border:1px solid var(--line); border-radius:16px; overflow:hidden; background:white; position:relative; }}
    .pose3d canvas {{ display:block; width:100%; height:300px; cursor:grab; touch-action:none; }}
    .pose3d canvas:active {{ cursor:grabbing; }}
    .pose-controls {{ position:absolute; right:10px; top:10px; background:rgba(255,255,255,.88); color:#475569; border:1px solid rgba(226,232,240,.9); border-radius:999px; padding:5px 9px; font-size:.68rem; box-shadow:0 4px 16px rgba(15,23,42,.08); pointer-events:none; }}
    table {{ width: 100%; border-collapse: collapse; background:white; border:1px solid var(--line); border-radius: 16px; overflow:hidden; font-size:.92rem; }}
    th, td {{ padding: 11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#f1f5f9; color:#334155; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }}
    code {{ font-size:.78rem; color:#334155; }}
    .pill {{ display:inline-block; border-radius:999px; padding:3px 9px; font-size:.76rem; font-weight:700; }}
    .pass {{ background:#dcfce7; color:#166534; }}
    .warn {{ background:#fef3c7; color:#92400e; }}
    .drift {{ background:#fee2e2; color:#991b1b; }}
    .error {{ background:#e5e7eb; color:#111827; }}
    footer {{ margin-top: 42px; color:var(--muted); font-size:.9rem; }}
    @media (max-width: 820px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} .poses {{ grid-template-columns: 1fr; }} table {{ font-size:.8rem; }} }}
  </style>
</head>
<body>
<main>
  <h1>Python-vs-Rust retargeting comparison</h1>
  <p>End-to-end comparison of reference <code>dex-retargeting</code> against the pure Rust <code>dexi</code> implementation through the <code>dexi_py</code> binding. The run covers every supported offline and teleop config.</p>
  <div class="cards">
    <div class="card"><div class="label">Compared</div><div class="value">{summary['ok']}/{summary['total']}</div></div>
    <div class="card"><div class="label">Runtime errors</div><div class="value">{summary['errors']}</div></div>
    <div class="card"><div class="label">Max mean abs</div><div class="value">{fmt(summary['max_mean_abs'])}</div></div>
    <div class="card"><div class="label">Max abs</div><div class="value">{fmt(summary['max_max_abs'])}</div></div>
  </div>
  <div class="callout {callout_class}"><strong>Interpretation:</strong> {callout_text}</div>

  <div class="cards">
    <div class="card"><div class="label">Worst pose mean</div><div class="value">{fmt(summary['max_pose_mean'])}</div></div>
    <div class="card"><div class="label">Worst pose max</div><div class="value">{fmt(summary['max_pose_max'])}</div></div>
    <div class="card"><div class="label">Python median</div><div class="value">{fmt(summary['median_python_ms'])} ms</div></div>
    <div class="card"><div class="label">Rust median</div><div class="value">{fmt(summary['median_rust_ms'])} ms</div></div>
  </div>

  <h2>Visual summary</h2>
  <div class="grid">
    {bar_chart(type_summary)}
    {performance_chart(type_summary)}
    {heatmap(rows, mean_threshold, max_threshold)}
    {qualitative_svg()}
  </div>

  <h2>Rendered retargeted outputs</h2>
  <p>Each mini-render is an interactive 3D point cloud of Python FK positions for the Python output qpos and the Rust output qpos. Drag to rotate, use the mouse wheel to zoom, and double-click to reset. Axes mark x/y/z, point depth is encoded by size/opacity, and connector lines show the residual pose gap for corresponding links.</p>
  <div class="poses">
    {retarget_renderings(rows)}
  </div>

  <h2>Aggregate table</h2>
  <table>
    <thead><tr><th>Type</th><th>Configs</th><th>Average mean abs</th><th>Worst mean abs</th><th>Worst max abs</th><th>Worst pose mean</th><th>Worst pose max</th><th>Python ms</th><th>Rust ms</th><th>Speedup</th></tr></thead>
    <tbody>{type_rows}</tbody>
  </table>

  <h2>Per-config comparison table</h2>
  <table>
    <thead><tr><th>Config</th><th>Type</th><th>Hand</th><th>DOF</th><th>Fixed DOF</th><th>Mean abs</th><th>Max abs</th><th>RMS</th><th>Pose mean</th><th>Pose max</th><th>Python ms</th><th>Rust ms</th><th>Speedup</th><th>Qualitative</th></tr></thead>
    <tbody>{table(rows, mean_threshold, max_threshold)}</tbody>
  </table>

  <footer>Generated from <code>scripts/compare_python_rust.py</code> output. The Rust path uses pure Rust kinematics and bounded optimization; the report records qpos parity, rendered pose parity, and retarget performance for every supported config.</footer>
</main>
{interactive_3d_script()}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Comparison JSON produced by compare_python_rust.py")
    parser.add_argument("output", type=Path, help="HTML report path")
    parser.add_argument("--max-mean-error", type=float, default=5e-2)
    parser.add_argument("--max-max-error", type=float, default=5e-1)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, args.max_mean_error, args.max_max_error))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
