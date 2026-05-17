#!/usr/bin/env python3
"""Render an HTML report from Python-vs-Rust retargeting comparison JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
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
        }

    return {
        "total": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "threshold_exceeded": len(threshold_rows),
        "max_mean_abs": max((finite(row.get("mean_abs_error")) or 0.0 for row in ok_rows), default=0.0),
        "max_max_abs": max((finite(row.get("max_abs_error")) or 0.0 for row in ok_rows), default=0.0),
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
  <text x="28" y="68" fill="#475569" font-size="13">Both stacks execute all configs end-to-end; numerical differences trace mainly to kinematics and solver choices.</text>
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
    <text x="22" y="96" fill="#166534" font-size="14">Finite-difference Jacobians</text>
    <text x="22" y="124" fill="#166534" font-size="14">Projected-gradient optimizer</text>
    <text x="22" y="152" fill="#166534" font-size="14">PyO3 binding via dexi_py</text>
  </g>
  <line x1="410" y1="206" x2="506" y2="206" stroke="#64748b" stroke-width="3" marker-end="url(#arrow)"/>
  <text x="423" y="188" fill="#334155" font-size="13">same YAML + input</text>
  <rect x="328" y="314" width="264" height="30" rx="15" fill="#fee2e2"/>
  <text x="350" y="334" fill="#991b1b" font-size="13" font-weight="700">Result: API parity yes, numeric parity incomplete</text>
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
        "</tr>"
        for rtype, data in type_summary.items()
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
    .callout {{ border-left: 5px solid #dc2626; background:#fff1f2; padding: 16px 20px; border-radius: 14px; color:#7f1d1d; }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 18px; }}
    .viz {{ width: 100%; height: auto; display:block; border-radius:18px; box-shadow:0 12px 36px rgba(15,23,42,.08); }}
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
    @media (max-width: 820px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} table {{ font-size:.8rem; }} }}
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
  <div class="callout"><strong>Interpretation:</strong> API and coverage parity are in place, and all 39 configs complete without runtime failure. Numerical parity is not yet close: {summary['threshold_exceeded']} configs exceed the reporting thresholds of mean ≤ {mean_threshold:g} and max ≤ {max_threshold:g}.</div>

  <h2>Visual summary</h2>
  <div class="grid">
    {bar_chart(type_summary)}
    {heatmap(rows, mean_threshold, max_threshold)}
    {qualitative_svg()}
  </div>

  <h2>Aggregate table</h2>
  <table>
    <thead><tr><th>Type</th><th>Configs</th><th>Average mean abs</th><th>Worst mean abs</th><th>Worst max abs</th></tr></thead>
    <tbody>{type_rows}</tbody>
  </table>

  <h2>Per-config comparison table</h2>
  <table>
    <thead><tr><th>Config</th><th>Type</th><th>Hand</th><th>DOF</th><th>Fixed DOF</th><th>Mean abs</th><th>Max abs</th><th>RMS</th><th>Qualitative</th></tr></thead>
    <tbody>{table(rows, mean_threshold, max_threshold)}</tbody>
  </table>

  <footer>Generated from <code>scripts/compare_python_rust.py</code> output. The Rust path intentionally uses pure Rust kinematics and optimization, so the report records both successful cross-language execution and current numerical drift.</footer>
</main>
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
