"""Generate a richer quality report for exported synthetic datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exported synthetic datasets.")
    parser.add_argument("--data-dir", default="simulated_data", help="Directory containing simulated datasets.")
    parser.add_argument("--output-dir", default="simulated_data/analysis", help="Directory for generated analysis artifacts.")
    return parser.parse_args()


def load_split_arrays(data_dir: Path, split_name: str) -> Dict[str, np.ndarray]:
    npz = np.load(data_dir / f"{split_name}.npz", allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def load_split_summary(data_dir: Path, split_name: str) -> Dict:
    return json.loads((data_dir / f"{split_name}_summary.json").read_text(encoding="utf-8"))


def histogram_counts(values: np.ndarray, bins: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    return np.histogram(values.astype(np.float64), bins=bins)


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    flattened = x[:, :, :, 0].transpose(1, 0, 2).reshape(x.shape[1], -1)
    return np.corrcoef(flattened)


def fault_coverage(split_summaries: Dict[str, Dict]) -> Dict[str, int]:
    coverage: Dict[str, int] = {}
    for summary in split_summaries.values():
        for fault_type, count in summary.get("fault_type_counts", {}).items():
            coverage[fault_type] = coverage.get(fault_type, 0) + int(count)
    return dict(sorted(coverage.items()))


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222} .title{font-size:16px;font-weight:bold} .axis{stroke:#555;stroke-width:1} .grid{stroke:#ddd;stroke-width:1} .bar{fill:#4c78a8} .bar2{fill:#f58518} .cellText{font-size:11px;fill:#111}</style>',
    ]


def write_svg(path: Path, lines: Iterable[str]):
    path.write_text("\n".join(list(lines) + ["</svg>"]), encoding="utf-8")


def build_histogram_svg(series_map: Dict[str, np.ndarray], path: Path, title: str):
    width = 1100
    height = 700
    margin_left = 60
    margin_bottom = 40
    plot_width = width - margin_left - 20
    plot_height = height - 60 - margin_bottom
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>')

    panel_count = len(series_map)
    cols = 2
    rows = math.ceil(panel_count / cols)
    panel_w = plot_width / cols
    panel_h = plot_height / rows

    for idx, (name, values) in enumerate(series_map.items()):
        row = idx // cols
        col = idx % cols
        origin_x = margin_left + col * panel_w
        origin_y = 50 + row * panel_h
        hist, edges = histogram_counts(values)
        max_count = max(int(hist.max()), 1)
        lines.append(f'<text x="{origin_x + panel_w/2}" y="{origin_y + 14}" text-anchor="middle">{name}</text>')
        lines.append(f'<line x1="{origin_x+30}" y1="{origin_y+panel_h-25}" x2="{origin_x+panel_w-10}" y2="{origin_y+panel_h-25}" class="axis"/>')
        lines.append(f'<line x1="{origin_x+30}" y1="{origin_y+25}" x2="{origin_x+30}" y2="{origin_y+panel_h-25}" class="axis"/>')

        bar_area_w = panel_w - 50
        bar_area_h = panel_h - 55
        bar_w = max(bar_area_w / len(hist), 1)
        for bar_idx, count in enumerate(hist):
            bar_h = (count / max_count) * bar_area_h
            x = origin_x + 30 + bar_idx * bar_w
            y = origin_y + panel_h - 25 - bar_h
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-1, 1):.2f}" height="{bar_h:.2f}" class="bar"/>'
            )

        lines.append(f'<text x="{origin_x+30}" y="{origin_y+panel_h-5}" text-anchor="start">{edges[0]:.1f}</text>')
        lines.append(f'<text x="{origin_x+panel_w-10}" y="{origin_y+panel_h-5}" text-anchor="end">{edges[-1]:.1f}</text>')
        lines.append(f'<text x="{origin_x+34}" y="{origin_y+35}" text-anchor="start">max {max_count}</text>')

    write_svg(path, lines)


def build_bar_chart_svg(values: Dict[str, int], path: Path, title: str):
    width = 900
    height = 480
    margin_left = 80
    margin_bottom = 80
    plot_width = width - margin_left - 20
    plot_height = height - 60 - margin_bottom
    max_value = max(values.values()) if values else 1
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>')
    lines.append(f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-20}" y2="{height-margin_bottom}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="50" x2="{margin_left}" y2="{height-margin_bottom}" class="axis"/>')

    bar_w = plot_width / max(len(values), 1)
    for idx, (label, value) in enumerate(values.items()):
        x = margin_left + idx * bar_w + 10
        bar_height = (value / max_value) * (plot_height - 20)
        y = height - margin_bottom - bar_height
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-20, 12):.2f}" height="{bar_height:.2f}" class="bar2"/>')
        lines.append(f'<text x="{x + (bar_w-20)/2:.2f}" y="{height-margin_bottom+18}" text-anchor="middle">{label}</text>')
        lines.append(f'<text x="{x + (bar_w-20)/2:.2f}" y="{y-6:.2f}" text-anchor="middle">{value}</text>')

    write_svg(path, lines)


def build_heatmap_svg(matrix: np.ndarray, labels: Sequence[str], path: Path, title: str):
    width = 700
    height = 700
    margin = 140
    cell_size = (width - margin - 40) / max(len(labels), 1)
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>')

    for row, row_label in enumerate(labels):
        y = margin + row * cell_size
        lines.append(f'<text x="{margin-10}" y="{y + cell_size/2:.2f}" text-anchor="end">{row_label}</text>')
        lines.append(f'<text x="{margin + row*cell_size + cell_size/2:.2f}" y="{margin-16}" text-anchor="middle">{row_label}</text>')
        for col in range(len(labels)):
            x = margin + col * cell_size
            value = float(matrix[row, col])
            normalized = (value + 1.0) / 2.0
            red = int(255 * (1 - normalized))
            blue = int(255 * normalized)
            green = 120
            color = f'rgb({red},{green},{blue})'
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_size:.2f}" height="{cell_size:.2f}" fill="{color}" stroke="#fff"/>')
            lines.append(f'<text x="{x + cell_size/2:.2f}" y="{y + cell_size/2 + 4:.2f}" text-anchor="middle" class="cellText">{value:.2f}</text>')

    write_svg(path, lines)


def build_markdown_report(report: Dict[str, object], path: Path):
    lines = ["# Simulated Data Quality Report", ""]
    lines.append("## Overall Assessment")
    lines.append("")
    lines.append(f"- Total samples: {report['total_samples']}")
    lines.append(f"- Overall fault density: {report['overall_fault_density']:.4f}")
    lines.append(f"- Overall faulty sample ratio: {report['overall_faulty_sample_ratio']:.4f}")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    for split_name, split_summary in report["split_summaries"].items():
        lines.append(f"### {split_name}")
        lines.append(f"- Sample count: {split_summary['sample_count']}")
        lines.append(f"- Fault density: {split_summary['fault_density']:.4f}")
        lines.append(f"- Faulty sample ratio: {split_summary['faulty_sample_ratio']:.4f}")
        lines.append(f"- Avg credibility target: {split_summary['avg_credibility_target']:.4f}")
        lines.append("")
    lines.append("## Fault Coverage")
    lines.append("")
    for fault_type, count in report["fault_coverage"].items():
        lines.append(f"- {fault_type}: {count}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Sensor distributions: {report['artifacts']['sensor_distributions_svg']}")
    lines.append(f"- Correlation heatmap: {report['artifacts']['correlation_heatmap_svg']}")
    lines.append(f"- Fault coverage chart: {report['artifacts']['fault_coverage_svg']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_arrays = {split: load_split_arrays(data_dir, split) for split in SPLITS}
    split_summaries = {split: load_split_summary(data_dir, split) for split in SPLITS}

    combined_x = np.concatenate([split_arrays[split]["X"] for split in SPLITS], axis=0)
    sensor_ids = [str(item) for item in split_arrays["train"]["sensor_ids"].tolist()]

    sensor_series = {
        sensor_ids[idx]: combined_x[:, idx, :, 0].reshape(-1)
        for idx in range(len(sensor_ids))
    }

    correlations = correlation_matrix(combined_x)
    coverage = fault_coverage(split_summaries)

    build_histogram_svg(sensor_series, output_dir / "sensor_distributions.svg", "Sensor Value Distributions")
    build_heatmap_svg(correlations, sensor_ids, output_dir / "sensor_correlation_heatmap.svg", "Sensor Correlation Heatmap")
    build_bar_chart_svg(coverage, output_dir / "fault_coverage.svg", "Fault Type Coverage")

    total_samples = sum(split_summaries[split]["sample_count"] for split in SPLITS)
    total_faulty_samples = sum(split_summaries[split]["faulty_sample_count"] for split in SPLITS)
    total_fault_points = sum(split_summaries[split]["fault_point_count"] for split in SPLITS)
    total_points = sum(
        split_summaries[split]["sample_count"]
        * split_summaries[split]["num_sensors"]
        * split_summaries[split]["window_size"]
        for split in SPLITS
    )

    report = {
        "total_samples": total_samples,
        "overall_faulty_sample_ratio": total_faulty_samples / max(total_samples, 1),
        "overall_fault_density": total_fault_points / max(total_points, 1),
        "split_summaries": split_summaries,
        "fault_coverage": coverage,
        "correlation_matrix": correlations.tolist(),
        "artifacts": {
            "sensor_distributions_svg": "sensor_distributions.svg",
            "correlation_heatmap_svg": "sensor_correlation_heatmap.svg",
            "fault_coverage_svg": "fault_coverage.svg",
        },
    }

    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    build_markdown_report(report, output_dir / "quality_report.md")

    print(f"Quality report generated in {output_dir.resolve()}")


if __name__ == "__main__":
    main()