"""Collect best-checkpoint CV test aggregates from experiment directories.

Usage:
    python scripts/compare_cv_summaries.py --results-dir results
    python scripts/compare_cv_summaries.py -r results -o results/comparison.xlsx

Requires: openpyxl (usually present when pandas.read_excel reads .xlsx files).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


PREFERRED_METRICS = tuple(f'{metric}_{domain}' for domain in ('valid', 'full')
                          for metric in ('MAE', 'RMSE', 'PSNR', 'SSIM', 'MAE_raw', 'RMSE_raw')) + (
                              'MAE', 'RMSE', 'PSNR', 'SSIM')
METRIC_DIRECTIONS = {name: 'max' if name.startswith(('PSNR', 'SSIM')) else 'min'
                     for name in PREFERRED_METRICS}
BASE_COLUMNS = (
    ("experiment", "Experiment"),
    ("status", "CV status"),
    ("completed_folds", "Completed folds"),
    ("requested_folds", "Requested folds"),
    ("model", "Model"),
    ("fusion_method", "Fusion"),
    ("clinical_features", "Clinical features"),
    ("clinical_embed_dim", "Clinical dim"),
    ("sampler", "Sampler"),
    ("time_steps", "Train steps"),
    ("sample_steps", "Sample steps"),
    ("ddim_eta", "DDIM eta"),
    ("split_file", "Split file"),
    ("summary_file", "Summary file"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare best.aggregate across results/**/cv_summary.json"
    )
    parser.add_argument("--results-dir", "-r", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Default: RESULTS_DIR/cv_best_comparison.xlsx",
    )
    return parser.parse_args()


def finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def read_experiment(path: Path, root: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("JSON root is not an object")

    aggregate = document.get("aggregate")
    best = aggregate.get("best") if isinstance(aggregate, dict) else None
    if not isinstance(best, dict) or not best:
        raise ValueError("aggregate.best is missing or empty")

    metrics = {}
    for name, values in best.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ValueError(f"invalid metric: {name!r}")
        # SSIM can be unavailable when no complete valid window exists.
        if values.get('mean') is None and values.get('std') is None:
            continue
        metrics[name] = {
            "mean": finite_number(values.get("mean"), field=f"{name}.mean"),
            "std": finite_number(values.get("std"), field=f"{name}.std"),
        }
        if metrics[name]["std"] < 0:
            raise ValueError(f"{name}.std cannot be negative")

    relative_dir = path.parent.relative_to(root)
    experiment = str(relative_dir) if relative_dir != Path(".") else root.name
    features = document.get("clinical_features", [])
    features = ", ".join(map(str, features)) if isinstance(features, list) else str(features or "")

    completed = document.get("completed_folds")
    requested = document.get("requested_folds")
    completed = int(completed) if isinstance(completed, int) and not isinstance(completed, bool) else None
    requested = int(requested) if isinstance(requested, int) and not isinstance(requested, bool) else None
    # requested_folds may itself be K<10: finishing that request is still a
    # partial result in the project's fixed 10-fold comparison.
    status = (
        "complete" if completed == requested == 10 else "partial/unknown"
    )

    return {
        "experiment": experiment,
        "status": status,
        "completed_folds": completed,
        "requested_folds": requested,
        "model": document.get("model"),
        "fusion_method": document.get("fusion_method"),
        "clinical_features": features,
        "clinical_embed_dim": document.get("clinical_embed_dim"),
        "sampler": document.get("sampler"),
        "time_steps": document.get("time_steps"),
        "sample_steps": document.get("sample_steps"),
        "ddim_eta": document.get("ddim_eta"),
        "split_file": document.get("split_file"),
        "summary_file": str(path.relative_to(root)),
        "metrics": metrics,
    }


def metric_order(rows: list[dict]) -> list[str]:
    found = {name for row in rows for name in row["metrics"]}
    return [name for name in PREFERRED_METRICS if name in found] + sorted(found.difference(PREFERRED_METRICS))


def write_workbook(rows: list[dict], output: Path) -> None:
    names = metric_order(rows)
    rows.sort(key=lambda row: (
        row["status"] != "complete",
        row["metrics"].get("MAE", {}).get("mean", float("inf")),
        row["experiment"].casefold(),
    ))
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Best aggregate"
    headers = [label for _, label in BASE_COLUMNS]
    headers += [f"{name} mean" for name in names]
    headers += [f"{name} std" for name in names]

    last_col = get_column_letter(len(headers))
    sheet.merge_cells(f"A1:{last_col}1")
    sheet["A1"] = "CV experiment comparison — best checkpoints"
    sheet["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="19324D")
    sheet["A1"].alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 30
    sheet.merge_cells(f"A2:{last_col}2")
    sheet["A2"] = (
        "TEST metrics for validation-selected checkpoints; mean/std over completed folds. "
        "full: all pixels; valid: ROI; raw: original units. Legacy names alias valid."
    )
    sheet["A2"].font = Font(size=10, color="495D72", italic=True)
    sheet.row_dimensions[2].height = 22

    header_row = 4
    for col, name in enumerate(headers, start=1):
        cell = sheet.cell(header_row, col, name)
        cell.fill = PatternFill("solid", fgColor="275D85")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[header_row].height = 32

    first_data_row = header_row + 1
    mean_offset = len(BASE_COLUMNS)
    std_offset = mean_offset + len(names)
    for row_index, record in enumerate(rows, start=first_data_row):
        base_values = [record.get(key) for key, _ in BASE_COLUMNS]
        metric_values = [record["metrics"].get(name, {}).get("mean") for name in names]
        metric_values += [record["metrics"].get(name, {}).get("std") for name in names]
        for col, value in enumerate(base_values + metric_values, start=1):
            cell = sheet.cell(row_index, col, value)
            cell.alignment = Alignment(vertical="center", wrap_text=False)
            if col > mean_offset and isinstance(value, (float, int)):
                cell.number_format = "0.0000"
            if row_index % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F2F6FA")
        if record["status"] != "complete":
            sheet.cell(row_index, 2).fill = PatternFill("solid", fgColor="FFF1CC")
            sheet.cell(row_index, 2).font = Font(bold=True, color="9C5700")

    # Mark the best mean per metric among experiments that completed their requested folds.
    complete = [row for row in rows if row["status"] == "complete"]
    for position, name in enumerate(names):
        if name == 'SIA_Bias':
            continue  # Signed bias is not a smaller-is-better error metric.
        values = [row["metrics"][name]["mean"] for row in complete if name in row["metrics"]]
        if not values:
            continue
        best = max(values) if METRIC_DIRECTIONS.get(name) == "max" else min(values)
        for row_index, record in enumerate(rows, start=first_data_row):
            found = record["metrics"].get(name)
            if record["status"] == "complete" and found and found["mean"] == best:
                cell = sheet.cell(row_index, mean_offset + position + 1)
                cell.fill = PatternFill("solid", fgColor="DCF1DE")
                cell.font = Font(bold=True, color="176239")

    sheet.freeze_panes = "C5"
    sheet.auto_filter.ref = f"A{header_row}:{last_col}{header_row + len(rows)}"
    for col, name in enumerate(headers, start=1):
        if name in {"Clinical features", "Split file", "Summary file"}:
            width = 45
        elif name == "Experiment":
            width = 32
        else:
            width = max(13, min(24, len(name) + 3))
        sheet.column_dimensions[get_column_letter(col)].width = width

    output.parent.mkdir(parents=True, exist_ok=True)
    # Write alongside the destination, then replace it only after a successful save.
    temp_file = tempfile.NamedTemporaryFile(
        prefix=".cv_comparison_", suffix=".xlsx", dir=output.parent, delete=False
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    try:
        wb.save(temp_path)
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    root = args.results_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Results directory does not exist: {root}")
    paths = sorted(root.rglob("cv_summary.json"))
    if not paths:
        raise SystemExit(f"No cv_summary.json found under: {root}")

    rows = []
    for path in paths:
        try:
            rows.append(read_experiment(path, root))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
    if not rows:
        raise SystemExit("No valid aggregate.best entries found")

    output = (args.output or root / "cv_best_comparison.xlsx").expanduser().resolve()
    write_workbook(rows, output)
    print(f"Saved {len(rows)} experiment(s) to {output}")
    if len(rows) < len(paths):
        print(f"Skipped {len(paths) - len(rows)} invalid file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
