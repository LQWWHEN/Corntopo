#!/usr/bin/env python3
"""Calculate anterior corneal surgically induced astigmatism (SIA).

The input workbook is expected to contain paired pre-operative and
post-operative Pentacam summary fields.  The script keeps every input row,
calculates SIA where possible, and records data-quality flags instead of
silently dropping questionable records.

Default axis convention:
    Pentacam ``Cornea Front Axis`` is treated as the flat meridian.  With a
    positive astigmatism magnitude C, the vector components are::

        X = C * cos(2 * axis)
        Y = C * sin(2 * axis)

    SIA is the vector difference between postoperative and preoperative
    astigmatism.  ``--axis-convention steep`` converts a steep-axis input to
    the equivalent flat-axis representation before calculating.

Example:
    python scripts/calculate_sia.py \
        --input paired_dataset.xlsx \
        --output paired_dataset_sia.xlsx

Strict export, retaining only quality-approved chronological pairs:
    python scripts/calculate_sia.py \
        --input paired_dataset.xlsx \
        --output paired_dataset_sia_valid.xlsx \
        --valid-only
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PRE = "pre"
DEFAULT_POST = "post"


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _normalise_numeric(value: Any) -> float:
    """Return a finite float or NaN."""
    if _is_missing(value):
        return float("nan")
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _parse_date(value: Any) :
    """Parse common Pentacam/Excel date representations.

    The current export contains values such as 26062025 and 4082025, which
    represent 26-06-2025 and 04-08-2025.  ISO dates and already parsed Excel
    timestamps are also accepted.
    """
    if _is_missing(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp,)):
        return value.normalize()

    text = str(value).strip()
    if not text:
        return pd.NaT

    if text.isdigit():
        if len(text) <= 8:
            text = text.zfill(8)
        if len(text) == 8:
            # Prefer YYYYMMDD when the first four characters look like a year.
            if 1900 <= int(text[:4]) <= 2100:
                parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
                if not pd.isna(parsed):
                    return parsed
            return pd.to_datetime(text, format="%d%m%Y", errors="coerce")

    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _parse_time(value: Any) -> str | None:
    """Return a zero-padded HHMMSS string for common time representations."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return text.zfill(6)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%H%M%S")


def parse_datetime(date_value: Any, time_value: Any) :
    """Combine the date and time fields into a timestamp."""
    date = _parse_date(date_value)
    time_text = _parse_time(time_value)
    if pd.isna(date) or time_text is None:
        return pd.NaT
    return pd.to_datetime(
        f"{date.strftime('%Y-%m-%d')} {time_text}",
        format="%Y-%m-%d %H%M%S",
        errors="coerce",
    )


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from candidates."""
    return next((column for column in candidates if column in df.columns), None)


def make_timestamp_columns(df: pd.DataFrame, side: str) -> pd.Series:
    """Read the normalised Chinese date/time fields with CUR fallbacks."""
    date_column = find_first_column(
        df,
        [
            f"{side}_检查日期",
            f"{side}_EXAMINATION_Date",
        ],
    )
    time_column = find_first_column(
        df,
        [
            f"{side}_检查时间",
            f"{side}_EXAMINATION_Time",
        ],
    )
    if date_column is None or time_column is None:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.Series(
        [
            parse_datetime(date_value, time_value)
            for date_value, time_value in zip(df[date_column], df[time_column])
        ],
        index=df.index,
        dtype="datetime64[ns]",
    )


def astig_to_vector(
    astigmatism: pd.Series,
    axis_deg: pd.Series,
    *,
    axis_convention: str = "flat",
) -> tuple[pd.Series, pd.Series]:
    """Convert positive astigmatism magnitude and axis to X/Y components."""
    c = pd.to_numeric(astigmatism, errors="coerce")
    axis = pd.to_numeric(axis_deg, errors="coerce")

    if axis_convention == "steep":
        axis = (axis + 90.0) % 180.0
    elif axis_convention != "flat":
        raise ValueError(f"未知轴位约定: {axis_convention}")

    theta = np.deg2rad(axis)
    x = c * np.cos(2.0 * theta)
    y = c * np.sin(2.0 * theta)
    return x, y


def _quality_ok(df: pd.DataFrame, side: str) -> tuple[pd.Series, bool]:
    """Return per-row QS status and whether a QS column was available."""
    column = find_first_column(
        df,
        [
            f"{side}_GENERAL DATA_QS Status",
            f"{side}_QS Status",
        ],
    )
    if column is None:
        return pd.Series(True, index=df.index, dtype="boolean"), False
    # Missing QS status is treated as not approved when a QS column exists.
    return df[column].astype("string").str.strip().eq("OK").fillna(False), True


def calculate_sia(
    df: pd.DataFrame,
    *,
    pre_prefix: str = DEFAULT_PRE,
    post_prefix: str = DEFAULT_POST,
    axis_convention: str = "flat",
) -> pd.DataFrame:
    """Calculate per-eye anterior corneal SIA and quality flags."""
    #pre_astig_column = f"{pre_prefix}_GENERAL DATA_Cornea Front Astig"
    #pre_axis_column = f"{pre_prefix}_GENERAL DATA_Cornea Front Axis"
    #post_astig_column = f"{post_prefix}_GENERAL DATA_Cornea Front Astig"
    #post_axis_column = f"{post_prefix}_GENERAL DATA_Cornea Front Axis"
    pre_astig_column = f"pre_astig"
    pre_axis_column = f"pre_axis"
    post_astig_column = f"post_astig"
    post_axis_column = f"post_axis"
    required = [
        pre_astig_column,
        pre_axis_column,
        post_astig_column,
        post_axis_column,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError("输入表缺少 SIA 必需字段: " + ", ".join(missing))

    output = df.copy()

    pre_c = pd.to_numeric(output[pre_astig_column], errors="coerce")
    post_c = pd.to_numeric(output[post_astig_column], errors="coerce")
    pre_axis = pd.to_numeric(output[pre_axis_column], errors="coerce")
    post_axis = pd.to_numeric(output[post_axis_column], errors="coerce")

    pre_x, pre_y = astig_to_vector(
        pre_c, pre_axis, axis_convention=axis_convention
    )
    post_x, post_y = astig_to_vector(
        post_c, post_axis, axis_convention=axis_convention
    )
    delta_x = post_x - pre_x
    delta_y = post_y - pre_y

    output["pre_front_X"] = pre_x
    output["pre_front_Y"] = pre_y
    output["post_front_X"] = post_x
    output["post_front_Y"] = post_y
    output["delta_front_X"] = delta_x
    output["delta_front_Y"] = delta_y

    # X and Y are twice the usual J0/J45 power-vector components.
    output["pre_front_J0"] = pre_x / 2.0
    output["pre_front_J45"] = pre_y / 2.0
    output["post_front_J0"] = post_x / 2.0
    output["post_front_J45"] = post_y / 2.0
    output["delta_front_J0"] = delta_x / 2.0
    output["delta_front_J45"] = delta_y / 2.0

    output["front_SIA_D"] = np.hypot(delta_x, delta_y)
    output["front_SIA_axis_deg"] = (
        np.rad2deg(np.arctan2(delta_y, delta_x)) / 2.0
    ) % 180.0

    pre_time = make_timestamp_columns(output, pre_prefix)
    post_time = make_timestamp_columns(output, post_prefix)
    output["pre_datetime"] = pre_time
    output["post_datetime"] = post_time

    chronology_known = pre_time.notna() & post_time.notna()
    chronology_valid = pd.Series(pd.NA, index=output.index, dtype="boolean")
    chronology_valid.loc[chronology_known] = (
        post_time.loc[chronology_known] > pre_time.loc[chronology_known]
    )
    output["chronology_valid"] = chronology_valid

    pre_qs_ok, pre_qs_available = _quality_ok(output, pre_prefix)
    post_qs_ok, post_qs_available = _quality_ok(output, post_prefix)
    output["pre_qs_ok"] = pre_qs_ok
    output["post_qs_ok"] = post_qs_ok
    output["qs_columns_available"] = pre_qs_available and post_qs_available

    numeric_valid = (
        pre_c.notna()
        & post_c.notna()
        & pre_axis.notna()
        & post_axis.notna()
        & pre_c.ge(0)
        & post_c.ge(0)
        & pre_axis.ge(0)
        & post_axis.ge(0)
        & pre_axis.lt(180)
        & post_axis.lt(180)
    )
    # SIA is considered strictly valid only when temporal order is known and
    # correct.  QS flags are also required when the workbook provides them.
    quality_valid = pre_qs_ok & post_qs_ok
    quality_valid = quality_valid.astype("boolean")
    strict_valid = numeric_valid & chronology_valid.fillna(False)
    if pre_qs_available and post_qs_available:
        strict_valid &= quality_valid
    output["sia_valid"] = strict_valid.astype("boolean")

    reasons: list[str] = []
    for row_index in output.index:
        row_reasons: list[str] = []
        if not bool(numeric_valid.loc[row_index]):
            row_reasons.append("散光度数或轴位缺失/越界")
        if pd.isna(chronology_valid.loc[row_index]):
            row_reasons.append("缺少可解析的术前/术后时间")
        elif not bool(chronology_valid.loc[row_index]):
            row_reasons.append("术后时间不晚于术前时间")
        if pre_qs_available and not bool(pre_qs_ok.loc[row_index]):
            row_reasons.append("术前QS非OK")
        if post_qs_available and not bool(post_qs_ok.loc[row_index]):
            row_reasons.append("术后QS非OK")
        reasons.append("; ".join(row_reasons) if row_reasons else "")
    output["sia_invalid_reason"] = reasons

    return output


def make_summary(
    result: pd.DataFrame,
    *,
    axis_convention: str,
    input_path: Path,
) -> pd.DataFrame:
    """Create a compact audit summary for the output workbook."""
    valid = result["sia_valid"].fillna(False).astype(bool)
    sia = pd.to_numeric(result["front_SIA_D"], errors="coerce")
    summary = [
        ("input_file", str(input_path)),
        ("row_count", len(result)),
        ("sia_numeric_count", int(sia.notna().sum())),
        ("sia_valid_count", int(valid.sum())),
        ("sia_invalid_count", int((~valid).sum())),
        ("pre_qs_ok_count", int(result["pre_qs_ok"].fillna(False).sum())),
        ("post_qs_ok_count", int(result["post_qs_ok"].fillna(False).sum())),
        (
            "both_qs_ok_count",
            int(
                (
                    result["pre_qs_ok"].fillna(False)
                    & result["post_qs_ok"].fillna(False)
                ).sum()
            ),
        ),
        (
            "chronology_valid_count",
            int(result["chronology_valid"].fillna(False).sum()),
        ),
        ("front_SIA_mean_D", float(sia.mean()) if sia.notna().any() else np.nan),
        (
            "front_SIA_median_D",
            float(sia.median()) if sia.notna().any() else np.nan,
        ),
        ("front_SIA_min_D", float(sia.min()) if sia.notna().any() else np.nan),
        ("front_SIA_max_D", float(sia.max()) if sia.notna().any() else np.nan),
        ("axis_convention", axis_convention),
        (
            "method",
            "X=C*cos(2*axis), Y=C*sin(2*axis), SIA=hypot(post-pre)",
        ),
        (
            "note",
            "front_SIA is anterior corneal SIA; it is not total corneal or refractive SIA",
        ),
    ]
    return pd.DataFrame(summary, columns=["item", "value"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算配对 Pentacam 数据的前表面角膜 SIA"
    )
    parser.add_argument("--input", required=True, help="输入 paired_dataset.xlsx")
    parser.add_argument(
        "--output",
        help="输出 Excel 路径，默认在输入文件名后加 _sia",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="输入工作表名称或索引，默认第一个工作表",
    )
    parser.add_argument(
        "--axis-convention",
        choices=["flat", "steep"],
        default="flat",
        help="输入 Axis 是平坦子午线还是陡峭子午线，默认 flat",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="仅导出 sia_valid=True 的记录；默认保留所有记录并标记异常",
    )
    return parser


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_sia.xlsx")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("输出文件不能覆盖输入文件，请使用不同的 --output 路径")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    try:
        sheet: int | str = int(args.sheet)
    except (TypeError, ValueError):
        sheet = args.sheet

    source = pd.read_excel(input_path, sheet_name=sheet)
    result = calculate_sia(
        source,
        axis_convention=args.axis_convention,
    )
    if args.valid_only:
        result = result[result["sia_valid"].fillna(False)].copy()

    output_path = resolve_output_path(input_path, args.output)
    summary = make_summary(
        calculate_sia(source, axis_convention=args.axis_convention),
        axis_convention=args.axis_convention,
        input_path=input_path,
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="sia_data", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    output_path = run(args)
    print(f"SIA 计算完成: {output_path}")


if __name__ == "__main__":
    main()
