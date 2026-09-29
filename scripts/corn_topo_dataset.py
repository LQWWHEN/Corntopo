#!/usr/bin/env python3
"""CornTopo CUR 数据集处理工具。

功能：解析 Pentacam CUR 文件、批量生成元数据、严格配对术前/术后样本，
计算前表面角膜 SIA 和术前/术后年龄，并导出原始矩阵、曲率矩阵、灰度图
以及统一/局部色标热力图。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_CURVATURE_MATRICES = {
    "FRONT", "BACK", "TANGENTIAL FRONT", "TANGENTIAL BACK",
}

# 软删除/忽略标记。示例：IGNORE_zhao_baoying_OS_15112024_103711_CUR.CSV
# 只检查文件名开头，文件名中间出现 IGNORE_ 不会被跳过。
DEFAULT_IGNORE_PREFIX = "IGNORE_"


def _is_number(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _to_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _parse_date_value(value: Any) -> pd.Timestamp | pd.NaT:
    """解析常见的 Pentacam/Excel 日期格式。"""
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number) and 1 <= number <= 60000 and number < 100000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(number, unit="D")
        text = str(int(number)) if np.isfinite(number) and number.is_integer() else str(value).strip()
    else:
        text = str(value).strip()
    if not text:
        return pd.NaT
    if text.isdigit():
        text = text.zfill(8)
        if len(text) == 8:
            if 1900 <= int(text[:4]) <= 2100:
                parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
                if not pd.isna(parsed):
                    return parsed
            return pd.to_datetime(text, format="%d%m%Y", errors="coerce")
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _first_nonempty(row: pd.Series, columns: Iterable[str]) -> Any:
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        return value
    return None


def _age_in_years(birth_date: Any, examination_date: Any) -> float:
    birth = _parse_date_value(birth_date)
    examination = _parse_date_value(examination_date)
    if pd.isna(birth) or pd.isna(examination) or examination < birth:
        return float("nan")
    return float(
        examination.year
        - birth.year
        - ((examination.month, examination.day) < (birth.month, birth.day))
    )


def _calculate_front_sia_and_age(
    merged: pd.DataFrame,
    *,
    axis_convention: str = "flat",
) -> pd.DataFrame:
    """添加前表面 SIA、质量标记和术前/术后周岁。"""
    required = [
        "pre_GENERAL DATA_Cornea Front Astig",
        "pre_GENERAL DATA_Cornea Front Axis",
        "post_GENERAL DATA_Cornea Front Astig",
        "post_GENERAL DATA_Cornea Front Axis",
    ]
    missing = [column for column in required if column not in merged.columns]
    if missing:
        raise KeyError("配对表缺少 SIA 必需字段: " + ", ".join(missing))
    if axis_convention not in {"flat", "steep"}:
        raise ValueError(f"未知轴位约定: {axis_convention}")

    output = merged.copy()
    pre_c = pd.to_numeric(output[required[0]], errors="coerce")
    pre_axis = pd.to_numeric(output[required[1]], errors="coerce")
    post_c = pd.to_numeric(output[required[2]], errors="coerce")
    post_axis = pd.to_numeric(output[required[3]], errors="coerce")
    if axis_convention == "steep":
        pre_axis = (pre_axis + 90.0) % 180.0
        post_axis = (post_axis + 90.0) % 180.0

    pre_theta = np.deg2rad(pre_axis)
    post_theta = np.deg2rad(post_axis)
    pre_j0 = pre_c * np.cos(2.0 * pre_theta) / 2.0
    pre_j45 = pre_c * np.sin(2.0 * pre_theta) / 2.0
    post_j0 = post_c * np.cos(2.0 * post_theta) / 2.0
    post_j45 = post_c * np.sin(2.0 * post_theta) / 2.0
    delta_j0 = post_j0 - pre_j0
    delta_j45 = post_j45 - pre_j45

    # J0/J45 是半柱镜尺度，X/Y 是完整柱镜尺度。
    output["pre_front_J0"] = pre_j0
    output["pre_front_J45"] = pre_j45
    output["post_front_J0"] = post_j0
    output["post_front_J45"] = post_j45
    output["delta_front_J0"] = delta_j0
    output["delta_front_J45"] = delta_j45
    output["pre_front_X"] = 2.0 * pre_j0
    output["pre_front_Y"] = 2.0 * pre_j45
    output["post_front_X"] = 2.0 * post_j0
    output["post_front_Y"] = 2.0 * post_j45
    output["delta_front_X"] = 2.0 * delta_j0
    output["delta_front_Y"] = 2.0 * delta_j45
    output["front_SIA_J_norm"] = np.hypot(delta_j0, delta_j45)
    output["front_SIA_D"] = 2.0 * output["front_SIA_J_norm"]
    output["front_SIA_axis_deg"] = (
        np.rad2deg(np.arctan2(delta_j45, delta_j0)) / 2.0
    ) % 180.0
    output["SIA"] = output["front_SIA_D"]
    output["Pre J0"] = pre_j0
    output["Pre 45"] = pre_j45
    output["Post J0"] = post_j0
    output["Post 45"] = post_j45

    pre_exam = output.apply(
        lambda row: _parse_date_value(_first_nonempty(
            row, ["pre_EXAMINATION_Date", "pre_检查日期"]
        )), axis=1,
    )
    post_exam = output.apply(
        lambda row: _parse_date_value(_first_nonempty(
            row, ["post_EXAMINATION_Date", "post_检查日期"]
        )), axis=1,
    )
    pre_birth = output.apply(
        lambda row: _parse_date_value(_first_nonempty(
            row, ["pre_PATIENT_DateOfBirth", "post_PATIENT_DateOfBirth"]
        )), axis=1,
    )
    post_birth = output.apply(
        lambda row: _parse_date_value(_first_nonempty(
            row, ["post_PATIENT_DateOfBirth", "pre_PATIENT_DateOfBirth"]
        )), axis=1,
    )
    output["pre_exam_date_used"] = pre_exam
    output["post_exam_date_used"] = post_exam
    output["pre_birth_date_used"] = pre_birth
    output["post_birth_date_used"] = post_birth
    output["pre_age_years"] = [
        _age_in_years(birth, exam) for birth, exam in zip(pre_birth, pre_exam)
    ]
    output["post_age_years"] = [
        _age_in_years(birth, exam) for birth, exam in zip(post_birth, post_exam)
    ]
    birth_both_known = pre_birth.notna() & post_birth.notna()
    birth_consistent = pd.Series(pd.NA, index=output.index, dtype="boolean")
    birth_consistent.loc[birth_both_known] = (
        pre_birth.loc[birth_both_known] == post_birth.loc[birth_both_known]
    )
    output["birth_date_consistent"] = birth_consistent

    chronology_known = pre_exam.notna() & post_exam.notna()
    chronology_valid = pd.Series(pd.NA, index=output.index, dtype="boolean")
    chronology_valid.loc[chronology_known] = (
        post_exam.loc[chronology_known] > pre_exam.loc[chronology_known]
    )
    output["chronology_valid"] = chronology_valid

    numeric_valid = (
        pre_c.notna() & post_c.notna() & pre_axis.notna() & post_axis.notna()
        & pre_c.ge(0) & post_c.ge(0)
        & pre_axis.ge(0) & post_axis.ge(0)
        & pre_axis.lt(180) & post_axis.lt(180)
    )
    pre_qs = output.get("pre_GENERAL DATA_QS Status")
    post_qs = output.get("post_GENERAL DATA_QS Status")
    qs_available = pre_qs is not None and post_qs is not None
    if qs_available:
        pre_qs_ok = pre_qs.astype("string").str.strip().eq("OK").fillna(False)
        post_qs_ok = post_qs.astype("string").str.strip().eq("OK").fillna(False)
    else:
        pre_qs_ok = pd.Series(True, index=output.index, dtype="boolean")
        post_qs_ok = pd.Series(True, index=output.index, dtype="boolean")
    output["pre_qs_ok"] = pre_qs_ok
    output["post_qs_ok"] = post_qs_ok
    output["sia_valid"] = (
        numeric_valid & chronology_valid.fillna(False)
        & (pre_qs_ok & post_qs_ok if qs_available else True)
    ).astype("boolean")

    reasons: list[str] = []
    for index in output.index:
        row_reasons: list[str] = []
        if not bool(numeric_valid.loc[index]):
            row_reasons.append("散光度数或轴位缺失/越界")
        if pd.isna(chronology_valid.loc[index]):
            row_reasons.append("缺少可解析的术前/术后检查日期")
        elif not bool(chronology_valid.loc[index]):
            row_reasons.append("术后检查日期不晚于术前检查日期")
        if qs_available and not bool(pre_qs_ok.loc[index]):
            row_reasons.append("术前QS非OK")
        if qs_available and not bool(post_qs_ok.loc[index]):
            row_reasons.append("术后QS非OK")
        reasons.append("; ".join(row_reasons))
    output["sia_invalid_reason"] = reasons
    return output


def parse_filename(path: str | Path) -> dict[str, str]:
    """解析 surname_firstname_eye_date_time_CUR.CSV 文件名。"""
    parts = Path(path).stem.split("_")
    if len(parts) < 6 or parts[-1].upper() != "CUR":
        return {}
    return {
        "姓氏": parts[0],
        "名称": "_".join(parts[1:-4]),
        "眼别": parts[-4],
        "检查日期": parts[-3],
        "检查时间": parts[-2],
    }


def parse_sections(path: str | Path, encoding: str = "latin1") -> dict[str, dict[str, Any]]:
    """解析 [SECTION] 字段，跳过 PUPIL/CORNEA EDGE 中的坐标点表。"""
    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None
    skip_coordinate_sections = {"PUPIL", "CORNEA EDGE"}
    with open(path, "r", encoding=encoding, errors="replace", newline="") as handle:
        for raw in handle:
            line = raw.strip().rstrip(';')
            if not line:
                continue
            section_match = re.match(r"^\[([^]]+)\]", line)
            if section_match:
                current = section_match.group(1).strip().upper()
                sections.setdefault(current, {})
                continue
            if current is None:
                continue
            fields = re.split(r"[;\t]", line, maxsplit=1)
            if len(fields) < 2:
                fields = line.split(None, 1)
            key = fields[0].strip()
            value = fields[1].strip() if len(fields) > 1 else ""
            if current in skip_coordinate_sections and _is_number(key):
                continue
            sections[current][key] = _to_scalar(value)
    return sections


def _read_semicolon_rows(path: str | Path, encoding: str = "latin1") -> list[list[str]]:
    with open(path, "r", encoding=encoding, errors="replace", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    return [row[:-1] if row and row[-1] == "" else row for row in rows]


def parse_stacked_matrices(
    path: str | Path,
    *,
    matrix_count: int | None = 5,
    encoding: str = "latin1",
) -> list[dict[str, Any]]:
    """解析文件开头连续排列的矩阵块。

    矩阵大小优先从 [SYSTEM] 的 Matrixsize X/Y 读取，避免固定写死 141。
    """
    rows = _read_semicolon_rows(path, encoding)
    sections = parse_sections(path, encoding)
    system = sections.get("SYSTEM", {})
    n_x = int(system.get("Matrixsize X") or 141)
    n_y = int(system.get("Matrixsize Y") or 141)
    expected_rows = n_y + 1
    expected_cols = n_x + 1
    section_start = next(
        (i for i, row in enumerate(rows) if row and row[0].strip().startswith("[")),
        len(rows),
    )
    header_indices = [
        i for i in range(section_start)
        if len(rows[i]) >= 3
        and rows[i][0].strip()
        and not _is_number(rows[i][0])
        and i + 1 < section_start
        and _is_number(rows[i + 1][0])
    ]
    if matrix_count is not None:
        header_indices = header_indices[:matrix_count]
    if not header_indices:
        raise ValueError(f"未识别到矩阵块: {path}")

    matrices: list[dict[str, Any]] = []
    for header_row in header_indices:
        block = rows[header_row:header_row + expected_rows]
        if len(block) != expected_rows:
            raise ValueError(f"矩阵 {rows[header_row][0]!r} 行数不足: {path}")
        if any(len(row) < expected_cols for row in block):
            bad = min(len(row) for row in block)
            raise ValueError(
                f"矩阵 {rows[header_row][0]!r} 列数不足，期望至少 {expected_cols}，实际最少 {bad}: {path}"
            )
        name = rows[header_row][0].strip()
        x_coords = np.array([
            float(v) if _is_number(v) else np.nan
            for v in block[0][1:expected_cols]
        ])
        y_coords = np.array([
            float(row[0]) if _is_number(row[0]) else np.nan
            for row in block[1:]
        ])
        values = np.full((n_y, n_x), np.nan, dtype=np.float64)
        for r, row in enumerate(block[1:]):
            for c, value in enumerate(row[1:expected_cols]):
                if _is_number(value):
                    values[r, c] = float(value)
        matrices.append({
            "name": name,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "values": values,
            "curvature": None,
        })
    return matrices


def convert_radius_to_curvature(
    matrices: list[dict[str, Any]],
    target_names: Iterable[str] = DEFAULT_CURVATURE_MATRICES,
    refractive_index: float = 1.3375,
) -> list[dict[str, Any]]:
    """按 K=(n-1)*1000/R 转换半径矩阵；0 和 NaN 转成 NaN。"""
    targets = {str(x).strip().upper() for x in target_names}
    factor = (refractive_index - 1.0) * 1000.0
    for matrix in matrices:
        if matrix["name"].strip().upper() not in targets:
            continue
        radius = np.asarray(matrix["values"], dtype=np.float64)
        curvature = np.full_like(radius, np.nan)
        valid = np.isfinite(radius) & (radius != 0)
        curvature[valid] = factor / radius[valid]
        matrix["curvature"] = curvature
    return matrices


def extract_record(path: str | Path, encoding: str = "latin1") -> dict[str, Any]:
    """提取一份 CUR 的配对元数据，并保留真实来源路径。"""
    path = Path(path).resolve()
    filename_meta = parse_filename(path)
    sections = parse_sections(path, encoding)
    patient = sections.get("PATIENT", {})
    examination = sections.get("EXAMINATION", {})
    record = dict(filename_meta)
    record["姓氏"] = record.get("姓氏") or patient.get("LastName", "")
    record["名称"] = record.get("名称") or patient.get("FirstName", "")
    record["眼别"] = record.get("眼别") or patient.get("Eye", "")
    record["检查日期"] = record.get("检查日期") or examination.get("Date", "")
    record["检查时间"] = record.get("检查时间") or examination.get("Time", "")
    record["source_file"] = str(path)
    for section_name, fields in sections.items():
        for key, value in fields.items():
            record[f"{section_name}_{key}"] = value
    return record


def _has_ignore_prefix(path: str | Path | None, ignore_prefix: str = DEFAULT_IGNORE_PREFIX) -> bool:
    """判断文件名是否以软删除前缀开头，比较时不区分大小写。"""
    if path is None or not ignore_prefix:
        return False
    return Path(path).name.lower().startswith(ignore_prefix.lower())


def discover_cur_files(
    folder: str | Path,
    ignore_prefix: str = DEFAULT_IGNORE_PREFIX,
) -> list[Path]:
    """发现有效 CUR 文件；以 ignore_prefix 开头的文件会被跳过。"""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() == ".csv"
            and "cur" in p.stem.lower()
            and not _has_ignore_prefix(p, ignore_prefix)
        )
    )


def build_metadata_table(
    folder: str | Path,
    encoding: str = "latin1",
    ignore_prefix: str = DEFAULT_IGNORE_PREFIX,
) -> pd.DataFrame:
    records = [
        extract_record(p, encoding)
        for p in discover_cur_files(folder, ignore_prefix)
    ]
    if not records:
        raise FileNotFoundError(f"目录中没有 *CUR.CSV 文件: {folder}")
    return pd.DataFrame(records)


def make_merge_key(row: pd.Series) -> str:
    return "_".join((
        _norm_text(row.get("姓氏")),
        _norm_text(row.get("名称")),
        _norm_text(row.get("眼别")),
    ))


def merge_pre_post(
    pre_dir: str | Path,
    post_dir: str | Path,
    output_file: str | Path,
    *,
    encoding: str = "latin1",
    duplicate_policy: str = "error",
    ignore_prefix: str = DEFAULT_IGNORE_PREFIX,
    axis_convention: str = "flat",
) -> pd.DataFrame:
    """按患者+眼别配对；默认遇到重复键直接报错，避免静默丢样本。"""
    pre = build_metadata_table(pre_dir, encoding, ignore_prefix)
    post = build_metadata_table(post_dir, encoding, ignore_prefix)
    pre["merge_key"] = pre.apply(make_merge_key, axis=1)
    post["merge_key"] = post.apply(make_merge_key, axis=1)
    for label, frame in (("术前", pre), ("术后", post)):
        duplicates = frame[frame["merge_key"].duplicated(False)]
        if not duplicates.empty and duplicate_policy == "error":
            keys = sorted(duplicates["merge_key"].unique())
            raise ValueError(f"{label}存在重复患者+眼别键，已停止以避免错误配对: {keys}")
        if duplicate_policy == "first":
            frame.drop_duplicates("merge_key", keep="first", inplace=True)

    pre_keys, post_keys = set(pre["merge_key"]), set(post["merge_key"])
    matched = sorted(pre_keys & post_keys)
    only_pre, only_post = sorted(pre_keys - post_keys), sorted(post_keys - pre_keys)
    if not matched:
        raise ValueError("术前和术后没有成功匹配的患者+眼别")
    print(f"术前文件: {len(pre)}，术后文件: {len(post)}，匹配: {len(matched)}")
    print(f"仅术前: {len(only_pre)}，仅术后: {len(only_post)}")
    if only_pre:
        print("仅术前键:", ", ".join(only_pre))
    if only_post:
        print("仅术后键:", ", ".join(only_post))

    pre = pre[pre["merge_key"].isin(matched)].copy().sort_values("merge_key")
    post = post[post["merge_key"].isin(matched)].copy().sort_values("merge_key")
    pre = pre.rename(columns={
        c: f"pre_{c}" for c in pre.columns
        if c not in {"merge_key", "姓氏", "名称", "眼别"}
    })
    post = post.rename(columns={
        c: f"post_{c}" for c in post.columns
        if c not in {"merge_key", "姓氏", "名称", "眼别"}
    })
    # merge_key 已经经过 strip、空白压缩和 lower 规范化。
    # 不能再把原始的「姓氏/名称/眼别」作为连接字段，否则：
    #   pre:  Zhao_Baoying_OS
    #   post: zhao_baoying_os
    # 虽然 merge_key 相同，但原始字段不同，pandas.merge 会匹配失败。
    # 只用规范化后的 merge_key 连接，并保留术前侧的原始身份字段。
    post_for_merge = post.drop(columns=["姓氏", "名称", "眼别"])
    merged = pre.merge(
        post_for_merge,
        on="merge_key",
        how="inner",
        validate="one_to_one",
    )
    merged.insert(0, "label_index", np.arange(1, len(merged) + 1))
    merged.drop(columns=["merge_key"], inplace=True)
    merged = _calculate_front_sia_and_age(
        merged,
        axis_convention=axis_convention,
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output_file, index=False)
    print(f"已保存配对表: {output_file}，最终样本数: {len(merged)}")
    return merged


def _valid_range(values: np.ndarray) -> tuple[float, float] | None:
    valid = values[np.isfinite(values)]
    return (float(valid.min()), float(valid.max())) if valid.size else None


def compute_global_ranges(
    files: Iterable[Path],
    target_names: set[str],
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, list[float]] = {}
    for path in files:
        matrices = convert_radius_to_curvature(parse_stacked_matrices(path), target_names)
        for matrix in matrices:
            values = matrix["curvature"] if matrix["curvature"] is not None else matrix["values"]
            bounds = _valid_range(values)
            if bounds is None:
                continue
            name = matrix["name"]
            if name not in ranges:
                ranges[name] = [bounds[0], bounds[1]]
            else:
                ranges[name][0] = min(ranges[name][0], bounds[0])
                ranges[name][1] = max(ranges[name][1], bounds[1])
    return {name: (bounds[0], bounds[1]) for name, bounds in ranges.items()}


def save_heatmap(
    values: np.ndarray,
    path: Path,
    vmin: float,
    vmax: float,
    cmap: str = "jet",
) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(values)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="black")
    fig = plt.figure(figsize=(1, 1), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.imshow(
        masked, cmap=cmap_obj, origin="lower",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    fig.savefig(path, dpi=512, pad_inches=0)
    plt.close(fig)


def save_grayscale(values: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt
    bounds = _valid_range(values)
    if bounds is None:
        return
    vmin, vmax = bounds
    normalized = np.zeros_like(values, dtype=float)
    valid = np.isfinite(values)
    normalized[valid] = (values[valid] - vmin) / (vmax - vmin + 1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, normalized, cmap="gray", vmin=0, vmax=1)


def _find_file_by_key(
    folder: Path,
    key: str,
    ignore_prefix: str = DEFAULT_IGNORE_PREFIX,
) -> Path | None:
    for path in discover_cur_files(folder, ignore_prefix):
        meta = parse_filename(path)
        candidate = "_".join((
            _norm_text(meta.get("姓氏")),
            _norm_text(meta.get("名称")),
            _norm_text(meta.get("眼别")),
        ))
        if candidate == key:
            return path.resolve()
    return None


def generate_arrays_and_images(
    merged_excel: str | Path,
    pre_dir: str | Path,
    post_dir: str | Path,
    output_dir: str | Path,
    *,
    matrix_count: int | None = 5,
    target_names: Iterable[str] = DEFAULT_CURVATURE_MATRICES,
    encoding: str = "latin1",
    ignore_prefix: str = DEFAULT_IGNORE_PREFIX,
) -> dict[str, Any]:
    """从配对表生成 raw/*.npy、raw_radius/*.npy 和图像。"""
    df = pd.read_excel(merged_excel)
    output_dir = Path(output_dir)
    target_names = {str(x).strip().upper() for x in target_names}
    all_files = (
        discover_cur_files(pre_dir, ignore_prefix)
        + discover_cur_files(post_dir, ignore_prefix)
    )
    global_ranges = compute_global_ranges(all_files, target_names)
    if not global_ranges:
        raise ValueError("无法为输入文件计算任何有效矩阵范围")

    generated = 0
    missing: list[str] = []
    for _, row in df.iterrows():
        label = int(row["label_index"])
        key = "_".join((
            _norm_text(row["姓氏"]),
            _norm_text(row["名称"]),
            _norm_text(row["眼别"]),
        ))
        pre_value = row.get("pre_source_file", "")
        post_value = row.get("post_source_file", "")
        pre_path = (
            Path(str(pre_value))
            if pd.notna(pre_value) and str(pre_value)
            else _find_file_by_key(Path(pre_dir), key, ignore_prefix)
        )
        post_path = (
            Path(str(post_value))
            if pd.notna(post_value) and str(post_value)
            else _find_file_by_key(Path(post_dir), key, ignore_prefix)
        )
        # 即使旧的 paired Excel 中仍保留了被软删除的源文件路径，生成阶段
        # 也再次检查，保证 IGNORE_ 数据不会被重新读取。
        if _has_ignore_prefix(pre_path, ignore_prefix) or _has_ignore_prefix(post_path, ignore_prefix):
            missing.append(key)
            continue
        if not pre_path or not pre_path.exists() or not post_path or not post_path.exists():
            missing.append(key)
            continue

        for phase, path in (("pre", pre_path), ("post", post_path)):
            matrices = convert_radius_to_curvature(
                parse_stacked_matrices(path, matrix_count=matrix_count, encoding=encoding),
                target_names,
            )
            for matrix in matrices:
                name = matrix["name"]
                values = matrix["curvature"] if matrix["curvature"] is not None else matrix["values"]
                bounds = _valid_range(values)
                if bounds is None or name not in global_ranges:
                    continue
                base = output_dir
                raw_path = base / "raw" / name / f"{label}_{phase}.npy"
                radius_path = base / "raw_radius" / name / f"{label}_{phase}.npy"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                radius_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, values)
                np.save(radius_path, matrix["values"])
                save_heatmap(values, base / "heatmaps_global" / name / f"{label}_{phase}.png", *global_ranges[name])
                save_heatmap(values, base / "heatmaps_local" / name / f"{label}_{phase}.png", *bounds)
                save_grayscale(values, base / "grayscale" / name / f"{label}_{phase}.png")
                generated += 1

    manifest = {
        "samples": int(len(df)),
        "generated_arrays_and_images": generated,
        "missing_pairs": missing,
        "global_ranges": {k: list(v) for k, v in global_ranges.items()},
        "curvature_matrices": sorted(target_names),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成 {generated} 个矩阵数组/图像，清单: {output_dir / 'manifest.json'}")
    if missing:
        print("缺失配对:", ", ".join(missing))
    return manifest


def run_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_table = output_dir / "pre_metadata.xlsx"
    post_table = output_dir / "post_metadata.xlsx"
    total_table = output_dir / "paired_dataset.xlsx"
    build_metadata_table(
        args.pre_dir, args.encoding, args.ignore_prefix
    ).to_excel(pre_table, index=False)
    build_metadata_table(
        args.post_dir, args.encoding, args.ignore_prefix
    ).to_excel(post_table, index=False)
    merge_pre_post(
        args.pre_dir, args.post_dir, total_table,
        encoding=args.encoding,
        duplicate_policy=args.duplicate_policy,
        ignore_prefix=args.ignore_prefix,
        axis_convention=args.axis_convention,
    )
    generate_arrays_and_images(
        total_table, args.pre_dir, args.post_dir, output_dir,
        matrix_count=args.matrix_count,
        encoding=args.encoding,
        ignore_prefix=args.ignore_prefix,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CornTopo Pentacam CUR 数据集处理")
    sub = parser.add_subparsers(dest="command", required=True)

    p_meta = sub.add_parser("metadata", help="批量导出一个目录的元数据")
    p_meta.add_argument("--input-dir", required=True)
    p_meta.add_argument("--output", required=True)
    p_meta.add_argument("--encoding", default="latin1")
    p_meta.add_argument("--ignore-prefix", default=DEFAULT_IGNORE_PREFIX)

    p_merge = sub.add_parser("merge", help="配对术前和术后文件")
    p_merge.add_argument("--pre-dir", required=True)
    p_merge.add_argument("--post-dir", required=True)
    p_merge.add_argument("--output", required=True)
    p_merge.add_argument("--encoding", default="latin1")
    p_merge.add_argument("--duplicate-policy", choices=["error", "first"], default="error")
    p_merge.add_argument("--ignore-prefix", default=DEFAULT_IGNORE_PREFIX)
    p_merge.add_argument(
        "--axis-convention", choices=["flat", "steep"], default="flat",
        help="Cornea Front Axis 的含义，默认按平坦子午线解释",
    )

    p_generate = sub.add_parser("generate", help="根据配对表导出数组和图像")
    p_generate.add_argument("--paired-excel", required=True)
    p_generate.add_argument("--pre-dir", required=True)
    p_generate.add_argument("--post-dir", required=True)
    p_generate.add_argument("--output-dir", required=True)
    p_generate.add_argument("--matrix-count", type=int, default=5)
    p_generate.add_argument("--encoding", default="latin1")
    p_generate.add_argument("--ignore-prefix", default=DEFAULT_IGNORE_PREFIX)

    p_all = sub.add_parser("all", help="执行完整流程")
    p_all.add_argument("--pre-dir", required=True)
    p_all.add_argument("--post-dir", required=True)
    p_all.add_argument("--output-dir", required=True)
    p_all.add_argument("--matrix-count", type=int, default=5)
    p_all.add_argument("--encoding", default="latin1")
    p_all.add_argument("--duplicate-policy", choices=["error", "first"], default="error")
    p_all.add_argument("--ignore-prefix", default=DEFAULT_IGNORE_PREFIX)
    p_all.add_argument(
        "--axis-convention", choices=["flat", "steep"], default="flat",
        help="Cornea Front Axis 的含义，默认按平坦子午线解释",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "metadata":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        build_metadata_table(
            args.input_dir, args.encoding, args.ignore_prefix
        ).to_excel(output, index=False)
    elif args.command == "merge":
        merge_pre_post(
            args.pre_dir, args.post_dir, args.output,
            encoding=args.encoding,
            duplicate_policy=args.duplicate_policy,
            ignore_prefix=args.ignore_prefix,
            axis_convention=args.axis_convention,
        )
    elif args.command == "generate":
        generate_arrays_and_images(
            args.paired_excel, args.pre_dir, args.post_dir, args.output_dir,
            matrix_count=args.matrix_count,
            encoding=args.encoding,
            ignore_prefix=args.ignore_prefix,
        )
    elif args.command == "all":
        run_all(args)


if __name__ == "__main__":
    main()
