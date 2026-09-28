#!/usr/bin/env python3
"""按文件名时间戳将未分离的 Pentacam CUR 数据分成 pre/post。

支持的典型文件名：
    surname_firstname_eye_DDMMYYYY_HHMMSS_CUR.CSV

规则：
    1. 按「姓氏 + 名称 + 眼别」分组；
    2. 每组按检查日期和时间升序排列；
    3. 最早一份复制到 pre，最晚一份复制到 post；
    4. 中间检查不复制到 pre/post，但写入报告；
    5. 同组少于两份、时间戳无法解析或时间相同的记录不自动配对。

默认只复制文件，不删除或移动原始文件。运行结束后生成：
    output/pre/              术前文件
    output/post/             术后文件
    output/split_report.csv  所有文件的分组和处理结果
    output/paired.csv        成功配对的 pre/post 文件

示例：
    python split_pre_post_by_time.py \
        --input-dir data/new_batch \
        --output-dir data/new_batch_split
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FileRecord:
    path: Path
    surname: str
    given_name: str
    eye: str
    date_text: str
    time_text: str
    timestamp: datetime | None

    @property
    def patient_eye_key(self) -> str:
        return "_".join((normalize(self.surname), normalize(self.given_name), normalize(self.eye)))


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def parse_timestamp(date_text: str, time_text: str) -> datetime | None:
    """支持 DDMMYYYY/HHMMSS 和 YYYYMMDD/HHMMSS 两种常见格式。"""
    date_text = re.sub(r"\D", "", date_text)
    time_text = re.sub(r"\D", "", time_text)
    candidates = (
        f"{date_text}{time_text}",
    )
    for value in candidates:
        for fmt in ("%d%m%Y%H%M%S", "%Y%m%d%H%M%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    return None


def parse_filename(path: Path) -> FileRecord | None:
    """解析 surname_firstname_eye_date_time_CUR.CSV。

    名称允许包含下划线，因此从末尾反向解析 eye/date/time/CUR。
    """
    parts = path.stem.split("_")
    if len(parts) < 6 or parts[-1].upper() != "CUR":
        return None
    surname = parts[0]
    given_name = "_".join(parts[1:-4])
    eye, date_text, time_text = parts[-4], parts[-3], parts[-2]
    return FileRecord(
        path=path,
        surname=surname,
        given_name=given_name,
        eye=eye,
        date_text=date_text,
        time_text=time_text,
        timestamp=parse_timestamp(date_text, time_text),
    )


def discover_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and "cur" in p.stem.lower()
    )


def copy_or_move(source: Path, destination: Path, operation: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"目标文件已存在，拒绝覆盖: {destination}")
    if operation == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def process(input_dir: str | Path, output_dir: str | Path, operation: str = "copy") -> dict[str, int]:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    files = discover_files(input_dir)
    if not files:
        raise FileNotFoundError(f"输入目录中没有 *CUR.CSV 文件: {input_dir}")

    parsed: list[FileRecord] = []
    report_rows: list[dict[str, str]] = []
    for path in files:
        record = parse_filename(path)
        if record is None:
            report_rows.append({
                "source_file": str(path),
                "patient_eye_key": "",
                "surname": "",
                "given_name": "",
                "eye": "",
                "date": "",
                "time": "",
                "timestamp": "",
                "role": "unparsed",
                "status": "filename_format_error",
                "note": "文件名不符合 *_*_EYE_DATE_TIME_CUR.CSV 格式",
            })
        else:
            parsed.append(record)

    groups: dict[str, list[FileRecord]] = {}
    for record in parsed:
        groups.setdefault(record.patient_eye_key, []).append(record)

    paired_rows: list[dict[str, str]] = []
    counts = {
        "input_files": len(files),
        "parsed_files": len(parsed),
        "paired_groups": 0,
        "pre_files": 0,
        "post_files": 0,
        "unpaired_groups": 0,
        "ambiguous_groups": 0,
        "intermediate_files": 0,
    }

    for key, records in sorted(groups.items()):
        invalid_time = [r for r in records if r.timestamp is None]
        if invalid_time:
            counts["ambiguous_groups"] += 1
            for record in records:
                report_rows.append(record_row(record, "unassigned", "timestamp_parse_error"))
            continue

        ordered = sorted(records, key=lambda r: (r.timestamp, r.path.name))
        timestamps = [r.timestamp for r in ordered]
        if len(ordered) < 2:
            counts["unpaired_groups"] += 1
            for record in ordered:
                report_rows.append(record_row(record, "unassigned", "only_one_record"))
            continue
        if len(set(timestamps)) != len(timestamps):
            counts["ambiguous_groups"] += 1
            for record in ordered:
                report_rows.append(record_row(record, "unassigned", "duplicate_timestamp"))
            continue

        pre = ordered[0]
        post = ordered[-1]
        middle = ordered[1:-1]
        counts["paired_groups"] += 1
        counts["pre_files"] += 1
        counts["post_files"] += 1
        counts["intermediate_files"] += len(middle)

        pre_destination = output_dir / "pre" / pre.path.name
        post_destination = output_dir / "post" / post.path.name
        copy_or_move(pre.path, pre_destination, operation)
        copy_or_move(post.path, post_destination, operation)
        report_rows.append(record_row(pre, "pre", "paired"))
        report_rows.append(record_row(post, "post", "paired"))
        for record in middle:
            report_rows.append(record_row(record, "intermediate", "not_exported"))

        paired_rows.append({
            "patient_eye_key": key,
            "surname": pre.surname,
            "given_name": pre.given_name,
            "eye": pre.eye,
            "pre_file": str(pre_destination),
            "post_file": str(post_destination),
            "pre_timestamp": pre.timestamp.isoformat(sep=" "),
            "post_timestamp": post.timestamp.isoformat(sep=" "),
            "intermediate_count": str(len(middle)),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "split_report.csv", report_rows)
    write_csv(output_dir / "paired.csv", paired_rows)
    return counts


def record_row(record: FileRecord, role: str, status: str) -> dict[str, str]:
    return {
        "source_file": str(record.path),
        "patient_eye_key": record.patient_eye_key,
        "surname": record.surname,
        "given_name": record.given_name,
        "eye": record.eye,
        "date": record.date_text,
        "time": record.time_text,
        "timestamp": record.timestamp.isoformat(sep=" ") if record.timestamp else "",
        "role": role,
        "status": status,
        "note": "",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file", "patient_eye_key", "surname", "given_name", "eye",
        "date", "time", "timestamp", "role", "status", "note",
    ]
    if path.name == "paired.csv":
        fieldnames = [
            "patient_eye_key", "surname", "given_name", "eye", "pre_file",
            "post_file", "pre_timestamp", "post_timestamp", "intermediate_count",
        ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="按时间戳分离 Pentacam CUR 数据的 pre/post")
    parser.add_argument("--input-dir", required=True, help="未分离 CUR 文件目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--operation", choices=["copy", "move"], default="copy", help="默认复制，使用 move 会移动原始文件")
    args = parser.parse_args()

    counts = process(args.input_dir, args.output_dir, args.operation)
    print("分离完成")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"术前目录: {Path(args.output_dir).resolve() / 'pre'}")
    print(f"术后目录: {Path(args.output_dir).resolve() / 'post'}")
    print(f"配对报告: {Path(args.output_dir).resolve() / 'paired.csv'}")
    print(f"完整报告: {Path(args.output_dir).resolve() / 'split_report.csv'}")


if __name__ == "__main__":
    main()
