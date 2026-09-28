import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ------------------------------
# 解析文件名得到附加字段
# ------------------------------
def parse_filename(file_path):
    """
    文件名格式: 姓氏_名称_眼别_检查日期_检查时间_CUR.CSV
    返回 dict
    """
    name = Path(file_path).stem
    parts = name.split("_")
    if len(parts) < 6:
        # 不符合格式返回空
        return {}
    return {
        "姓氏": parts[0],
        "名称": parts[1],
        "眼别": parts[2],
        "检查日期": parts[3],
        "检查时间": parts[4],
    }


def parse_sections_with_field_check(csv_path, encoding="latin1"):
    sections_data = {}
    current_section = None
    skip_coords_sections = ["PUPIL", "CORNEA EDGE"]

    with open(csv_path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 段落识别
            m_sec = re.match(r"\[(.+?)\]", line)
            if m_sec:
                current_section = m_sec.group(1).strip().upper()
                sections_data[current_section] = {}
                continue

            if current_section:
                # 拆分字段和值
                parts = re.split(r"[;\t]", line)
                if len(parts) < 2:
                    parts = line.split(None, 1)
                if len(parts) >= 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    if current_section in skip_coords_sections:
                        nums = re.findall(r"-?\d+\.?\d*", value)
                        if key.replace("-","").replace(".","").isdigit() and len(nums) >= 1:
                            continue
                    try:
                        value = float(value)
                    except:
                        pass
                    sections_data[current_section][key] = value
                else:
                    sections_data[current_section][parts[0].strip()] = None
    return sections_data

def process_cur_files(data_dir, output_file="labels.xlsx"):
    data_dir = Path(data_dir)
    # 筛选CUR文件
    cur_files = [f for f in data_dir.glob("*.CSV") if "CUR" in f.name.upper()]

    all_records = []

    for fpath in cur_files:
        # 文件名解析
        meta = parse_filename(fpath)

        # CSV段落字段解析
        sections = parse_sections_with_field_check(fpath)

        # 合并字典
        record = meta.copy()
        for sec_name, fields in sections.items():
            for k,v in fields.items():
                col_name = f"{sec_name}_{k}"
                record[col_name] = v

        all_records.append(record)

    # 汇总 DataFrame
    df = pd.DataFrame(all_records)

    # 保存Excel
    df.to_excel(output_file, index=False)
    print(f"完成汇总，共 {len(df)} 条记录，保存至 {output_file}")


if __name__ == "__main__":
    process_cur_files(data_dir="./data/batch2/batch2_post", output_file="./data/post_op_2.xlsx")
    process_cur_files(data_dir="./data/batch2/batch2_pre", output_file="./data/pre_op_2.xlsx")