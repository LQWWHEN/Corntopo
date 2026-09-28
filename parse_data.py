import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
import re

# ------------------------------
# 1. 解析堆叠矩阵
# ------------------------------
def parse_stacked_csv(csv_path, matrix_count=5, data_rows=141, data_cols=141, encoding="latin1"):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=None, encoding=encoding, dtype=str, sep=None, engine='python')

    block_rows = data_rows + 1
    block_cols = data_cols + 1

    results = []
    for i in range(matrix_count):
        start_row = i * block_rows
        end_row = start_row + block_rows
        block = df.iloc[start_row:end_row, :block_cols].copy()

        if block.shape[0] < block_rows or block.shape[1] < block_cols:
            raise ValueError(f"第 {i+1} 个矩阵块尺寸不足，实际为 {block.shape}")

        matrix_name = block.iat[0, 0]
        x_coords = pd.to_numeric(block.iloc[0, 1:], errors='coerce').to_numpy()
        y_coords = pd.to_numeric(block.iloc[1:, 0], errors='coerce').to_numpy()
        values = block.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        matrix_df = pd.DataFrame(values, index=y_coords, columns=x_coords)

        results.append({
            "name": matrix_name,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "values": values,
            "dataframe": matrix_df,
            "curvature": None
        })

    return results

# ------------------------------
# 2. 曲率转换
# ------------------------------
def convert_radius_to_curvature(matrices, target_names=None):
    for m in matrices:
        name = str(m["name"]).strip()
        if target_names is not None and name not in target_names:
            m["curvature"] = None
            continue

        R = m["values"]
        kappa = np.full_like(R, np.nan, dtype=float)
        mask = (~np.isnan(R)) & (R != 0)
        # 假设角膜折射率 1.3375，单位换算示例
        kappa[mask] = (1.3375 - 1) * 1000 / R[mask]
        kappa[R == 0] = np.nan
        m["curvature"] = kappa

    return matrices



def parse_sections_skip_coordinates(csv_path, encoding="latin1"):
    """
    解析 CSV 文件所有段落字段：
    - [PUPIL] 和 [CORNEA EDGE] 段落只保留非坐标字段
    - 其他段落全部解析
    """
    sections_data = {}
    current_section = None
    skip_coords_sections = ["PUPIL", "CORNEA EDGE"]

    with open(csv_path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 匹配段落名 [XXX]
            m_sec = re.match(r"\[(.+?)\]", line)
            if m_sec:
                current_section = m_sec.group(1).strip().upper()
                sections_data[current_section] = {}
                continue

            if current_section:
                # 尝试分割字段和值
                parts = re.split(r"[;\t]", line)
                if len(parts) < 2:
                    parts = line.split(None, 1)

                if len(parts) >= 2:
                    key, value = parts[0].strip(), parts[1].strip()

                    # 针对[PUPIL]和[CORNEA EDGE]跳过坐标矩阵
                    if current_section in skip_coords_sections:
                        # 检查value是否为纯数字且长度>=2
                        nums = re.findall(r"-?\d+\.?\d*", value)
                        # 如果value里有两个及以上数字且key也是数字 → 坐标行，跳过
                        if key.replace("-","").replace(".","").isdigit() and len(nums) >= 1:
                            continue

                    # 尝试转数字
                    try:
                        value = float(value)
                    except:
                        pass

                    sections_data[current_section][key] = value
                else:
                    # 单独字段保留
                    sections_data[current_section][parts[0].strip()] = None

    return sections_data
# ------------------------------
# 4. 可视化矩阵
# ------------------------------
def visualize_matrices(matrices, cols=3, cmap="jet", figsize=(15, 10), save_path=None, key='values'):
    n = len(matrices)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, m in zip(axes, matrices):
        values = m[key] if m[key] is not None else m['values']
        valid = values[~np.isnan(values)]
        vmin, vmax = (np.min(valid), np.max(valid)) if len(valid) > 0 else (0,1)

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="lightgray")
        masked = np.ma.masked_invalid(values)

        im = ax.imshow(masked, cmap=cmap_obj, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{m['name']}\n[{vmin:.2f},{vmax:.2f}]")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        x_len, y_len = len(m["x_coords"]), len(m["y_coords"])
        x_ticks = np.arange(0, x_len, max(1,x_len//6))
        y_ticks = np.arange(0, y_len, max(1,y_len//6))
        ax.set_xticks(x_ticks); ax.set_yticks(y_ticks)
        ax.set_xticklabels([m["x_coords"][i] for i in x_ticks], rotation=45)
        ax.set_yticklabels([m["y_coords"][i] for i in y_ticks])

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(matrices), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ------------------------------
# 5. 主程序整合
# ------------------------------
if __name__ == "__main__":
    csv_file = "data/hu_yuhua_OS_09022026_111230_CUR.CSV"
    target_names = {"FRONT","BACK","TANGENTIAL FRONT","TANGENTIAL BACK"}

    # 解析矩阵
    matrices = parse_stacked_csv(csv_file)
    matrices = convert_radius_to_curvature(matrices, target_names=target_names)

    # 解析所有段落字段
    sections = parse_sections_skip_coordinates(csv_file)

    # 打印矩阵信息
    print(f"共解析出 {len(matrices)} 个矩阵\n")
    for idx, m in enumerate(matrices, start=1):
        print(f"矩阵 {idx} - {m['name']} - shape: {m['values'].shape}, 空值数: {np.isnan(m['values']).sum()}")

    # 打印 SYSTEM 段落示例
    print("\n[SYSTEM]:")
    print(sections.get("SYSTEM"))

    print("\n[PATIENT]:")
    print(sections.get("PATIENT"))

    print("\n[EXAMINATION]:")
    print(sections.get("EXAMINATION"))

    print("\n[PUPIL]:")
    print(sections.get("PUPIL"))

    print("\n[CORNEA EDGE]:")
    print(sections.get("CORNEA EDGE"))

    print("\n[GENERAL DATA]:")
    print(sections.get("GENERAL DATA"))
    
    print("\n[GENERAL DATA 2]:")
    print(sections.get("GENERAL DATA 2"))

    # 可视化曲率矩阵
    visualize_matrices(matrices, cols=3, cmap="jet", figsize=(16,9), key="curvature")