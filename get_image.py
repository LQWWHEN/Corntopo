import pandas as pd
import numpy as np
def convert_radius_to_curvature(matrices, target_names=None):

    for m in matrices:

        name = str(m["name"]).strip()

        if target_names is not None and name not in target_names:
            m["curvature"] = None
            continue

        R = m["values"]

        kappa = np.full_like(R, np.nan, dtype=float)

        mask = (~np.isnan(R)) & (R != 0)

        kappa[mask] = (1.3375 - 1) * 1000 / R[mask]

        kappa[R == 0] = np.nan

        m["curvature"] = kappa

    return matrices

def compute_global_ranges(all_matrices):

    ranges = {}

    for matrices in all_matrices:

        for m in matrices:

            name = str(m["name"]).strip()

            values = (
                m["curvature"]
                if m["curvature"] is not None
                else m["values"]
            )

            valid = values[~np.isnan(values)]

            if len(valid) == 0:
                continue

            vmin = np.min(valid)
            vmax = np.max(valid)

            if name not in ranges:
                ranges[name] = [vmin, vmax]
            else:
                ranges[name][0] = min(ranges[name][0], vmin)
                ranges[name][1] = max(ranges[name][1], vmax)

    return ranges
def save_clean_heatmap(
        values,
        save_path,
        vmin,
        vmax,
        cmap="jet"
):
    """
    保存纯热力图
    无标题/坐标/colorbar
    """

    masked = np.ma.masked_invalid(values)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="black")

    fig = plt.figure(frameon=False)

    fig.set_size_inches(4, 4)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()

    fig.add_axes(ax)

    ax.imshow(
        masked,
        cmap=cmap_obj,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close(fig)
def save_grayscale_image(
        values,
        save_path
):
    """
    保存原始灰度图
    不使用伪彩色
    """

    arr = values.copy()

    mask = np.isnan(arr)

    valid = arr[~mask]

    if len(valid) == 0:
        return

    vmin = np.min(valid)
    vmax = np.max(valid)

    norm = (arr - vmin) / (vmax - vmin + 1e-8)

    norm[mask] = 0

    plt.imsave(
        save_path,
        norm,
        cmap='gray',
        vmin=0,
        vmax=1
    )

def parse_stacked_csv(
        csv_path,
        matrix_count=5,
        data_rows=141,
        data_cols=141,
        encoding="latin1"
):
    """
    解析 Pentacam CUR CSV 前5个矩阵
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        encoding=encoding,
        dtype=str,
        sep=None,
        engine='python'
    )

    block_rows = data_rows + 1
    block_cols = data_cols + 1

    matrices = []

    for i in range(matrix_count):

        start_row = i * block_rows
        end_row = start_row + block_rows

        block = df.iloc[start_row:end_row, :block_cols]

        matrix_name = str(block.iat[0, 0]).strip()

        values = (
            block.iloc[1:, 1:]
            .apply(pd.to_numeric, errors='coerce')
            .to_numpy(dtype=float)
        )

        matrices.append({
            "name": matrix_name,
            "values": values
        })

    return matrices

import matplotlib.pyplot as plt
from pathlib import Path

def save_matrix_heatmap(
        matrix,
        save_path,
        cmap="jet"
):
    """
    保存单个矩阵热力图
    """

    values = matrix["values"]

    # mask NaN
    masked = np.ma.masked_invalid(values)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="black")

    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return

    vmin = np.min(valid)
    vmax = np.max(valid)

    plt.figure(figsize=(6, 6))

    plt.imshow(
        masked,
        cmap=cmap_obj,
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar()

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

def generate_all_images(
        total_excel,
        pre_dir,
        post_dir,
        heatmap_root="heatmaps",
        gray_root="grayscale"
):

    df = pd.read_excel(total_excel)

    pre_files = {
        f.stem.lower(): f
        for f in Path(pre_dir).glob("*CUR*.CSV")
    }

    post_files = {
        f.stem.lower(): f
        for f in Path(post_dir).glob("*CUR*.CSV")
    }

    all_matrices = []

    # 先读取所有矩阵计算统一范围
    for file_dict in [pre_files, post_files]:

        for fpath in file_dict.values():

            matrices = parse_stacked_csv(fpath)

            # 前4个转换曲率
            target_names = [
                matrices[i]["name"]
                for i in range(min(4, len(matrices)))
            ]

            matrices = convert_radius_to_curvature(
                matrices,
                target_names
            )

            all_matrices.append(matrices)

    global_ranges = compute_global_ranges(all_matrices)

    print("统一colorbar范围:")
    print(global_ranges)

    for _, row in df.iterrows():

        label = row["label_index"]

        surname = str(row["姓氏"]).lower()
        name = str(row["名称"]).lower()
        eye = str(row["眼别"])
        key = f"{surname}_{name}_{eye}".lower()
        pre_path = next((f for f in Path(pre_dir).glob("*.CSV") if key in f.stem.lower()), None)
        post_path = next((f for f in Path(post_dir).glob("*.CSV") if key in f.stem.lower()), None)

        if not pre_path.exists():
            print(f"[缺失术前] {pre_path}")
            continue

        if not post_path.exists():
            print(f"[缺失术后] {post_path}")
            continue

        for phase, fpath in [
            ("pre", pre_path),
            ("post", post_path)
        ]:

            matrices = parse_stacked_csv(fpath)

            target_names = [
                matrices[i]["name"]
                for i in range(min(4, len(matrices)))
            ]

            matrices = convert_radius_to_curvature(
                matrices,
                target_names
            )

            for m in matrices:

                matrix_name = str(m["name"]).strip()

                values = (
                    m["curvature"]
                    if m["curvature"] is not None
                    else m["values"]
                )

                heat_dir = Path(heatmap_root) / matrix_name
                heat_dir.mkdir(parents=True, exist_ok=True)

                gray_dir = Path(gray_root) / matrix_name
                gray_dir.mkdir(parents=True, exist_ok=True)

                heat_path = heat_dir / f"{label}_{phase}.png"
                gray_path = gray_dir / f"{label}_{phase}.png"

                vmin, vmax = global_ranges[matrix_name]

                save_clean_heatmap(
                    values,
                    heat_path,
                    vmin,
                    vmax
                )

                save_grayscale_image(
                    values,
                    gray_path
                )

        print(f"[完成] {label}")

if __name__ == "__main__":

    generate_all_images(
        total_excel="data/total.xlsx",
        pre_dir="data/pre-op",
        post_dir="data/post-op",
        heatmap_root="heatmaps",
        gray_root="grayscale"
    )