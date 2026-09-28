import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# 解析 Pentacam 曲率矩阵
# =========================================================
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
    try:
        for i in range(matrix_count):

            start_row = i * block_rows
            end_row = start_row + block_rows
        
            block = df.iloc[start_row:end_row, :block_cols]
            #print(block)
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
    except Exception as e:
        print(csv_path)
        #print(block)

    return matrices


# =========================================================
# 曲率半径 -> 曲率
# =========================================================
def convert_radius_to_curvature(
        matrices,
        target_names=None
):

    for m in matrices:

        name = str(m["name"]).strip()

        # 不转换的矩阵
        if target_names is not None and name not in target_names:
            m["curvature"] = None
            continue

        R = m["values"]

        kappa = np.full_like(
            R,
            np.nan,
            dtype=float
        )

        mask = (~np.isnan(R)) & (R != 0)

        # 曲率转换
        kappa[mask] = (
            (1.3375 - 1) * 1000 / R[mask]
        )

        kappa[R == 0] = np.nan

        m["curvature"] = kappa

    return matrices


# =========================================================
# 计算统一 colorbar
# （不使用百分位）
# =========================================================
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

                ranges[name][0] = min(
                    ranges[name][0],
                    vmin
                )

                ranges[name][1] = max(
                    ranges[name][1],
                    vmax
                )

    return ranges


# =========================================================
# 保存纯热力图
# =========================================================
def save_clean_heatmap(
        values,
        save_path,
        vmin,
        vmax,
        cmap="jet"
):

    masked = np.ma.masked_invalid(values)

    cmap_obj = plt.get_cmap(cmap).copy()

    # NaN区域黑色
    cmap_obj.set_bad(color="black")

    fig = plt.figure(frameon=False)

    fig.set_size_inches(1, 1)

    ax = plt.Axes(
        fig,
        [0., 0., 1., 1.]
    )

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
        dpi=512,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close(fig)


# =========================================================
# 保存灰度图
# =========================================================
def save_grayscale_image(
        values,
        save_path
):

    arr = values.copy()

    mask = np.isnan(arr)

    valid = arr[~mask]

    if len(valid) == 0:
        return

    vmin = np.min(valid)
    vmax = np.max(valid)

    norm = (
        (arr - vmin)
        / (vmax - vmin + 1e-8)
    )

    norm[mask] = 0

    plt.imsave(
        save_path,
        norm,
        cmap='gray',
        vmin=0,
        vmax=1
    )


# =========================================================
# 忽略大小写搜索文件
# =========================================================
def find_matching_file(folder, key):

    files = (
        list(Path(folder).glob("*.CSV"))
        + list(Path(folder).glob("*.csv"))
    )

    key = key.lower()

    for f in files:

        stem_lower = f.stem.lower()

        # 保留原来的病例匹配逻辑
        # 额外增加 _cur 筛选
        if (
            key in stem_lower
            and stem_lower.endswith("_cur")
        ):
            return f

    return None

# =========================================================
# 主函数
# =========================================================
def generate_all_images(

        total_excel,

        pre_dir,
        post_dir,

        heatmap_global_root="heatmaps_global",
        heatmap_local_root="heatmaps_local",
        gray_root="grayscale",
        raw_root = "raw"
):

    df = pd.read_excel(total_excel)

    # =====================================================
    # 先读取所有文件
    # 计算统一colorbar
    # =====================================================

    all_files = (

        list(Path(pre_dir).glob("*CUR*.CSV"))
        + list(Path(pre_dir).glob("*CUR*.csv"))

        + list(Path(post_dir).glob("*CUR*.CSV"))
        + list(Path(post_dir).glob("*CUR*.csv"))

    )

    all_matrices = []

    print("=" * 50)
    print("开始计算统一 colorbar")
    print("=" * 50)

    for fpath in all_files:

        try:

            matrices = parse_stacked_csv(
                fpath
            )

            matrices = [m for m in matrices]

            target_names = [
            "FRONT",
            "BACK",
            "TANGENTIAL FRONT",
            "TANGENTIAL BACK"
        ]

            matrices = convert_radius_to_curvature(
                matrices,
                target_names
            )

            all_matrices.append(
                matrices
            )

        except Exception as e:

            print(f"[读取失败] {fpath}")
            print(e)
    
    global_ranges = compute_global_ranges(
        all_matrices
    )

    print("=" * 50)
    print("统一 colorbar 范围")
    print("=" * 50)

    for k, v in global_ranges.items():

        print(
            f"{k}: "
            f"vmin={v[0]:.4f}, "
            f"vmax={v[1]:.4f}"
        )
    
    # =====================================================
    # 正式生成图像
    # =====================================================

    print("=" * 50)
    print("开始生成图像")
    print("=" * 50)

    for _, row in df.iterrows():

   
            label = row["label_index"]

            surname = str(
                row["姓氏"]
            ).strip().lower()

            name = str(
                row["名称"]
            ).strip().lower()

            eye = str(
                row["眼别"]
            ).strip().lower()

            key = f"{surname}_{name}_{eye}"

            # =============================================
            # 搜索 pre/post 文件
            # =============================================

            pre_path = find_matching_file(
                pre_dir,
                key
            )

            post_path = find_matching_file(
                post_dir,
                key
            )

            if pre_path is None:

                print(f"[缺失术前] {key}")

                continue

            if post_path is None:

                print(f"[缺失术后] {key}")

                continue

            print(f"[处理中] label={label}")

            # =============================================
            # pre/post
            # =============================================

            for phase, fpath in [

                ("pre", pre_path),
                ("post", post_path)

            ]:

                matrices = parse_stacked_csv(
                    fpath
                )

                # 前4个矩阵转换
                target_names = [
            "FRONT",
            "BACK",
            "TANGENTIAL FRONT",
            "TANGENTIAL BACK"
        ]

                matrices = convert_radius_to_curvature(
                    matrices,
                    target_names
                )

                # =========================================
                # 每个矩阵
                # =========================================

                for m in matrices:

                    matrix_name = str(
                        m["name"]
                    ).strip()

                    values = (

                        m["curvature"]

                        if m["curvature"] is not None

                        else m["values"]

                    )

                    valid = values[
                        ~np.isnan(values)
                    ]

                    if len(valid) == 0:
                        continue

                    # =====================================
                    # global colorbar
                    # =====================================

                    global_vmin, global_vmax = (
                        global_ranges[matrix_name]
                    )

                    # =====================================
                    # local colorbar
                    # =====================================

                    local_vmin = np.min(valid)
                    local_vmax = np.max(valid)

                    # =====================================
                    # 创建目录
                    # =====================================

                    global_dir = (
                        Path(heatmap_global_root)
                        / matrix_name
                    )

                    local_dir = (
                        Path(heatmap_local_root)
                        / matrix_name
                    )

                    gray_dir = (
                        Path(gray_root)
                        / matrix_name
                    )

                    global_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    local_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    gray_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    # =====================================
                    # 保存路径
                    # =====================================

                    global_path = (
                        global_dir
                        / f"{label}_{phase}.png"
                    )

                    local_path = (
                        local_dir
                        / f"{label}_{phase}.png"
                    )

                    gray_path = (
                        gray_dir
                        / f"{label}_{phase}.png"
                    )

                    # =====================================
                    # 保存 global 热力图
                    # =====================================

                    save_clean_heatmap(
                        values,
                        global_path,
                        global_vmin,
                        global_vmax
                    )

                    # =====================================
                    # 保存 local 热力图
                    # =====================================

                    save_clean_heatmap(
                        values,
                        local_path,
                        local_vmin,
                        local_vmax
                    )

                    # =====================================
                    # 保存灰度图
                    # =====================================

                    save_grayscale_image(
                        values,
                        gray_path
                    )
                    raw_dir = Path(raw_root) / matrix_name
                    raw_dir.mkdir(parents=True, exist_ok=True)

                    raw_path = raw_dir / f"{label}_{phase}.npy"
                    np.save(raw_path, values)

            print(f"[完成] label={label}")



    print("=" * 50)
    print("全部完成")
    print("=" * 50)


# =========================================================
# main
# =========================================================
if __name__ == "__main__":

    generate_all_images(

        total_excel="data/total_2.xlsx",

        pre_dir="data/batch2/batch2_pre",
        post_dir="data/batch2/batch2_post",

        heatmap_global_root="img_batch2/heatmaps_global",
        heatmap_local_root="img_batch2/heatmaps_local",
        gray_root="img_batch2/grayscale",
        raw_root="img_batch2/raw"
    )