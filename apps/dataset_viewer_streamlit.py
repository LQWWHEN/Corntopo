#!/usr/bin/env python3
"""CornTopo Streamlit 数据查看器。

运行：
    streamlit run dataset_viewer_streamlit.py

支持配对数据和未配对目录两种模式，复用 dataset_viewer.py 的数据读取、
矩阵解析和 IGNORE_ 软删除逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    from apps.dataset_viewer import ViewerState
except ModuleNotFoundError:
    from dataset_viewer import ViewerState


DEFAULT_IGNORE_PREFIX = "IGNORE_"


def matrix_to_image(values: list[list[float | None]], vmin: float, vmax: float) -> np.ndarray:
    """将矩阵转换为可供 st.image 显示的 RGB 数组。"""
    from matplotlib import colormaps

    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    normalized = np.zeros_like(array, dtype=float)
    normalized[valid] = (array[valid] - vmin) / (vmax - vmin or 1.0)
    rgb = (
        colormaps["turbo"](np.clip(normalized, 0.0, 1.0))[..., :3] * 255
    ).astype(np.uint8)
    rgb[~valid] = np.array([15, 15, 15], dtype=np.uint8)
    return np.flipud(rgb)


def clear_viewer() -> None:
    st.session_state.pop("viewer", None)
    st.session_state.pop("viewer_config", None)
    st.session_state.pop("sample_id", None)
    st.session_state.pop("matrix_name", None)


def make_viewer(
    mode: str,
    paired_excel: str,
    dataset_dir: str,
    pre_dir: str,
    post_dir: str,
    ignore_prefix: str,
) -> ViewerState:
    config = (mode, paired_excel, dataset_dir, pre_dir, post_dir, ignore_prefix)
    if st.session_state.get("viewer_config") != config:
        clear_viewer()
        if mode == "配对数据":
            viewer = ViewerState(
                paired_excel,
                None,
                pre_dir or None,
                post_dir or None,
                ignore_prefix,
            )
        else:
            viewer = ViewerState(
                None,
                dataset_dir,
                None,
                None,
                ignore_prefix,
            )
        st.session_state.viewer = viewer
        st.session_state.viewer_config = config
    return st.session_state.viewer


def sample_label(sample: dict[str, Any]) -> str:
    name = " ".join(
        part for part in (sample["surname"], sample["given_name"], sample["eye"]) if part
    )
    timestamp = sample.get("post_timestamp") or sample.get("pre_timestamp") or ""
    status = " [已忽略]" if sample["ignored"] else ""
    return f'{sample["id"]} · {name or sample["key"]} · {timestamp}{status}'


def main() -> None:
    st.set_page_config(
        page_title="CornTopo 数据查看器",
        page_icon="👁",
        layout="wide",
    )
    st.title("CornTopo 数据查看器")
    st.caption("Streamlit 前端 · 支持配对查看、未配对浏览和 IGNORE_ 软删除")

    with st.sidebar:
        st.header("数据源")
        mode = st.radio("数据模式", ["配对数据", "未配对目录"])
        if mode == "配对数据":
            paired_excel = st.text_input("paired_dataset.xlsx 路径")
            pre_dir = st.text_input("术前目录")
            post_dir = st.text_input("术后目录")
            dataset_dir = ""
        else:
            dataset_dir = st.text_input("未配对 CUR 数据目录")
            paired_excel = ""
            pre_dir = ""
            post_dir = ""

        ignore_prefix = st.text_input("软删除前缀", value=DEFAULT_IGNORE_PREFIX)
        show_ignored = st.checkbox("显示已软删除数据")
        search = st.text_input("搜索患者、眼别或文件名")

    required_path = paired_excel if mode == "配对数据" else dataset_dir
    if not required_path:
        st.info("请在左侧填写数据路径。")
        st.stop()

    try:
        viewer = make_viewer(
            mode,
            paired_excel,
            dataset_dir,
            pre_dir,
            post_dir,
            ignore_prefix,
        )
    except Exception as exc:
        st.error(f"加载数据失败：{exc}")
        st.stop()

    samples = viewer.samples(search, show_ignored)
    st.sidebar.metric("当前样本数", len(samples))
    if not samples:
        st.warning("没有符合条件的数据。可以尝试清空搜索，或勾选显示已软删除数据。")
        st.stop()

    sample_ids = [sample["id"] for sample in samples]
    if st.session_state.get("sample_id") not in sample_ids:
        st.session_state.sample_id = sample_ids[0]
    selected_id = st.selectbox(
        "选择样本",
        sample_ids,
        format_func=lambda value: sample_label(
            next(x for x in samples if x["id"] == value)
        ),
        key="sample_id",
    )
    current = next(sample for sample in samples if sample["id"] == selected_id)

    matrix_names = viewer.matrix_names(selected_id)
    if not matrix_names:
        st.error("当前样本没有可读取的矩阵。")
        st.stop()
    if st.session_state.get("matrix_name") not in matrix_names:
        st.session_state.matrix_name = matrix_names[0]
    matrix_name = st.selectbox("矩阵", matrix_names, key="matrix_name")

    try:
        detail = viewer.detail(selected_id, matrix_name)
    except Exception as exc:
        st.error(f"读取矩阵失败：{exc}")
        st.stop()

    is_unpaired = not detail["post_file"]
    columns = st.columns(4)
    columns[0].metric("患者", f'{current["surname"]} {current["given_name"]}')
    columns[1].metric("眼别", current["eye"])
    columns[2].metric("Pre 时间", detail.get("pre_timestamp") or "未知")
    columns[3].metric(
        "Post 时间",
        "不适用" if is_unpaired else (detail.get("post_timestamp") or "未知"),
    )

    st.write(f'矩阵尺寸：{detail["shape"][0]} × {detail["shape"][1]}')
    st.caption(f'Pre 文件：{detail["pre_file"]}')
    if not is_unpaired:
        st.caption(f'Post 文件：{detail["post_file"]}')

    image_columns = st.columns(3)
    with image_columns[0]:
        st.subheader("当前文件" if is_unpaired else "Pre-op")
        st.image(
            matrix_to_image(detail["pre"], detail["vmin"], detail["vmax"]),
            caption=f'范围：{detail["vmin"]:.3f} ～ {detail["vmax"]:.3f}',
            use_container_width=True,
        )
    with image_columns[1]:
        st.subheader("Post-op" if not is_unpaired else "Post-op（不适用）")
        if is_unpaired:
            st.info("未配对数据没有 Post-op。")
        else:
            st.image(
                matrix_to_image(detail["post"], detail["vmin"], detail["vmax"]),
                caption=f'范围：{detail["vmin"]:.3f} ～ {detail["vmax"]:.3f}',
                use_container_width=True,
            )
    with image_columns[2]:
        st.subheader("绝对差异" if not is_unpaired else "绝对差异（不适用）")
        if is_unpaired:
            st.info("未配对数据无法计算差异。")
        else:
            st.image(
                matrix_to_image(detail["difference"], 0.0, detail["diff_max"]),
                caption=f'范围：0.000 ～ {detail["diff_max"]:.3f}',
                use_container_width=True,
            )

    st.divider()
    action = (
        "恢复当前文件"
        if current["ignored"] and is_unpaired
        else "恢复当前配对"
        if current["ignored"]
        else "软删除当前文件"
        if is_unpaired
        else "软删除当前配对"
    )
    if st.button(action, type="primary" if current["ignored"] else "secondary"):
        try:
            viewer.set_ignored(selected_id, not current["ignored"])
            st.success("操作完成。")
            st.rerun()
        except Exception as exc:
            st.error(f"操作失败：{exc}")

    with st.expander("查看当前记录信息"):
        st.dataframe(pd.DataFrame([current]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
