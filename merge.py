import pandas as pd

import pandas as pd

def merge_pre_post(pre_file, post_file, output_file="total.xlsx"):
    # ------------------------------
    # 读取数据
    # ------------------------------
    df_pre = pd.read_excel(pre_file, dtype=str)
    df_post = pd.read_excel(post_file, dtype=str)

    # 匹配键：患者 + 眼别
    key_cols = ["姓氏", "名称", "眼别"]
    # 去除空格并统一小写
    for col in ["姓氏", "名称"]:
        df_pre[col] = df_pre[col].astype(str).str.strip().str.lower()
        df_post[col] = df_post[col].astype(str).str.strip().str.lower()
    df_pre["眼别"] = df_pre["眼别"].astype(str).str.strip()
    df_post["眼别"] = df_post["眼别"].astype(str).str.strip()

    # 构建唯一key
    df_pre["merge_key"] = (
        df_pre["姓氏"] + "_" +
        df_pre["名称"] + "_" +
        df_pre["眼别"]
    )

    df_post["merge_key"] = (
        df_post["姓氏"] + "_" +
        df_post["名称"] + "_" +
        df_post["眼别"]
    )

    # ------------------------------
    # 检查重复
    # ------------------------------
    pre_dup = df_pre[df_pre["merge_key"].duplicated(keep=False)]
    post_dup = df_post[df_post["merge_key"].duplicated(keep=False)]

    # ------------------------------
    # 集合比较
    # ------------------------------
    pre_keys = set(df_pre["merge_key"])
    post_keys = set(df_post["merge_key"])

    matched_keys = pre_keys & post_keys
    only_pre = pre_keys - post_keys
    only_post = post_keys - pre_keys

    # ------------------------------
    # 输出统计信息
    # ------------------------------
    print("=" * 60)
    print("数据对齐报告")
    print("=" * 60)

    print(f"术前总例数: {len(df_pre)}")
    print(f"术后总例数: {len(df_post)}")
    print(f"成功匹配数: {len(matched_keys)}")
    print(f"仅术前存在(剔除): {len(only_pre)}")
    print(f"仅术后存在(剔除): {len(only_post)}")

    print("\n")

    # ------------------------------
    # 输出重复信息
    # ------------------------------
    if len(pre_dup) > 0:
        print("=" * 60)
        print("术前重复病例:")
        print(pre_dup[["merge_key"]])
        print("\n")

    if len(post_dup) > 0:
        print("=" * 60)
        print("术后重复病例:")
        print(post_dup[["merge_key"]])
        print("\n")

    # ------------------------------
    # 输出剔除名单
    # ------------------------------
    if len(only_pre) > 0:
        print("=" * 60)
        print("仅术前存在（被剔除）:")
        for k in sorted(only_pre):
            print(k)
        print("\n")

    if len(only_post) > 0:
        print("=" * 60)
        print("仅术后存在（被剔除）:")
        for k in sorted(only_post):
            print(k)
        print("\n")

    # ------------------------------
    # 仅保留匹配病例
    # ------------------------------
    df_pre = df_pre[df_pre["merge_key"].isin(matched_keys)].copy()
    df_post = df_post[df_post["merge_key"].isin(matched_keys)].copy()

    # ------------------------------
    # 去除merge_key重复
    # ------------------------------
    df_pre = df_pre.drop_duplicates("merge_key")
    df_post = df_post.drop_duplicates("merge_key")

    # ------------------------------
    # 添加前后缀
    # ------------------------------
    pre_cols = [
        c for c in df_pre.columns
        if c not in key_cols + ["merge_key"]
    ]

    post_cols = [
        c for c in df_post.columns
        if c not in key_cols + ["merge_key"]
    ]

    df_pre = df_pre.rename(
        columns={c: f"pre_{c}" for c in pre_cols}
    )

    df_post = df_post.rename(
        columns={c: f"post_{c}" for c in post_cols}
    )

    # ------------------------------
    # 合并
    # ------------------------------
    df_merged = pd.merge(
        df_pre,
        df_post,
        on=key_cols + ["merge_key"],
        how="inner"
    )

    # ------------------------------
    # 分配label索引
    # ------------------------------
    df_merged.insert(
        0,
        "label_index",
        range(1, len(df_merged) + 1)
    )

    # ------------------------------
    # 删除merge_key
    # ------------------------------
    df_merged = df_merged.drop(columns=["merge_key"])

    # ------------------------------
    # 保存
    # ------------------------------
    df_merged.to_excel(output_file, index=False)

    print("=" * 60)
    print(f"最终保留病例数: {len(df_merged)}")
    print(f"保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    merge_pre_post(
        pre_file="./data/pre_op_2.xlsx",
        post_file="./data/post_op_2.xlsx",
        output_file="data/total_2.xlsx"
    )

