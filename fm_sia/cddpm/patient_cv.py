"""Reproducible patient-level cross-validation for paired corneal maps."""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


SPLIT_VERSION = 1


def _value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = " ".join(str(value).split()).casefold()
    return text if text and text not in {"nan", "none", "null"} else None


def _labels_and_groups(df):
    if "label_index" not in df:
        raise ValueError("Excel 中缺少 label_index 列")
    columns = [
        f"{side}_PATIENT_{field}"
        for side in ("pre", "post")
        for field in ("PatientID", "LastName", "FirstName", "DateOfBirth")
    ]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Excel 中缺少患者身份列: {sorted(missing)}")

    labels = df["label_index"].tolist()
    if any(pd.isna(label) for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("label_index 必须非空且唯一")
    labels = [label.item() if isinstance(label, np.generic) else label for label in labels]

    # A row can link pre/post IDs and demographic identities. This also keeps
    # both eyes together when one examination is missing a patient ID.
    parent = list(range(len(labels)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    seen = {}
    identity_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        identities = set()
        for side in ("pre", "post"):
            prefix = f"{side}_PATIENT_"
            patient_id = _value(row[prefix + "PatientID"])
            if patient_id is not None:
                identities.add(("id", patient_id))
            demographic = tuple(
                _value(row[prefix + field])
                for field in ("LastName", "FirstName", "DateOfBirth")
            )
            if all(part is not None for part in demographic):
                identities.add(("name_dob", *demographic))
        if not identities:
            raise ValueError(f"label_index={labels[i]} 缺少可用于患者分组的身份信息")
        identity_rows.append((labels[i], sorted(identities)))
        for identity in identities:
            if identity in seen:
                union(i, seen[identity])
            else:
                seen[identity] = i

    roots = [find(i) for i in range(len(labels))]
    group_by_label = dict(zip(labels, roots))
    payload = json.dumps(
        sorted(identity_rows, key=lambda item: (str(type(item[0])), str(item[0]))),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return labels, group_by_label, fingerprint


def _validate_splits(splits, labels, group_by_label, fingerprint, seed, n_splits):
    if (splits.get("version"), splits.get("seed"), splits.get("n_splits")) != (
        SPLIT_VERSION, seed, n_splits
    ):
        raise ValueError("划分文件版本、随机种子或折数与当前参数不一致")
    if splits.get("dataset_fingerprint") != fingerprint:
        raise ValueError("划分文件对应的 Excel 患者或样本已变化，请使用新的 --split_file")
    folds = splits.get("folds", [])
    if len(folds) != n_splits:
        raise ValueError("划分文件的折数不正确")
    expected = set(labels)
    all_tests = []
    for i, fold in enumerate(folds, start=1):
        if fold.get("fold") != i:
            raise ValueError(f"第 {i} 折编号不正确")
        sets = []
        for partition in ("train", "val", "test"):
            values = fold.get(partition, [])
            if not values or len(values) != len(set(values)):
                raise ValueError(f"第 {i} 折 {partition} 为空或含重复样本")
            sets.append(set(values))
        train, val, test = sets
        if (train | val | test) != expected or train & val or train & test or val & test:
            raise ValueError(f"第 {i} 折样本未完整覆盖或跨子集重复")
        patient_sets = [{group_by_label[label] for label in part} for part in sets]
        if any(patient_sets[a] & patient_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"第 {i} 折出现患者跨子集泄漏")
        all_tests.extend(test)
    if len(all_tests) != len(expected) or set(all_tests) != expected:
        raise ValueError("10 折测试集未确保每个样本恰好出现一次")


def load_or_create_cv_splits(df, split_file, *, seed=42, n_splits=10):
    """Return folds of label_index; reuse JSON only when the dataset matches."""
    if n_splits != 10:
        raise ValueError("当前实验要求固定 10 折")
    labels, group_by_label, fingerprint = _labels_and_groups(df)
    # Sorting by member labels keeps fresh generation stable if Excel row order changes.
    members = {}
    for label, group in group_by_label.items():
        members.setdefault(group, []).append(label)
    unique_groups = sorted(
        members, key=lambda group: min(str(label) for label in members[group])
    )
    if len(unique_groups) < n_splits:
        raise ValueError(f"只有 {len(unique_groups)} 个患者，无法进行 {n_splits} 折划分")

    split_file = Path(split_file)
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            splits = json.load(f)
    else:
        # Randomize group IDs before GroupKFold to make tie breaking seeded.
        shuffled = np.random.default_rng(seed).permutation(len(unique_groups))
        group_codes = dict(zip(unique_groups, shuffled.tolist()))
        groups = np.array([group_codes[group_by_label[label]] for label in labels])
        labels_arr = np.array(labels)
        folds = []
        outer = GroupKFold(n_splits=n_splits)
        for fold_number, (train_val_idx, test_idx) in enumerate(
            outer.split(labels_arr, groups=groups), start=1
        ):
            # GroupShuffleSplit selects patients; pick the closest sample count
            # among deterministic candidates because some patients have two eyes.
            inner = GroupShuffleSplit(n_splits=128, test_size=1 / 9, random_state=seed + fold_number)
            train_idx, val_idx = min(
                inner.split(labels_arr[train_val_idx], groups=groups[train_val_idx]),
                key=lambda pair: abs(len(pair[1]) - len(labels) / n_splits),
            )
            folds.append({
                "fold": fold_number,
                "train": sorted(labels_arr[train_val_idx[train_idx]].tolist()),
                "val": sorted(labels_arr[train_val_idx[val_idx]].tolist()),
                "test": sorted(labels_arr[test_idx].tolist()),
            })
        splits = {
            "version": SPLIT_VERSION,
            "seed": seed,
            "n_splits": n_splits,
            "dataset_fingerprint": fingerprint,
            "folds": folds,
        }
        _validate_splits(splits, labels, group_by_label, fingerprint, seed, n_splits)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = split_file.with_name(split_file.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)
        temporary.replace(split_file)

    _validate_splits(splits, labels, group_by_label, fingerprint, seed, n_splits)
    return splits["folds"]
