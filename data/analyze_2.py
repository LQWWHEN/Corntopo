import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_ROOT = Path("./img512/raw")
MAP_NAME = "TOTAL CORNEAL REFRACTIVE POWER"

df = pd.read_excel("total.xlsx")

added_sum = None
removed_sum = None

N = 0

for label in df["label_index"]:

    pre = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_pre.npy"
    )

    post = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_post.npy"
    )

    pre_mask = ~np.isnan(pre)
    post_mask = ~np.isnan(post)

    added = post_mask & (~pre_mask)
    removed = pre_mask & (~post_mask)

    if added_sum is None:

        added_sum = added.astype(np.float32)
        removed_sum = removed.astype(np.float32)

    else:

        added_sum += added
        removed_sum += removed

    N += 1

added_prob = added_sum / N
removed_prob = removed_sum / N

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(added_prob, cmap="hot")
plt.colorbar()
plt.title("Added Region Probability")

plt.subplot(1,2,2)
plt.imshow(removed_prob, cmap="hot")
plt.colorbar()
plt.title("Removed Region Probability")

plt.tight_layout()
plt.show()

mask_pre_sum = None
mask_post_sum = None

N = 0

for label in df["label_index"]:

    pre = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_pre.npy"
    )

    post = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_post.npy"
    )

    pre_mask = (~np.isnan(pre)).astype(np.float32)
    post_mask = (~np.isnan(post)).astype(np.float32)

    if mask_pre_sum is None:

        mask_pre_sum = pre_mask
        mask_post_sum = post_mask

    else:

        mask_pre_sum += pre_mask
        mask_post_sum += post_mask

    N += 1

mean_pre = mask_pre_sum / N
mean_post = mask_post_sum / N

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(mean_pre,cmap="jet")
plt.colorbar()
plt.title("Mean Pre Mask")

plt.subplot(1,2,2)
plt.imshow(mean_post,cmap="jet")
plt.colorbar()
plt.title("Mean Post Mask")

plt.tight_layout()
plt.show()

diffs = []

for label in df["label_index"]:

    pre = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_pre.npy"
    )

    post = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_post.npy"
    )

    pre_area = (~np.isnan(pre)).sum()
    post_area = (~np.isnan(post)).sum()

    diffs.append(
        post_area - pre_area
    )

diffs = np.array(diffs)

print("Mean =", diffs.mean())
print("Std =", diffs.std())

plt.hist(diffs,bins=40)
plt.xlabel("Post Area - Pre Area")
plt.ylabel("Count")
plt.show()

from scipy.stats import pearsonr

dice_list = []
ratio_list = []

for label in df["label_index"]:

    pre = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_pre.npy"
    )

    post = np.load(
        DATA_ROOT /
        MAP_NAME /
        f"{label}_post.npy"
    )

    pre_mask = ~np.isnan(pre)
    post_mask = ~np.isnan(post)

    intersection = (
        pre_mask &
        post_mask
    ).sum()

    dice = (
        2 * intersection
        /
        (
            pre_mask.sum()
            +
            post_mask.sum()
        )
    )

    ratio = (
        post_mask.sum()
        /
        pre_mask.sum()
    )

    dice_list.append(dice)
    ratio_list.append(ratio)

corr,p = pearsonr(
    ratio_list,
    dice_list
)

print("Dice-Ratio Corr =",corr)

plt.scatter(
    ratio_list,
    dice_list,
    s=15
)

plt.xlabel("Area Ratio")
plt.ylabel("Dice")
plt.show()