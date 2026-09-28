import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_ROOT = Path("./img512/raw")
MAP_NAME = "TOTAL CORNEAL REFRACTIVE POWER"

excel = pd.read_excel("total.xlsx")

ratios = []
pre_areas = []
post_areas = []

for label in excel["label_index"]:

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

    pre_mask = (~np.isnan(pre))
    post_mask = (~np.isnan(post))

    pre_area = pre_mask.sum()
    post_area = post_mask.sum()

    ratio = post_area / (pre_area + 1e-8)

    pre_areas.append(pre_area)
    post_areas.append(post_area)
    ratios.append(ratio)

ratios = np.array(ratios)

print("Samples:", len(ratios))
print("Mean :", ratios.mean())
print("Std  :", ratios.std())
print("Min  :", ratios.min())
print("Max  :", ratios.max())

plt.figure(figsize=(6,4))
plt.hist(ratios, bins=30)
plt.xlabel("Post Area / Pre Area")
plt.ylabel("Count")
plt.title("Area Ratio Distribution")
plt.show()
from scipy.stats import pearsonr

corr, p = pearsonr(
    pre_areas,
    post_areas
)

print("Pearson r =", corr)
print("p =", p)

dice_list = []
iou_list = []

for label in excel["label_index"]:

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

    pre_mask = (~np.isnan(pre))
    post_mask = (~np.isnan(post))

    intersection = np.logical_and(
        pre_mask,
        post_mask
    ).sum()

    union = np.logical_or(
        pre_mask,
        post_mask
    ).sum()

    dice = (
        2 * intersection
        /
        (
            pre_mask.sum()
            +
            post_mask.sum()
            + 1e-8
        )
    )

    iou = intersection / (union + 1e-8)

    dice_list.append(dice)
    iou_list.append(iou)

print()
print("Dice")
print(np.mean(dice_list))
print(np.std(dice_list))

print()
print("IoU")
print(np.mean(iou_list))
print(np.std(iou_list))