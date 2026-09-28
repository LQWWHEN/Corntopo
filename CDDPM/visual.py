import os
import re
import cv2
import matplotlib.pyplot as plt


def visualize_results(result_dir, save_path="comparison.png"):
    """
    将 test_results 中的 pre / pred / gt / noise 可视化

    每一行：
    Pre | Pred | GT | Noise
    """

    # -------------------------
    # 获取所有样本编号
    # -------------------------
    ids = []

    for file in os.listdir(result_dir):
        m = re.match(r"(\d+)_pre\.png", file)
        if m:
            ids.append(int(m.group(1)))

    ids = sorted(ids)

    n = len(ids)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=4,
        figsize=(12, 3 * n)
    )

    if n == 1:
        axes = [axes]

    titles = ["Pre-OP", "Pred", "Post-OP", "Noise"]
    print(ids)
    for row, idx in enumerate(ids):

        images = [
            cv2.imread(
                os.path.join(result_dir, f"{idx}_pre.png")
            ),
            cv2.imread(
                os.path.join(result_dir, f"{idx}_pred.png")
            ),
            cv2.imread(
                os.path.join(result_dir, f"{idx}_gt.png")
            ),
            cv2.imread(
                os.path.join(result_dir, f"{idx}_error.png")
            ),
        ]

        for col in range(4):

            img = cv2.cvtColor(images[col], cv2.COLOR_BGR2RGB)

            axes[row][col].imshow(img)

            axes[row][col].axis("off")

            if row == 0:
                axes[row][col].set_title(
                    titles[col],
                    fontsize=14
                )

    plt.tight_layout()

    plt.savefig(
        save_path,
        figsize=(4,4),
        dpi=128,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved to {save_path}")


if __name__ == "__main__":

    visualize_results(
        "./test_results/final",
        "./comparison.png"
    )