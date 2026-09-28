import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# raw数据目录
raw_root = Path("raw")

# 输出256x256目录（可选保存）
out_root = Path("raw_256")
out_root.mkdir(exist_ok=True)
from scipy.interpolate import griddata

def upscale_griddata(arr, target_size=256):
    h, w = arr.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    mask = ~np.isnan(arr)
    points = np.stack([x[mask], y[mask]], axis=-1)
    values = arr[mask]

    # 新网格
    xi = np.linspace(0, w-1, target_size)
    yi = np.linspace(0, h-1, target_size)
    xi, yi = np.meshgrid(xi, yi)

    arr_up = griddata(points, values, (xi, yi), method='linear')

    return arr_up
# 遍历矩阵文件
for map_dir in raw_root.iterdir():
    if not map_dir.is_dir():
        continue

    out_map_dir = out_root / map_dir.name
    out_map_dir.mkdir(exist_ok=True)

    for npy_file in map_dir.glob("*.npy"):
        # 加载矩阵
        arr = np.load(npy_file)

        arr_up = upscale_griddata(arr,512)

   

        # 显示效果
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(arr, cmap="jet")
        plt.title("Original 141x141")
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(arr_up, cmap="jet")
        plt.title("Upscaled 256x256")
        plt.colorbar()
        plt.show()

        # 可选保存
        out_path = out_map_dir / npy_file.name
        np.save(out_path, arr_up)

