"""Dataset API compatible with train.py clinical_mean/clinical_std arguments."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transforms import DataTransform


DATASET_API_VERSION = "clinical_stats_v1"


def center_crop(arr, crop_size=128):

    h, w = arr.shape

    if h < crop_size or w < crop_size:
        raise ValueError(
            f"{arr.shape} 小于 crop_size={crop_size}"
        )

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    return arr[
        top:top+crop_size,
        left:left+crop_size
    ]
class CornealTopoDataset(Dataset):
    def __init__(self,opt, excel_path, data_root, input_maps=None, output_maps=None,
                 data_type="raw", image_size=None, normalize=True,transforms:DataTransform=None,
                 seed=42,indices=None,labels=None,dataframe=None,
                 clinical_mean=None,clinical_std=None):
        if indices is not None and labels is not None:
            raise ValueError("indices 和 labels 不能同时指定")
        self.opt = opt
        self.df = dataframe if dataframe is not None else pd.read_excel(excel_path)
        self.data_root = Path(data_root)
        self.data_type = data_type.lower()
        self.image_size = image_size
        self.normalize = normalize
        self.input_maps = input_maps or ["TOTAL CORNEAL REFRACTIVE POWER"]
        self.output_maps = output_maps or ["TOTAL CORNEAL REFRACTIVE POWER"]
        self.labels = sorted(self.df["label_index"].tolist())
        self.transforms = transforms
        self.seed = seed
        self.conditions = getattr(opt, "conditions", [])
        self.targets = getattr(opt, "targets", [])
        if (clinical_mean is None) != (clinical_std is None):
            raise ValueError("clinical_mean 和 clinical_std 必须同时提供")
        self.clinical_mean = None if clinical_mean is None else np.asarray(clinical_mean, dtype=np.float32)
        self.clinical_std = None if clinical_std is None else np.asarray(clinical_std, dtype=np.float32)
        if self.clinical_mean is not None:
            if (self.clinical_mean.shape != (len(self.conditions),)
                    or self.clinical_std.shape != (len(self.conditions),)
                    or not np.isfinite(self.clinical_mean).all()
                    or not np.isfinite(self.clinical_std).all()
                    or np.any(self.clinical_std <= 0)):
                raise ValueError("临床参数的均值、标准差必须与 conditions 对齐且为有效数值")
        all_labels = sorted(
            self.df["label_index"].tolist()
        )

        if labels is not None:
            if len(labels) != len(set(labels)) or not set(labels).issubset(set(all_labels)):
                raise ValueError("labels 中含重复或不属于当前 Excel 的 label_index")
            self.labels = list(labels)
        elif indices is not None:
            self.labels = [all_labels[i] for i in indices]
        else:
            self.labels = all_labels

    def get_type_folder(self):
        mapping = {"raw": self.data_root / "raw",
                   "grayscale": self.data_root / "grayscale",
                   "heatmaplocal": self.data_root / "heatmaps_local",
                   "heatmapglobal": self.data_root / "heatmaps_global"}
        if self.data_type not in mapping:
            raise ValueError(f"未知 data_type: {self.data_type}")
        return mapping[self.data_type]
    
    def load_excel_features(self, label):
        row = self.df[self.df["label_index"] == label]

        if len(row) == 0:
            raise ValueError(f"label {label} not found in excel")

        row = row.iloc[0]

        feats = []
        for f in self.conditions:
            feats.append(float(row[f]))

        features = np.array(feats, dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"label_index={label} 的临床参数包含缺失或无穷值")
        if self.clinical_mean is not None:
            features = (features - self.clinical_mean) / self.clinical_std
        return features
    
    def load_excel_targets(self, label):
        row = self.df[self.df["label_index"] == label]

        if len(row) == 0:
            raise ValueError(f"label {label} not found in excel")

        row = row.iloc[0]

        targets = []
        for t in self.targets:
            targets.append(float(row[t]))

        return np.array(targets, dtype=np.float32)

    def load_map(self, map_name, label, phase="pre"):
        root = self.get_type_folder()
        if self.data_type == "raw":
            path = root / map_name / f"{label}_{phase}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            img = np.load(path).astype(np.float32)
    
    
            img = center_crop(img,self.image_size)
  

            if self.normalize:
                valid = img[~np.isnan(img)]
                if len(valid) > 0:
                    vmin, vmax = valid.min(), valid.max()
                    img = (img - vmin) / (vmax - vmin + 1e-8)
                    img[np.isnan(img)] = 0
            return img
        else:
            path = root / map_name / f"{label}_{phase}.png"
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            img = img.astype(np.float32)
            if self.normalize:
                img /= 255.0
            if self.image_size is not None and img.shape[0] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            return img

    def load_input(self, label):
        
        channels = [self.load_map(map_name, label, phase="pre") for map_name in self.input_maps]
        return np.stack(channels, axis=0)

    def load_target(self, label):
        channels = [self.load_map(map_name, label, phase="post") for map_name in self.output_maps]
        return np.stack(channels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]


            
        x = self.load_input(label)
        y = self.load_target(label)
        mask_post = (y > 0).astype(np.float32)
        mask_pre = (x > 0).astype(np.float32)
        cond = self.load_excel_features(label)
        targets = self.load_excel_targets(label)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        mask_post = torch.from_numpy(mask_post).float()
        mask_pre = torch.from_numpy(mask_pre).float()
        cond = torch.from_numpy(cond).float()
        targets = torch.from_numpy(targets).float()

        if self.transforms is not None:
            params =  self.transforms.get_params(seed=self.seed)
            x = self.transforms(x, params)
            y = self.transforms(y, params)
            mask_pre= self.transforms(mask_pre, params)
            mask_post= self.transforms(mask_post, params)
            
  
        return {"input": x, 
                "target": y,
                "mask_pre": mask_pre,
                "mask_post":mask_post,         
                "condition": cond,
                "targets":targets,
                "label": label}




def build_dataloader(excel_path, data_root, input_maps, output_maps, data_type="raw",
                     batch_size=4, shuffle=True, num_workers=4, image_size=None, opt=None):
    dataset = CornealTopoDataset(opt=opt, excel_path=excel_path, data_root=data_root,
                                 input_maps=input_maps, output_maps=output_maps,
                                 data_type=data_type, image_size=image_size,normalize=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    loader = build_dataloader(
        excel_path="../../data/total.xlsx",
        data_root="../../data/img512",
        input_maps=["TOTAL CORNEAL REFRACTIVE POWER"],
        output_maps=["TOTAL CORNEAL REFRACTIVE POWER"],
        data_type="raw",
        image_size=128,
        batch_size=4,
    
    )
    batch = next(iter(loader))
    print(batch["input"].shape, batch["target"].shape, batch["mask_pre"].shape, batch["label"])
