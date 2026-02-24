# file: datasets/parcel_pastis.py
import os, json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from skimage.measure import block_reduce        # 用来把 128×128 → 16×16 取众数
import torch
from torch.utils.data import Dataset
from scipy.stats import mode


import torch.nn.functional as F
import random

# ---------- 数据增强工具 ---------- #
def _rand_flip_rot(img, k):
    # 0:原图 1:水平翻 2:垂直翻 3:水平+垂直 4~7:再各自旋转 90°
    if k & 1:            # 水平翻
        img = img.flip(-1)
    if k & 2:            # 垂直翻
        img = img.flip(-2)
    rot_times = (k >> 2) & 3
    if rot_times:
        img = torch.rot90(img, k=rot_times, dims=(-2, -1))
    return img

def augment_sample(data_dict, pid_raster, pixel_label=None, p_flip=0.5, p_noise=0.3):
    """
    data_dict : {'S1': [T,C,H,W], ...}  (torch.Tensor)
                — 或者单独的 torch.Tensor，此时会被自动包装成 {'data': Tensor}
    pid_raster: [H,W] (torch.LongTensor)
    pixel_label: [H,W] (torch.LongTensor or None)

    随机水平/垂直翻转 + 90 度旋转；保持所有张量一致变换
    随机高斯噪声 (同一幅图所有帧相同 σ)
    """

    # —— 兼容单个 Tensor 输入 ——
    single_tensor = False
    if not isinstance(data_dict, dict):
        data_dict = {'data': data_dict}
        single_tensor = True

    # —— 随机几何变换 ——
    if random.random() < p_flip:
        k = random.randint(1, 7)  # 0 表示什么都不做
        # 对字典里每个 Tensor 一致变换
        data_dict = {s: _rand_flip_rot(v, k) for s, v in data_dict.items()}
        pid_raster = _rand_flip_rot(pid_raster, k)
        if pixel_label is not None:
            pixel_label = _rand_flip_rot(pixel_label, k)

    # —— 随机加噪声 ——
    if random.random() < p_noise:
        sigma = random.uniform(0.01, 0.03)  # 归一化后通道尺度
        for s, tensor in data_dict.items():
            noise = torch.randn_like(tensor) * sigma
            data_dict[s] = tensor + noise

    # —— 返回结果 ——
    if single_tensor:
        # 如果最初传入的是单个 Tensor，就拆回来
        return data_dict['data'], pid_raster, pixel_label
    else:
        return data_dict, pid_raster, pixel_label


class ParcelPASTIS(Dataset):
    """
    返回
        imgs          : [T,C,H,W] (torch.float32)
        date_positions: [T]       (torch.int16)   – 以 ref_date 为零点的天数
        pid_raster    : [H',W']   (torch.long)    – 下采样后 parcel_id
        target        : dict{
                             'parcel_label': [P] (torch.long)
                             'pixel_label' : [H',W'] (optional, 与 pid_raster 同分辨率)
                           }
    """
    # -----------------------  ↓↓↓ 根据你的 U-TAE 下采样倍数来设 ↓↓↓ --------------------
    # 把原来写死的倍数去掉
    DOWNSAMPLE = 1  # ← 先设成 1

    def _load_raster_id(self, patch_name):
        with rasterio.open(os.path.join(self.pid_dir, f"{patch_name}_parcel_id.tif")) as src:
            pid = src.read(1)  # (H,W) = 256,256
        # 只在 DOWNSAMPLE>1 时才做 block_reduce
        if self.DOWNSAMPLE >= 1:
            pid = block_reduce(
                pid,
                (self.DOWNSAMPLE, self.DOWNSAMPLE),
                func=lambda x: np.bincount(x.astype(int)).argmax())
        return torch.from_numpy(pid.astype(np.int64))

    def __init__(
        self,
        root: str,
        sats      = ("S1",),               # DATA_S1/TARGET 等子目录
        norm      = False,
        folds     = None,
        pid_dir   = r"C:\Shandong\sddataset\parcel",      # 保存 <tile.npy>_parcel_id.tif 的目录
        meta_csv  = "parcel_meta.csv",     # 内含 parcel_id,type,area_m2
        class_map = None,                  # dict{type(str)->class_idx(int)}
        use_pixel_label = False,           # 要不要同时返回像元语义 (做多任务)
        reference_date  = "2024-09-01",
        mono_date       = None,            # 与老版保持一致
        cache     = False,
        mem16     = False,
    ):
        super().__init__()
        self.root  = root
        self.sats  = sats
        self.norm_ = norm
        self.ref_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = (
            datetime(*map(int, mono_date.split("-"))) if mono_date and "-" in mono_date else mono_date
        )
        self.memory       = {}
        self.memory_dates = {}

        self.enable_augment = True  # 训练集会打开；验证 / 测试自动关闭

        # ---------- 1) patch 元数据 ----------
        self.meta_patch = gpd.read_file(os.path.join(root, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )
        self.id_patches = self.meta_patch.index.tolist()
        self.len        = len(self.id_patches)

        # ---------- 2) 预加载日期掩码 ----------
        self.date_range  = np.arange(-300, 300)
        self.date_tables = {s: None for s in self.sats}
        for s in self.sats:
            tables = {}
            for pid, date_seq in self.meta_patch[f"dates-{s}"].items():
                if isinstance(date_seq, str):
                    date_seq = json.loads(date_seq)
                d = prepare_dates(date_seq, self.ref_date)      # → np.array([..])
                mask       = np.zeros_like(self.date_range, dtype=np.int8)
                mask[np.isin(self.date_range, d)] = 1
                tables[pid] = mask
            self.date_tables[s] = tables

        # ---------- 3) 归一化统计 ----------
        if self.norm_:
            self.norm = {}
            for s in self.sats:
                fn = os.path.join(root, f"NORM_{s}_patch.json")
                with open(fn) as f:
                    allfold = json.load(f)
                keep = folds if folds else range(1,6)
                means = [allfold[f"Fold_{k}"]["mean"] for k in keep]
                stds  = [allfold[f"Fold_{k}"]["std"]  for k in keep]
                self.norm[s] = (
                    torch.tensor(np.mean(means,0), dtype=torch.float32),
                    torch.tensor(np.mean(stds ,0), dtype=torch.float32)
                )
        else:
            self.norm = None

        # ---------- 4) parcel_id → class_idx ----------
        meta_df = pd.read_csv(os.path.join(pid_dir, meta_csv))
        if class_map is None:                 # 自动映射成 0..K-1
            cats = sorted(meta_df["type"].unique())
            class_map = {t:i for i,t in enumerate(cats)}
        self.pid2cls = dict(zip(meta_df["parcel_id"].values,
                                meta_df["type"].map(class_map).values))

        self.use_pixel_label = use_pixel_label
        self.pid_dir = pid_dir
        print("Dataset ready.")

    # --------------------- utils ---------------------
    def _load_raster_id(self, patch_name: str):
        tif_path = os.path.join(self.pid_dir, f"S1_{patch_name}.npy_parcel_id.tif")
        with rasterio.open(tif_path) as src:
            pid = src.read(1)                # (H,W) numpy.int32
        # 下采样到 H//D, W//D 取众数
        D = self.DOWNSAMPLE
        if D > 1:
            # pid_ds = block_reduce(pid, (D,D), func=lambda x: np.bincount(x.astype(int)).argmax())
            # parceldataset.py  →  _load_raster_id
            # pid_ds = block_reduce(
            #     pid,
            #     (D, D),
            #     func=lambda x, axis=None: np.bincount(x.astype(int)).argmax()
            #
            pid_ds = block_reduce(
                pid, (D, D),
                func=lambda x, axis=None: mode(
                    x.astype(int), axis=axis, keepdims=False
                )[0]
            )


        else:
            pid_ds = pid
        return torch.from_numpy(pid_ds.astype(np.int64))   # [H',W']

    def _parcel_labels(self, pid_tensor: torch.Tensor):
        pids = torch.unique(pid_tensor)
        pids = pids[pids>0]
        labels = torch.tensor([self.pid2cls[int(p)] for p in pids], dtype=torch.long)
        return pids, labels
    # --------------------------------------------------

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        patch_id = self.id_patches[idx]

        # ===== 1. 图像 & dates =====
        if not self.cache or idx not in self.memory:
            data = {s: np.load(os.path.join(
                        self.root, f"DATA_{s}", f"{s}_{patch_id}.npy"
                   )).astype(np.float32)
                    for s in self.sats}                           # T,C,H,W
            data = {s: torch.from_numpy(arr) for s,arr in data.items()}

            if self.norm:
                for s in self.sats:
                    mu, sig = self.norm[s]
                    data[s] = (data[s] - mu[None,:,None,None]) / sig[None,:,None,None]

            if self.cache:
                saved = {k:v.half() if self.mem16 else v.clone()
                         for k,v in data.items()}
                self.memory[idx] = saved
        else:
            data = {k:(v.float() if self.mem16 else v.clone())
                    for k,v in self.memory[idx].items()}

        # ---- 日期 (与原版相同) ----
        dates = {s: torch.from_numpy(
                    self.date_range[np.where(self.date_tables[s][patch_id]==1)[0]]
                 ) for s in self.sats}

        # 单时相裁剪
        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: d[self.mono_date].unsqueeze(0) for s,d in data.items()}
                dates= {s: dt[self.mono_date]             for s,dt in dates.items()}
            else:                                          # 指定日期
                mono_delta = (self.mono_date - self.ref_date).days
                chosen = {s: int((dt-mono_delta).abs().argmin()) for s,dt in dates.items()}
                data  = {s: d[chosen[s]].unsqueeze(0) for s,d  in data.items()}
                dates = {s: dates[s][chosen[s]]       for s in self.sats}

        # 若只用一个卫星 → 去掉 dict
        if len(self.sats)==1:
            data  = data[self.sats[0]]
            dates = dates[self.sats[0]]

        # ===== 2. parcel_id 栅格 =====
        pid_raster = self._load_raster_id(patch_id)        # [H',W']

        # ===== 3. targets =====
        pids, plabels = self._parcel_labels(pid_raster)
        tgt = {"parcel_label": plabels,
               "parcel_id_vec": pids}

        if self.use_pixel_label:
            # 把原 128×128 像元标签读进来并同步下采样
            # pix = np.load(os.path.join(self.root,"ANNOTATIONS",f"TARGET_{patch_id}.npy"))
            # pix_ds = block_reduce(pix, (self.DOWNSAMPLE,self.DOWNSAMPLE), func=lambda x: np.bincount(x).argmax())
            # tgt["pixel_label"] = torch.from_numpy(pix_ds.astype(np.int64))
            # 下采样、众数池化这两步都不要了，直接用原尺寸
            pix = np.load(os.path.join(self.root, "ANNOTATIONS", f"TARGET_{patch_id}.npy"))
            pix = pix.astype(np.int64)

            # 把背景 0 设成 ignore_index = -1
            pix[pix == 0] = -1

            tgt["pixel_label"] = torch.from_numpy(pix)

        # ---------------- 数据增强 ----------------
        # 仅在训练模式、且打开开关时起作用
        if self.enable_augment:
            data, pid_raster, pix_lbl = augment_sample(
                data, pid_raster,
                tgt.get("pixel_label") if "pixel_label" in tgt else None
            )
            tgt["pixel_label"] = pix_lbl if pix_lbl is not None else tgt.get("pixel_label")
        # -----------------------------------------

        # ===== 4. 返回 =====__getitem__
        return (data, dates, pid_raster,tgt,patch_id)
        # return (data, dates, pid_raster,tgt)  #训练时取消注释

# ------------------------- util --------------------------
def prepare_dates(date_dict, ref_date):
    """date_dict: metadata 里某 patch 的日期字典 (key: str idx, val: 'YYYYMMDD' int)"""
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    return ((pd.to_datetime(d[0].astype(str), format='%Y%m%d') - ref_date).dt.days).values

