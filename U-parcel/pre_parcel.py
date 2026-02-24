#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
#  predict_parcel.py
#
#  读取 conf.json 与模型权重（.pth.tar），对数据集根目录下 *所有* Patch
#  进行推理，并把结果存成：
#      patch_id , parcel_id , pred_class
#  的 CSV（若同一 parcel_id 出现在多个 patch，会重复记录）。
# ────────────────────────────────────────────────────────────────

import os, json, pickle as pkl
from types import SimpleNamespace

import torch
import torch.utils.data as data
import pandas as pd

from src import utils, model_utils                 # 训练阶段已有的工具
from src.parceldataset import ParcelPASTIS         # 数据集类（保持与训练一致）


# ──────────────────── ❶ 手动填写路径 ────────────────────
# ROOT_DATASET   = r"C:\Shandong\sddataset\dataset"         # 数据集根目录
# CHECKPOINT_TAR = r"C:\utae-paps-main\utae-paps-main\sd_results\Fold_5\model.pth.tar"   # 训练好的权重
# CONF_JSON      = r"C:\utae-paps-main\utae-paps-main\sd_results\conf.json"               # 训练时保存的配置
# OUTPUT_CSV     = r"C:\Shandong\sddataset\parcel1\parcel_preds_603.csv"
# DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────

ROOT_DATASET   = r"C:\Shandong\sddataset\dataset"         # 数据集根目录
CHECKPOINT_TAR = r"C:\utae-paps-main\utae-paps-main\sd_100\Fold_4\model.pth.tar"   # 训练好的权重
CONF_JSON      = r"C:\utae-paps-main\utae-paps-main\sd_100\conf.json"             # 训练时保存的配置
OUTPUT_CSV     = r"C:\FYgao\对比实验\parcel\parcel_preds.csv"
OUTPUT_JSON    = r"C:\FYgao\对比实验\parcel\metrics.json"  # 用于保存指标的文件
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ╭───────────────────── 工具函数 ─────────────────────╮
def recursive_todevice(x, device):
    """把任意嵌套结构递归搬到指定 device（None / 标量保持原状）"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [recursive_todevice(i, device) for i in x]
    return x


def inference_collate(batch):
    """
    ParcelPASTIS 在推理模式下返回 5 元组：
        imgs, dates, pid_raster, tgt_dict, patch_id
    把它们整理成批。
    """
    imgs, dates, pid, tgt, patch_ids = zip(*batch)
    return (
        torch.stack(imgs),          # [B,T,C,H,W]
        torch.stack(dates),         # [B,T]
        torch.stack(pid),           # [B,H,W]
        list(tgt),                  # List[dict]，此处用不上，可传 None 跳过
        list(patch_ids)             # List[int]
    )
# ╰───────────────────────────────────────────────────╯


def main():
    # ──────────────────── ❷ 读取训练阶段配置 ────────────────────
    with open(CONF_JSON, "r", encoding="utf-8") as f:
        conf_dict = json.load(f)

    # conf.json 里本来就已经把 list 字段解析好了，这里直接转成对象
    config = SimpleNamespace(**conf_dict)
    config.dataset_folder = ROOT_DATASET        # 覆盖成当前数据路径
    config.device         = DEVICE



    # ──────────────────── ❸ 数据集 & DataLoader ────────────────────
    # folds=None ⇒ 读取全部 patch；augment 需手动关掉
    dt_args = dict(
        root=ROOT_DATASET,
        pid_dir=r"C:\Shandong\sddataset\parcel",   # 与训练时保持相同
        norm=config.__dict__.get("norm", False),
        reference_date=config.ref_date,
        mono_date=config.mono_date,
        sats=["S1"],                    # 若训练用 S2/S1+S2，请自行调整
        use_pixel_label=False,          # 推理不需要 GT
        folds=None,
        cache=getattr(config, "cache", False)
    )
    dataset = ParcelPASTIS(**dt_args)
    dataset.enable_augment = False      # 关闭数据增强

    loader = data.DataLoader(
        dataset,
        batch_size=getattr(config, "batch_size", 4),
        shuffle=False,                  # 推理不打乱，方便对照
        num_workers=getattr(config, "num_workers", 4),
        collate_fn=inference_collate,
        pin_memory=True
    )



    # ──────────────────── ❹ 构建并加载模型 ────────────────────
    print("Building model . . .")
    model = model_utils.get_model(config, mode="semantic")
    ckpt  = torch.load(CHECKPOINT_TAR, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE).eval()
    print("Model ready.")

    # （如需 AMP，可在下方 with block 里加 autocast）



    # ──────────────────── ❺ 前向推理并收集结果 ────────────────────
    # ───────────── ❺ 推理 ─────────────
    rows = []

    with torch.no_grad():
        for imgs, dates, pid_raster, tgt, patch_ids in loader:
            # 前向推理
            imgs, dates, pid_raster, tgt = recursive_todevice(
                [imgs, dates, pid_raster, tgt], DEVICE
            )
            out = model(
                imgs,
                batch_positions=dates,
                pid_raster=pid_raster,
                targets=tgt  # 记得传 targets，让模型给所有 parcel 预测
            )

            # ① 先做一个  parcel_id → 预测类别  的字典
            pred_classes = out["parcel_logits"].argmax(dim=1).cpu()  # [P]
            parcel_ids = out["parcel_id"].cpu()  # [P]
            pred_dict = {int(pid): int(cls) for pid, cls in zip(parcel_ids,
                                                                pred_classes)}

            # ② 再逐个 patch 回到它自己的 pid_raster 写行
            for b, patch_id in enumerate(patch_ids):
                # patch 内实际出现的所有 parcel_id（>0）
                ids_in_patch = torch.unique(pid_raster[b]).cpu().tolist()
                ids_in_patch = [pid for pid in ids_in_patch if pid > 0]

                for pid in ids_in_patch:
                    cls = pred_dict.get(pid)+1
                    if cls is None:  # 理论上不会发生，保险检查
                        continue
                    rows.append({
                        "patch_id": int(patch_id),
                        "parcel_id": int(pid),
                        "pred_class": cls
                    })

    # ───────────── ❻ 保存 CSV ─────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(rows, columns=["patch_id", "parcel_id", "pred_class"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓  完成！共写入 {len(df):,} 行 → {OUTPUT_CSV}")

    # ──────────────────── ❻ 保存为 CSV ────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(rows, columns=["patch_id", "parcel_id", "pred_class"])
    df.sort_values(["patch_id", "parcel_id"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓  预测完成，已保存到:  {OUTPUT_CSV}")


# Windows 多进程入口保护
if __name__ == "__main__":
    main()