import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.dataset import PASTIS_Dataset
from src.model_utils import get_model
from src.metrics import compute_metrics_detailed

# ----------------- 配置写死部分 -----------------


# ----------------- 加载 JSON 配置 -----------------
with open(r"C:\FYgao\对比实验\FY\utae\conf.json", "r") as f:
    cfg = json.load(f)

# 设定路径
MODEL_NAME = "utae"
MODEL_PATH = r"C:\FYgao\对比实验\FY\utae\Fold_1\model.pth.tar"
DATA_ROOT = r"C:\FYgao\fydataset\dataset"
FOLD = 5
OUTPUT_DIR = r"C:\FYgao\对比实验\FY\utae\utae_fold1"

# ----------------- 定义 Config 类 -----------------
class Config:
    def __init__(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

config = Config(cfg)

# ----------------- 主推理逻辑 -----------------
def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred_npy_dir = os.path.join(OUTPUT_DIR, "preds_raw")
    os.makedirs(pred_npy_dir, exist_ok=True)

    # 加载模型
    model = get_model(config, mode="semantic")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.eval().cuda()

    # 加载测试集
    test_set = PASTIS_Dataset(
        folder=DATA_ROOT,
        norm=False,
        target="semantic",
        cache=config.cache,
        folds=[FOLD],
        sats=["S1"]
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    all_preds = []
    all_gts = []

    # 推理
    with torch.no_grad():
        for i, ((x, dates), y) in enumerate(tqdm(test_loader)):
            x, y, dates = x.cuda(), y.cuda(), dates.cuda()
            x = x.float()
            dates = dates.float()
            logits = model(x, dates)

            # logits = model(x.unsqueeze(0), dates)  # 传入位置编码
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            target = y.squeeze(0).cpu().numpy()

            all_preds.append(preds)
            all_gts.append(target)

            np.save(os.path.join(pred_npy_dir, f"{i:04}_pred.npy"), preds)
            np.save(os.path.join(pred_npy_dir, f"{i:04}_gt.npy"), target)

    # 计算并保存指标（含每类 F1）
    metrics = compute_metrics_detailed(all_preds, all_gts, num_classes=config.num_classes)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n===== Overall Results =====")
    print("Overall Accuracy:", metrics["overall_accuracy"])
    print("Mean IoU:", metrics["mean_iou"])
    print("Mean F1:", metrics["mean_f1"])
    print("Per-class F1:", metrics["f1_per_class"])


if __name__ == "__main__":
    evaluate()
