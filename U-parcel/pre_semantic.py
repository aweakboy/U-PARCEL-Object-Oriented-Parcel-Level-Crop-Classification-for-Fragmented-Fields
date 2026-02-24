#!/usr/bin/env python3
"""
predict_from_tar.py

基于你的原始脚本，修改为从 .tar checkpoint 中加载最佳权重，
对 .npy 文件夹做推理并输出同名 .npy 文件。
"""
import os
import json
import numpy as np
import torch

# 确保你的项目路径中能够 import 到 model_utils.get_model
from src import model_utils

def load_config(conf_path: str):
    """从 conf.json 构建一个简单的 config 对象"""
    with open(conf_path, 'r') as f:
        cfg = json.load(f)
    class Config: pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    return config

def load_model_from_tar(model_dir: str, device: torch.device):
    """
    model_dir 下应包含：
      - conf.json
      - model.pth.tar  （最佳 checkpoint）
    """
    # 1. 读 conf.json 重建 config
    conf_path = os.path.join(model_dir, "conf.json")
    config = load_config(conf_path)

    # 2. 用同样的 config 构建模型结构
    model = model_utils.get_model(config, mode="semantic")
    model.to(device)

    # 3. 加载 checkpoint，并恢复 state_dict
    ckpt_path = os.path.join(model_dir, "model.pth.tar")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model

def predict_folder(model, input_dir: str, output_dir: str, device: torch.device):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.npy'):
            continue

        arr = np.load(os.path.join(input_dir, fname))
        # arr: e.g. (T, C, H, W)  or (T, features)  depending on your model
        tensor = torch.from_numpy(arr).unsqueeze(0).float().to(device)   # (1, T, ...)

        # ---- here’s the new bit ----
        T = tensor.shape[1]
        # create a [1×T] long‐tensor [0,1,2,…,T−1]
        batch_positions = torch.arange(T, device=device).unsqueeze(0)    # (1, T)
        # ----------------------------

        with torch.no_grad():
            # pass both inputs
            pred = model(tensor, batch_positions)

        pred_np = pred.squeeze(0).cpu().numpy()
        out_path = os.path.join(output_dir, fname)
        np.save(out_path, pred_np)
        print(f"Saved: {fname} → {out_path}")


def main():
    # —— 请根据实际情况修改下面三个路径 —— #
    model_dir  = r"C:\FYgao\bestpkl"   # 包含 conf.json & model.pth.tar
    input_dir  = r"C:\FYgao\test"          # 待预测 .npy 文件夹
    output_dir = r"C:\FYgao\pre"                       # 输出预测结果的文件夹
    # ---------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_tar(model_dir, device)
    predict_folder(model, input_dir, output_dir, device)

if __name__ == '__main__':
    main()
