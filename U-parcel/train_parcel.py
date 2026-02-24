"""
Main script for semantic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt

from src import utils, model_utils
from src.dataset import PASTIS_Dataset
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.learning.weight_init import weight_init


# --- NEW ---
from src.parceldataset import ParcelPASTIS        # ← 路径按你项目实际放置

# train_parcel.py 顶部或 utils 里加
def parcel_collate(batch):
    # imgs, dates, pid, tgt,patch_ids = zip(*batch)   # 每项都是长度 B 的 tuple
    imgs, dates, pid, tgt = zip(*batch)  # 训练时取消注释
    return (
        torch.stack(imgs),                # [B,T,C,H,W]
        torch.stack(dates),               # [B,T]
        torch.stack(pid),                 # [B,H,W]
        list(tgt),                  # 保持 list，里面每个都是 dict
        # list(patch_id)
    )

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 5]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default=r"C:\Shandong\sddataset\dataset",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./sd_20",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=10,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)

# Training parameters
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2024-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=5, type=int)
parser.add_argument("--ignore_index", default=0, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)

parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--patience",     default=30,  type=int,   help="early-stop patience (epochs)")


# def iterate(model, data_loader, config, optimizer=None, mode="train", device=None):
#     loss_meter = tnt.meter.AverageValueMeter()
#     acc_meter  = tnt.meter.AverageValueMeter()
#
#     for i, batch in enumerate(data_loader):
#         if device is not None:
#             batch = recursive_todevice(batch, device)
#
#         imgs, dates, pid, tgt = batch
#         # --- 前向 ---
#         if mode != "train":
#             with torch.no_grad():
#                 out = model(imgs, batch_positions=dates, pid_raster=pid, targets=tgt)
#
#         else:
#             optimizer.zero_grad()
#             out = model(imgs, batch_positions=dates, pid_raster=pid, targets=tgt)
#             out["loss"].backward()
#             optimizer.step()
#
#         loss_meter.add(out["loss"].item())
#
#         # 简单 parcel-level accuracy
#         with torch.no_grad():
#             pred = out["parcel_logits"].argmax(dim=1)
#             valid_pred, valid_lbl = [], []
#
#             for b, t in enumerate(tgt):  # tgt 即 targets
#                 mask = (out["parcel_batch"] == b)
#                 pids_b = out["parcel_id"][mask]
#                 pred_b = pred[mask]
#
#                 id2lab = dict(zip(t["parcel_id_vec"].tolist(),
#                                   t["parcel_label"].tolist()))
#                 for i, pid in enumerate(pids_b):
#                     lab = id2lab.get(int(pid))
#                     if lab is not None:
#                         valid_pred.append(pred_b[i])
#                         valid_lbl.append(lab)
#
#             if valid_lbl:  # 避免空列表
#                 valid_pred = torch.stack(valid_pred)
#                 valid_lbl = torch.tensor(valid_lbl, device=valid_pred.device)
#                 acc_meter.add((valid_pred == valid_lbl).float().mean().item())
#
#         # with torch.no_grad():
#         #     lbl_list = []
#         #     for b, t in enumerate(tgt):
#         #         mask = (out["parcel_batch"] == b)
#         #         pids = out["parcel_id"][mask]
#         #         id2lab = dict(zip(t["parcel_id_vec"].tolist(),
#         #                           t["parcel_label"].tolist()))
#         #         labels = torch.tensor(
#         #             [id2lab[int(pid)] for pid in pids],
#         #             device=pred.device)
#         #         lbl_list.append(labels)
#         #     labels_all = torch.cat(lbl_list, dim=0)
#         #     acc = (pred == labels_all).float().mean()
#         #     acc_meter.add(acc.item())
#
#             # correct = (pred == lbl).float().mean()
#             # acc_meter.add(correct.item())
#
#         if (i + 1) % config.display_step == 0:
#             print(f"Step {i+1}/{len(data_loader)}  "
#                   f"Loss {loss_meter.value()[0]:.4f}  "
#                   f"Acc {acc_meter.value()[0]*100:.2f}%")
#
#     metrics = {f"{mode}_loss": loss_meter.value()[0],
#                f"{mode}_accuracy": acc_meter.value()[0]}
#     return metrics
def iterate(model, data_loader, config,
            optimizer=None, mode="train", device=None):
    """
    计算 Loss / Accuracy / mIoU（parcel-level）
    返回 metrics 字典，会带上 `${mode}_IoU`
    """
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter  = tnt.meter.AverageValueMeter()

    n_cls = config.num_classes
    ignore_idx = config.ignore_index

    # 按类别累计交并集
    inter = torch.zeros(n_cls, device=device)
    union = torch.zeros(n_cls, device=device)
    conf_mat = torch.zeros(n_cls, n_cls, device=device)  # <— 新增
    for i, batch in enumerate(data_loader):
        # ---------- 数据搬到设备 ----------
        if device is not None:
            batch = recursive_todevice(batch, device)
        #
        imgs, dates, pid, tgt= batch     # tgt 是 list[dict]

        # imgs, dates, pid, tgt,patch_ids = batch     #预测取消注释

        # ---------- 前向 ----------
        if mode == "train":
            optimizer.zero_grad()
            out = model(imgs, batch_positions=dates,
                        pid_raster=pid, targets=tgt)
            out["total"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(imgs, batch_positions=dates,
                            pid_raster=pid, targets=tgt)

        loss_meter.add(out["total"].item())

        # ---------- 取预测 & 真值（parcel 对齐版） ----------
        pred  = out["parcel_logits"].argmax(dim=1)      # [P]
        batch_idx = out["parcel_batch"]                 # [P]
        pid_vec   = out["parcel_id"]                    # [P]

        lbl_list, pred_list = [], []
        for b, t in enumerate(tgt):                     # 遍历 batch
            mask   = (batch_idx == b)
            pids_b = pid_vec[mask]
            pred_b = pred[mask]

            id2lab = dict(zip(t["parcel_id_vec"].tolist(),
                              t["parcel_label"].tolist()))
            for p, lab in zip(pred_b, pids_b):
                gt = id2lab.get(int(lab))
                if gt is not None and gt != ignore_idx:
                    lbl_list.append(gt)
                    pred_list.append(int(p))

        if lbl_list:                                    # 该 batch 至少有一个 parcel
            lbl_t  = torch.tensor(lbl_list, device=device)
            pred_t = torch.tensor(pred_list, device=device)

            acc_meter.add((pred_t == lbl_t).float().mean().item())

            # ---------- 累积 IoU ----------
            for c in range(n_cls):
                m_pred = pred_t == c
                m_lbl  = lbl_t  == c
                # 忽略 union 为 0 的类别（没有样本）
                if torch.any(m_pred | m_lbl):
                    inter[c] += torch.sum(m_pred & m_lbl)
                    union[c] += torch.sum(m_pred | m_lbl)
            # —— 累计混淆矩阵 —— #
            conf_mat += torch.bincount(
                lbl_t * n_cls + pred_t,
                minlength=n_cls * n_cls
            ).reshape(n_cls, n_cls)
        # ---------- 打印 step ----------
        if (i + 1) % config.display_step == 0:
            print(f"Step [{i+1}/{len(data_loader)}] "
                  f"Loss {loss_meter.value()[0]:.4f} "
                  f"Acc {acc_meter.value()[0]*100:.2f}%")

    # ---------- 计算 epoch 级指标 ----------
    miou = torch.mean(inter / union.clamp(min=1)).item()   # mean IoU

    metrics = {
        f"{mode}_loss":      loss_meter.value()[0],
        f"{mode}_accuracy":  acc_meter.value()[0],
        f"{mode}_IoU":       miou
    }

    return metrics,conf_mat



def recursive_todevice(x, device):
    # 1) 如果某个字段本来就是 None（比如 pixel_label=None），直接返回 None
    if x is None:
        return None

    # 2) Tensor 直接 .to(device)
    if isinstance(x, torch.Tensor):
        return x.to(device)

    # 3) dict 递归
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}

    # 4) list/tuple 递归
    elif isinstance(x, (list, tuple)):
        return [recursive_todevice(c, device) for c in x]

    # 5) 其它类型（int/float/str 等）原样返回
    else:
        return x



def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"), "wb"
        ),
    )


def overall_performance(config):
    cm = np.zeros((config.num_classes, config.num_classes))
    for fold in range(1, 6):
        cm += pkl.load(
            open(
                os.path.join(config.res_dir, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))

    with open(os.path.join(config.res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    fold_sequence = [
        [[1, 2, 3], [4], [4]],
        [[2, 3, 4], [5], [5]],
        [[3, 4, 5], [1], [1]],
        [[4, 5, 1], [2], [2]],
        [[5, 1, 2], [3], [3]],
    ]

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Dataset definition
        dt_args = dict(
            root=config.dataset_folder,
            pid_dir=os.path.join(config.dataset_folder,r"C:\Shandong\sddataset\parcel"),
            norm=False,
            reference_date=config.ref_date,

            mono_date=config.mono_date,
            sats=["S1"],  # Sentinel-2 就写 "S2"
            use_pixel_label=False  # 如果你想开像元辅助分支就设 True
        )
        dt_train = ParcelPASTIS(**dt_args, folds=train_folds, cache=config.cache)

        # dt_train = PASTIS_Dataset(**dt_args, folds=train_folds, cache=config.cache)
        dt_val =  ParcelPASTIS(**dt_args, folds=val_fold, cache=config.cache)
        dt_test = ParcelPASTIS(**dt_args, folds=test_fold)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            collate_fn=parcel_collate,
        )
        val_loader = data.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=parcel_collate,
        )
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=parcel_collate,
        )

        print(
            "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
        )

        # Model definition
        model = model_utils.get_model(config, mode="semantic")
        config.N_params = utils.get_ntrainparams(model)
        with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))
        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        model = model.to(device)
        model.apply(weight_init)

        # Optimizer and Loss
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.01  # 最低学习率
        )

        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        # criterion = nn.CrossEntropyLoss(weight=weights)

        # Training loop
        trainlog = {}
        best_mIoU = 0
        for epoch in range(1, config.epochs + 1):
            print("EPOCH {}/{}".format(epoch, config.epochs))

            model.train()
            train_metrics,conf_mat = iterate(
                model,
                data_loader=train_loader,
                # criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="train",
                device=device,
            )
            if epoch % config.val_every == 0 and epoch > config.val_after:
                print("Validation . . . ")
                model.eval()
                val_metrics,conf_mat = iterate(
                    model,
                    data_loader=val_loader,
                    # criterion=criterion,
                    config=config,
                    optimizer=optimizer,
                    mode="val",
                    device=device,
                )

                print(
                    "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                        val_metrics["val_loss"],
                        val_metrics["val_accuracy"],
                        val_metrics["val_IoU"],
                    )
                )

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config)
                if val_metrics["val_IoU"] >= best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(
                            config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                        ),
                    )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(fold + 1, trainlog, config)
            scheduler.step()
        best_epoch = 0
        no_improve = 0
        for epoch in range(1, config.epochs + 1):
            ...
            if epoch % config.val_every == 0 and epoch > config.val_after:
                ...
                if val_metrics["val_IoU"] > best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    best_epoch = epoch
                    print(best_epoch)
                    no_improve = 0
                    # 保存 best 模型的代码保留原状
                else:
                    no_improve += 1

            # # ---------- Early-Stopping ----------
            # if no_improve >= config.patience:
            #     print(f"No val IoU improvement for {config.patience} epochs, early stop at epoch {epoch}.")
            #     break

        print("Testing best epoch . . .")
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                )
            )["state_dict"]
        )
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            # criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="test",
            device=device,
        )
        print(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)

    if config.fold is None:
        overall_performance(config)


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    assert config.num_classes == config.out_conv[-1]

    pprint.pprint(config)
    main(config)
