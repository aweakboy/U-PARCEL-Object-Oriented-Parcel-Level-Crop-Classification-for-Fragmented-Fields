import torch
import torch.nn as nn


from src.backbones.utae import UTAE                               # 你之前的代码
from src.backbones.parcel_pool import ParcelPooling

class UTAEParcel(nn.Module):
    """
    U-TAE backbone + ParcelPooling + parcel classifier (+ 可选像元分割头)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone_kwargs: dict = None,
        embed_dim: int = 256,
        use_pixel_head: bool = True,
        lambda_pix: float = 0.3,
        ignore_index: int = -1,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = UTAE(
            input_dim=in_channels,
            encoder=True,              # 只要特征
            **backbone_kwargs
        )
        enc_dim = self.backbone.enc_dim  # 最顶层通道数
        self.ignore_index = ignore_index
        self.pool = ParcelPooling()
        self.head = nn.Sequential(
            nn.Linear(enc_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

        # 可选：像元级辅助头（复用 backbone 最后特征图）
        self.use_pixel_head = use_pixel_head
        if use_pixel_head:
            self.pixel_head = nn.Conv2d(enc_dim, num_classes, 1)
            self.lambda_pix = lambda_pix
        # ---------- Weighted FocalLoss ----------
        import os, torch
        from src.losses import WeightedFocalLoss
                ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        w_path = os.path.join(ROOT_DIR, "class_weights_all.pt")
        # 类别频次: [6180, 16750, 20469, 30560, 83668]
        # 最终权重: ['1.422', '1.054', '0.993', '0.880', '0.651']
                if os.path.exists(w_path):
            alpha = torch.load(w_path)
        else:
            alpha = torch.ones(num_classes, device="cpu")

        # alpha=torch.ones(6)
        print(alpha)
        self.parcel_loss = WeightedFocalLoss(
            alpha=alpha,
            gamma=1.0,                      # 可放进 config 调参
            ignore_index=ignore_index)
        # ★ 复用同一权重做像元损失//新增
        self.pixel_loss = self.parcel_loss  # <─ NEW

        self.lambda_pix = lambda_pix  # <─ 已有就保留

    def forward(self, imgs, batch_positions, pid_raster, targets=None):
        """
        imgs   : [B,T,C,H,W]   – 时序影像
        pid    : [B,H',W']     – 与 backbone output 分辨率一致的 parcel_id
        """
        feat, _ = self.backbone(imgs, batch_positions)   # [B,C,H',W']
        P_emb, batch_idx, pid_vec = self.pool(feat, pid_raster)

        logits_parcel = self.head(P_emb)                 # [P,num_cls]

        out = {
            "parcel_logits": logits_parcel,
            "parcel_batch" : batch_idx,
            "parcel_id"    : pid_vec
        }

        # ---------- 像元辅助 ----------
        if self.use_pixel_head:
            out["pixel_logits"] = self.pixel_head(feat)

        # ---------- loss ----------
        # if targets is not None:
        #     loss = self._loss(out, targets)
        #     out["loss"] = loss
        #
        # return out

        if targets is not None:
            loss_dict = self._loss(out, targets)  # <─ 接收 dict
            out.update(loss_dict)  # out["total"] 等
        return out




    def _loss(self, out, targets):
        device = out["parcel_logits"].device
        batch_idx, pid_vec = out["parcel_batch"], out["parcel_id"]

        # ---------- parcel focal loss ----------
        logits_list, label_list = [], []
        for b, tgt in enumerate(targets):
            mask = batch_idx == b
            pids = pid_vec[mask]
            log_b = out["parcel_logits"][mask]

            id2lab = dict(zip(
                tgt["parcel_id_vec"].tolist(),
                tgt["parcel_label"].tolist()
            ))

            keep, labs = [], []
            for i, pid in enumerate(pids):
                lab = id2lab.get(int(pid))
                if lab is not None:
                    keep.append(i)
                    labs.append(lab)

            if keep:
                logits_list.append(log_b[keep])
                label_list.append(torch.tensor(labs, device=device))

        if logits_list:
            lp = self.parcel_loss(
                torch.cat(logits_list, 0),
                torch.cat(label_list, 0)
            )
        else:
            lp = torch.tensor(0., device=device)

        # ---------- pixel focal loss ----------
        lpx = torch.tensor(0., device=device)
        if self.use_pixel_head and self.lambda_pix > 0 and "pixel_label" in targets[0]:
            pix_logits = out["pixel_logits"].permute(0, 2, 3, 1)  # B,H,W,C
            pix_gt = torch.stack([t["pixel_label"] for t in targets]).to(device)
            valid = pix_gt != self.ignore_index
            if valid.any():
                lpx = self.pixel_loss(pix_logits[valid], pix_gt[valid])

        total = lp + self.lambda_pix * lpx
        return {"total": total, "parcel": lp, "pixel": lpx}
