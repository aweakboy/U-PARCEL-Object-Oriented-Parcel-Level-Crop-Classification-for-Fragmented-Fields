import torch
from torch_scatter import scatter_add

class ParcelPooling(torch.nn.Module):
    """
    Masked average-pooling on variable-size parcels (polygons).

    Inputs
    ------
    feat   : FloatTensor  [B, C, H, W]   - backbone feature map
    pid    :   LongTensor [B,   H, W]   - parcel_id raster (0 = background / NODATA)

    Returns
    -------
    v      : FloatTensor  [P, C]        - parcel embeddings（所有 batch 拼一起）
    batch  : LongTensor   [P]           - 对应的 batch 索引（0..B-1）
    pid_out: LongTensor   [P]           - parcel_id（与 v 一一对应）
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feat: torch.Tensor, pid: torch.Tensor):

        B, C, H, W = feat.shape
        feat_ = feat.permute(0, 2, 3, 1).contiguous()      # [B,H,W,C]
        feat_ = feat_.view(-1, C)                          # [B*H*W, C]

        pid_   = pid.view(-1).long()                       # [B*H*W]
        mask   = pid_ > 0
        feat_  = feat_[mask]                               # 只保留 parcel 内像元
        pid_   = pid_[mask]

        # ---- batch 索引：把 0..B*H*W 展成批次号 ----
        grid  = torch.arange(B, device=feat.device).repeat_interleave(H*W)
        batch = grid[mask]                                 # [N_pix]

        # ---- scatter 累加 & 计数 ----
        v_sum = scatter_add(feat_, pid_, dim=0)            # Σ x
        cnt   = scatter_add(torch.ones_like(pid_, dtype=feat_.dtype), pid_, dim=0) # Σ 1
        v_avg = v_sum / (cnt[:,None] + self.eps)           # 平均

        # 取有效 pid 索引（>0 且出现过）
        valid = torch.nonzero(cnt).squeeze(1)
        v     = v_avg[valid]                               # [P,C]
        pid_o = valid                                      # [P] == parcel_id
        # 重新聚合同样的 valid 到 batch
        batch = scatter_add(batch.float(), pid_, dim=0)[valid].long()

        return v, batch, pid_o
