# -------------------------------------------------------------------------
# src/backbones/ltae_stsa.py
#  → 保留原 LTAE2d 的所有代码，新增“空间分支 + 融合”几行即可
# -------------------------------------------------------------------------
import copy, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F

from src.backbones.ltae import LTAE2d
from src.backbones.positional_encoding import PositionalEncoder

# ====== 复用原来的 ScaledDotProductAttention / MultiHeadAttention =========
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

# -------------------------------------------------------------------------
class ParallelSTSA_LTAE2d(nn.Module):
    """
    L-TAE + 并行 STSA：
      • 时间自注意力 = 原 LTAE2d
      • 空间自注意力：先对 T 做平均 → [B,C,H,W] → flatten 成 [B,N,C]，
        用 nn.MultiheadAttention 计算空间依赖
      • 拼接时空特征 → 1×1 Conv & MLP 融合
    输出 shape 与旧版保持一致；return_att 只返回 “时间注意力”。
    """
    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=4,
                 mlp=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 T=1000,
                 return_att=False,
                 positional_encoding=True,
                 spa_heads=4):                # ← 新增：空间注意力 head 数
        super().__init__()
        # ---------- 可学习二维位置编码 (Row + Col 分解) ----------
        # 行列各自一个向量，可适配任意 H,W；总维度 = d_model
        self.pos_row = nn.Parameter(torch.randn(1, d_model // 2, 1, 1))
        self.pos_col = nn.Parameter(torch.randn(1, d_model // 2, 1, 1))
        # 如果 d_model 不是偶数，额外补一维到 pos_col
        if (d_model % 2) != 0:
            extra = nn.Parameter(torch.randn(1, 1, 1, 1))
            self.pos_col = nn.Parameter(torch.cat([self.pos_col, extra], dim=1))

        self.return_att = return_att
        # ---------- 原 LTAE 初始化 ----------
        self.ltae_core = LTAE2d(            # 直接实例化原 LTAE2d 作时间分支
            in_channels, n_head, d_k, mlp, dropout,
            d_model, T, return_att=True, positional_encoding=positional_encoding
        )
        out_channels = mlp[-1]              # = 时间分支输出通道数

        # ---------- 空间注意力分支 ----------
        self.spa_norm   = nn.GroupNorm( num_groups=spa_heads, num_channels=in_channels )
        self.q_proj     = nn.Conv2d(in_channels, d_model, 1)
        self.mha_space  = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=spa_heads,
                                                batch_first=True)
        # ---------- 融合 & 输出 ----------
        self.fuse_conv  = nn.Conv2d(out_channels + d_model, out_channels, 1)
        self.fuse_mlp   = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    # ---------------------------------------------------------------------
    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        """
        x : [B,T,C,H,W]   (与旧版一致)
        """
        # ===== 时间注意力（原 LTAE2d） =====================================
        time_feat, attn = self.ltae_core(
            x, batch_positions=batch_positions,
            pad_mask=pad_mask, return_comp=False
        )   # time_feat : [B,C',H,W]  C' = mlp[-1]

        # ===== 空间注意力 ===============================================
        B, T, C, H, W = x.shape
        spa_in = x.mean(dim=1)                   # [B,C,H,W]  ← 对 T 求 mean
        spa_in = self.spa_norm(spa_in)           # 归一化
        # ---- 注入可学习 Row/Col 位置编码 ----
        pos_h = self.pos_row.expand(-1, -1, spa_in.size(2), spa_in.size(3))
        pos_w = self.pos_col.expand(-1, -1, spa_in.size(2), spa_in.size(3))
        spa_in = spa_in + torch.cat([pos_h, pos_w], dim=1)[:, :spa_in.size(1)]

        spa_in = self.q_proj(spa_in)             # [B,d_model,H,W]
        N = H * W
        spa_seq = spa_in.flatten(2).transpose(1,2)  # [B,N,d_model]

        # MultiheadAttention 要 (B,N,E) -> (B,N,E)
        spa_out, _ = self.mha_space(spa_seq, spa_seq, spa_seq)  # 自注意
        spa_out = spa_out.transpose(1,2).reshape(B, -1, H, W)   # [B,d_model,H,W]

        # ===== 时空融合  ===============================================
        fused = torch.cat([time_feat, spa_out], dim=1)          # [B,C'+d_model,H,W]
        fused = self.fuse_conv(fused)
        fused = self.fuse_mlp(fused) + time_feat                # 残差到时间分支

        if self.return_att:
            return fused, attn        # 只回原时间注意力
        else:
            return fused
