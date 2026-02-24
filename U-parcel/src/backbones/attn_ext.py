# ---------------------------------------------------------------------------
#  src/backbones/attn_ext.py
#  - 额外注意力 / 跨尺度 / PA 模块
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Pixel-wise Attention（深度可分离卷积 + Sigmoid）
# ---------------------------------------------------------------------------
class PA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act  = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.conv(x))


# ---------------------------------------------------------------------------
# 2. tensordot 版本的 Linear（方便一次性投影多维）
# ---------------------------------------------------------------------------
class LinearGeneral(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        in_dim  / out_dim 可以是 tuple，用来表示多轴
        例：in_dim=(C,)，out_dim=(heads, head_dim)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *out_dim))
        self.bias   = nn.Parameter(torch.zeros(*out_dim))

    def forward(self, x, dims):
        # dims: ([x_axes], [w_axes])
        # 与 jax.lax.dot_general / torch.tensordot 行为一致
        return torch.tensordot(x, self.weight, dims) + self.bias


# ---------------------------------------------------------------------------
# 3. 基础 Self-Attention（支持 mask；用于空间/时间两个分支）
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        self.h     = heads
        self.d_h   = dim // heads
        self.scale = self.d_h ** -0.5

        self.q = LinearGeneral((dim,), (heads, self.d_h))
        self.k = LinearGeneral((dim,), (heads, self.d_h))
        self.v = LinearGeneral((dim,), (heads, self.d_h))
        self.proj = LinearGeneral((heads, self.d_h), (dim,))

        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x, mask=None):
        """
        x    : [B, N, C]
        mask : [B, N] (True 表示要被 mask 掉)  or None
        """
        B, N, _ = x.shape
        # [B, h, N, d_h]
        q = self.q(x, dims=([2], [0])).permute(0, 2, 1, 3)
        k = self.k(x, dims=([2], [0])).permute(0, 2, 1, 3)
        v = self.v(x, dims=([2], [0])).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 2:                      # [B,N] -> [B,1,1,N]
                mask = mask[:, None, None, :]
            attn = attn.masked_fill(mask, -1e4)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3)         # [B,N,h,d_h]
        out = self.proj(out, dims=([2, 3], [0, 1]))  # [B,N,C]
        return out                                   # 不返回权重（用不到）


# ---------------------------------------------------------------------------
# 4. STSA_Encoder：空间-时间双 Attention + MLP
# ---------------------------------------------------------------------------
class STSA_Encoder(nn.Module):
    """
    x_space : [B, N=H*W, C]      – 每帧 flatten 得到空间 token
    x_time  : [B, T,      C']     – 将同一像素的时序 flatten
    返回    : [B, N, C]           – 与 x_space 同形状
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0.1):
        super().__init__()
        # 空间、时间分别一层 Self-Attention
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_s = SelfAttention(dim, num_heads, drop)
        self.attn_t = SelfAttention(dim, num_heads, drop)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop)
        )

    def forward(self, x_space, x_time, mask_s=None, mask_t=None):
        # --- 空间注意力 ---
        xs = self.attn_s(self.norm_s(x_space), mask_s) + x_space
        # --- 时间注意力 ---
        xt = self.attn_t(self.norm_t(x_time), mask_t) + x_time      # [B,T,C]
        xt = xt.mean(1, keepdim=True).repeat(1, xs.size(1), 1)      # broadcast
        # --- 融合 + MLP ---
        out = self.mlp(torch.cat([xs, xt], dim=-1)) + xs
        return out                                                  # [B,N,C]


# ---------------------------------------------------------------------------
# 5. Cross-Scale Attention（余弦版，简化）
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q   = nn.Conv2d(dim, dim, 1)
        self.kv  = nn.Conv2d(dim, dim * 2, 1)
        self.proj= nn.Conv2d(dim, dim, 1)

    def forward(self, x_q, x_kv):                 # 两者 [B,C,H,W]
        B, C, H, W = x_q.shape
        q = self.q(x_q).flatten(2)                # [B,C,N]
        k, v = self.kv(x_kv).chunk(2, 1)
        k = k.flatten(2); v = v.flatten(2)

        # 余弦相似度
        attn = torch.einsum('b c n, b c m -> b n m', q, k)
        attn = attn / (q.norm(dim=1, keepdim=True) * k.norm(dim=1, keepdim=True) + 1e-6)
        attn = attn.softmax(-1)

        out = torch.einsum('b n m, b c m -> b c n', attn, v).view(B, C, H, W)
        return self.proj(out) + x_q


# ---------------------------------------------------------------------------
# 6. STSA_TimeEncoder：封装以上模块，保持与 LTAE2d 接口一致
# ---------------------------------------------------------------------------
class STSA_TimeEncoder(nn.Module):
    """
    输入  : feat5d  – [B,T,C,H,W]
    输出  : [B,C,H,W]   （已融合时序 & 加 PA）
    ※ batch_positions 仅占位，为兼容旧接口，可后续加入时戳编码
    """
    def __init__(self, in_channels, n_head=8, mlp_ratio=4., drop=0.1):
        super().__init__()
        self.stsa = STSA_Encoder(
            dim        = in_channels,
            num_heads  = n_head,
            mlp_ratio  = mlp_ratio,
            drop       = drop
        )
        self.pa = PA(in_channels)

    from typing import Optional
    def forward(self,
                feat5d,  # [B,T,C,H,W]
                batch_positions=None,
                pad_mask=None):
        B, T, C, H, W = feat5d.shape

        # ---------- 空间 token ----------
        x_space = feat5d.mean(dim=1)  # [B,C,H,W]   先沿 T 平均
        x_space = x_space.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # ---------- 时间 token ----------
        x_time = feat5d.mean(dim=(3, 4))  # ★ [B,T,C]  ← 只做 H,W GlobalAvg
        # pad_mask 形状 [B,T] 可直接给 STSA

        out = self.stsa(x_space, x_time, None, pad_mask)  # [B,N,C]
        out = out.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]
        return out, None  # att_mask 暂时用 None 占位


