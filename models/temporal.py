import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint


class QueryDrivenTemporalFusion(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size_t: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        history_downsample_ratio: int = 4,
        chunk_size: int = 512,              
        use_checkpoint: bool = False,       
        use_flash_attn: bool = True,     
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size_t = window_size_t
        
        self.history_downsample_ratio = history_downsample_ratio
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')

        if history_downsample_ratio > 1:
            self.history_pool = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
            )
        else:
            self.history_pool = nn.Identity()

        self.motion_hint_mlp = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim)
        )
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.history_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=qkv_bias,
        )
        self.history_norm = nn.LayerNorm(dim)
        self.history_proj = nn.Linear(dim, dim)

        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cross_attn_drop = nn.Dropout(attn_drop)
        self.cross_proj = nn.Linear(dim, dim)
        self.cross_drop = nn.Dropout(proj_drop)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(proj_drop),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def _downsample_history(self, x: torch.Tensor) -> torch.Tensor:
        if self.history_downsample_ratio <= 1:
            return x
        
        B, T, N, D = x.shape
        ratio = self.history_downsample_ratio
        N_down = N // ratio
        
        x_down = x[:, :, ::ratio, :]
        x_down = self.history_pool(x_down)
        return x_down

    def _window_partition(self, x: torch.Tensor, shift: bool = False):
        B, T, N, D = x.shape
        if shift and T > 1:
            x = torch.roll(x, shifts=-1, dims=1)

        windows = []
        for i in range(0, T, self.window_size_t):
            end = min(i + self.window_size_t, T)
            win = x[:, i:end].reshape(B, -1, D)
            windows.append(win)
        return torch.cat(windows, dim=1)

    def _encode_history_inner(self, z: torch.Tensor) -> torch.Tensor:
        z = self.history_norm(z)
        z, _ = self.history_attn(z, z, z)
        z = self.history_proj(z)
        return z

    def encode_history(self, x_hist: torch.Tensor):
        x_hist = self._downsample_history(x_hist)
        
        z_reg = self._window_partition(x_hist, shift=False)
        if self.use_checkpoint and self.training:
            z_reg = checkpoint(self._encode_history_inner, z_reg, use_reentrant=False)
        else:
            z_reg = self._encode_history_inner(z_reg)

        return z_reg

    def _chunked_cross_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        B, H, N, head_dim = Q.shape
        M = K.shape[2]
        scale = head_dim ** -0.5
        
        if N * M <= self.chunk_size * self.chunk_size or self.use_flash_attn:
            if self.use_flash_attn:
                out = F.scaled_dot_product_attention(
                    Q, K, V,
                    dropout_p=self.cross_attn_drop.p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                attn = (Q @ K.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attn = self.cross_attn_drop(attn)
                out = attn @ V
            return out
        
        chunk_size = self.chunk_size
        outputs = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            Q_chunk = Q[:, :, i:end_i, :]
            
            attn_chunk = (Q_chunk @ K.transpose(-2, -1)) * scale
            attn_chunk = attn_chunk.softmax(dim=-1)
            attn_chunk = self.cross_attn_drop(attn_chunk)
            out_chunk = attn_chunk @ V 
            outputs.append(out_chunk)
        
        return torch.cat(outputs, dim=2)

    def forward(
        self,
        f_hist: torch.Tensor,
        f_curr: torch.Tensor, 
        f_hist_last: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D, N = f_hist.shape
        assert f_curr.shape == (B, D, N)

        x_hist = f_hist.permute(0, 1, 3, 2).contiguous()
        z_hist = self.encode_history(x_hist)

        f_curr_t = f_curr.permute(0, 2, 1)  # [B, N, D]

        if f_hist_last is not None:
            delta = f_curr - f_hist_last
            motion_hint = self.motion_hint_mlp(delta.mean(-1))
            Q_input = f_curr_t + motion_hint.unsqueeze(1)
        else:
            Q_input = f_curr_t

        Q = self.norm_q(Q_input)
        Q = self.q_proj(Q)

        KV = self.norm_kv(z_hist)
        KV = self.kv_proj(KV).reshape(B, -1, 2, self.num_heads, D // self.num_heads)
        KV = KV.permute(2, 0, 3, 1, 4)
        K, V = KV[0], KV[1]

        Q = Q.reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)

        out = self._chunked_cross_attention(Q, K, V)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        out = self.cross_proj(out)
        out = self.cross_drop(out)

        f_out = f_curr_t + out
        f_out = f_out + self.ffn(self.ffn_norm(f_out))
        
        return f_out.permute(0, 2, 1).contiguous()