"""
Channel-Swin-Unet: Channel-ViT variant of Swin-Unet.

Each input channel gets its own patch token sequence. Within each spatial window,
tokens from all C_in input channels attend to each other, enabling cross-channel
as well as spatial attention.

Internal token ordering throughout: (h, w, c) with c varying fastest,
so token index t = h*W'*C_in + w*C_in + c.

Resolution-agnostic at inference time: forward() derives the patch grid (H, W) from
the actual input tensor shape (see ChannelSwinTransformerSys), instead of relying on a
fixed resolution baked in at construction time. This lets a model trained at a fixed
img_size (e.g. 224) be run directly on larger images without tiling, using the same
learned window-attention weights (which depend only on window_size, not H, W).
Checkpoints trained before this change still load (state_dict keys/shapes unchanged).
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ---------------------------------------------------------------------------
# Utility: MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Utility: spatial-only window partition (used for shift-mask construction)
# ---------------------------------------------------------------------------

def _spatial_window_partition(x, window_size):
    """x: (B, H, W, 1) → (nW*B, window_size, window_size, 1)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# ---------------------------------------------------------------------------
# Utility: channel-aware window partition / reverse
# ---------------------------------------------------------------------------

def channel_window_partition(x, window_size):
    """
    x: (B, H, W, C_in, D)
    Returns: (nW*B, window_size, window_size, C_in, D)
    """
    B, H, W, C_in, D = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C_in, D)
    windows = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
    windows = windows.view(-1, window_size, window_size, C_in, D)
    return windows


def channel_window_reverse(windows, window_size, H, W):
    """
    windows: (nW*B, window_size, window_size, C_in, D)
    Returns: (B, H, W, C_in, D)
    """
    C_in = windows.shape[3]
    D = windows.shape[4]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C_in, D)
    x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous().view(B, H, W, C_in, D)
    return x


# ---------------------------------------------------------------------------
# ChannelWindowAttention
# ---------------------------------------------------------------------------

class ChannelWindowAttention(nn.Module):
    """Window-based multi-head self-attention with spatial+channel position bias.

    Operates on tokens (num_windows*B, Wh*Ww*C_in, D).
    Relative position bias is factored:
        bias(Δh, Δw, Δc) = spatial_bias[Δh, Δw] + channel_bias[Δc]

    Resolution-agnostic: all parameters depend only on window_size and in_chans,
    never on the full feature-map resolution.
    """

    def __init__(self, dim, window_size, in_chans, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.in_chans = in_chans
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Spatial relative position bias table: (2Wh-1)*(2Ww-1), nH
        self.spatial_rp_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Channel relative position bias table: 2*C_in-1, nH
        self.channel_rp_bias_table = nn.Parameter(
            torch.zeros(2 * in_chans - 1, num_heads))

        # Spatial relative position index: (Wh*Ww, Wh*Ww)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, Wh, Ww)
        coords_flat = torch.flatten(coords, 1)  # (2, Wh*Ww)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]   # (2, L, L)
        rel = rel.permute(1, 2, 0).contiguous()                   # (L, L, 2)
        rel[:, :, 0] += self.window_size[0] - 1
        rel[:, :, 1] += self.window_size[1] - 1
        rel[:, :, 0] *= 2 * self.window_size[1] - 1
        spatial_rp_idx = rel.sum(-1)  # (L, L)
        self.register_buffer("spatial_rp_idx", spatial_rp_idx)

        # Channel relative position index: (C_in, C_in)
        c = torch.arange(in_chans)
        channel_rp_idx = (c[:, None] - c[None, :]) + in_chans - 1  # range [0, 2*C_in-2]
        self.register_buffer("channel_rp_idx", channel_rp_idx)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.spatial_rp_bias_table, std=.02)
        trunc_normal_(self.channel_rp_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x: (nW*B, Wh*Ww*C_in, D)
        mask: (nW, Wh*Ww*C_in, Wh*Ww*C_in) or None
        """
        B_, N, C = x.shape
        L = self.window_size[0] * self.window_size[1]
        C_in = self.in_chans
        nH = self.num_heads

        qkv = self.qkv(x).reshape(B_, N, 3, nH, C // nH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, nH, N, N)

        # --- Factored relative position bias ---
        # Spatial bias: (nH, L, L) → expanded to (nH, L*C_in, L*C_in)
        s_bias = self.spatial_rp_bias_table[self.spatial_rp_idx.view(-1)].view(L, L, nH)
        s_bias = s_bias.permute(2, 0, 1)                                    # (nH, L, L)
        s_bias = s_bias.unsqueeze(2).unsqueeze(4).expand(-1, -1, C_in, -1, C_in)  # (nH, L, C_in, L, C_in)
        s_bias = s_bias.reshape(nH, N, N)

        # Channel bias: (nH, C_in, C_in) → expanded to (nH, L*C_in, L*C_in)
        c_bias = self.channel_rp_bias_table[self.channel_rp_idx.view(-1)].view(C_in, C_in, nH)
        c_bias = c_bias.permute(2, 0, 1)                                    # (nH, C_in, C_in)
        c_bias = c_bias.unsqueeze(1).unsqueeze(3).expand(-1, L, -1, L, -1)  # (nH, L, C_in, L, C_in)
        c_bias = c_bias.reshape(nH, N, N)

        attn = attn + s_bias.unsqueeze(0) + c_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, nH, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, nH, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# ChannelSwinTransformerBlock
# ---------------------------------------------------------------------------

class ChannelSwinTransformerBlock(nn.Module):
    """Swin Transformer Block with cross-channel attention.

    forward(x, H, W): (B, H*W*C_in, D) — channel dimension interleaved with spatial.
    Cyclic shift operates on spatial dims only; channel dim is never shifted.

    Resolution-agnostic: `input_resolution` is used only to build the registered
    (checkpoint-compatible) default shift-mask buffer. Any (H, W) divisible by
    window_size can be passed to forward(); if it differs from the default
    resolution, the mask is rebuilt on the fly (pure geometry, no learned params).
    """

    def __init__(self, dim, input_resolution, in_chans, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.default_resolution = input_resolution  # (H', W') spatial
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = ChannelWindowAttention(
            dim, window_size=to_2tuple(self.window_size), in_chans=in_chans,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self._build_attn_mask(*input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _build_attn_mask(self, H, W, device=None):
        """Build the channel-expanded SW-MSA mask for an arbitrary (H, W).
        Pure geometry, no learned parameters -- safe to recompute on the fly."""
        C_in = self.in_chans
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_wins = _spatial_window_partition(img_mask, self.window_size)  # (nW, Wh, Ww, 1)
        mask_wins = mask_wins.view(-1, self.window_size * self.window_size)  # (nW, L)
        sp_mask = mask_wins.unsqueeze(1) - mask_wins.unsqueeze(2)  # (nW, L, L)
        sp_mask = sp_mask.masked_fill(sp_mask != 0, -100.0).masked_fill(sp_mask == 0, 0.0)

        # Expand spatial mask to channel dimension:
        # For token pair (h1,w1,c1)-(h2,w2,c2): mask depends only on spatial positions
        nW, L, _ = sp_mask.shape
        attn_mask = sp_mask[:, :, None, :, None].expand(nW, L, C_in, L, C_in)
        attn_mask = attn_mask.reshape(nW, L * C_in, L * C_in)
        return attn_mask

    def forward(self, x, H, W):
        C_in = self.in_chans
        B, L_total, D = x.shape
        assert L_total == H * W * C_in

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C_in, D)  # (B, H, W, C_in, D)

        # Cyclic shift — spatial dims only
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if (H, W) == tuple(self.default_resolution):
                attn_mask = self.attn_mask
            else:
                attn_mask = self._build_attn_mask(H, W, device=x.device)
        else:
            attn_mask = None

        # Window partition: (B, H, W, C_in, D) → (nW*B, Wh*Ww*C_in, D)
        x_wins = channel_window_partition(x, self.window_size)  # (nW*B, Wh, Ww, C_in, D)
        x_wins = x_wins.view(-1, self.window_size * self.window_size * C_in, D)

        # Attention
        attn_wins = self.attn(x_wins, mask=attn_mask)  # (nW*B, Wh*Ww*C_in, D)

        # Reverse partition: → (B, H, W, C_in, D)
        attn_wins = attn_wins.view(-1, self.window_size, self.window_size, C_in, D)
        x = channel_window_reverse(attn_wins, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W * C_in, D)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# ChannelPatchMerging  (spatial 2×2 merge, C_in preserved)
# ---------------------------------------------------------------------------

class ChannelPatchMerging(nn.Module):
    """Downsample by 2× in spatial dimensions; preserve channel count C_in.

    Resolution-agnostic: no learned parameter depends on H, W.
    """

    def __init__(self, dim, in_chans, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.in_chans = in_chans
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        C_in = self.in_chans
        B, L, D = x.shape
        assert L == H * W * C_in

        x = x.view(B, H, W, C_in, D)
        x0 = x[:, 0::2, 0::2, :, :]  # (B, H/2, W/2, C_in, D)
        x1 = x[:, 1::2, 0::2, :, :]
        x2 = x[:, 0::2, 1::2, :, :]
        x3 = x[:, 1::2, 1::2, :, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, C_in, 4D)
        x = x.view(B, (H // 2) * (W // 2) * C_in, 4 * D)

        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------------------------------------------------------
# ChannelPatchExpand  (spatial ×2 upsample, C_in preserved)
# ---------------------------------------------------------------------------

class ChannelPatchExpand(nn.Module):
    """Upsample by 2× in spatial dimensions; preserve channel count C_in.

    Resolution-agnostic: no learned parameter depends on H, W.
    """

    def __init__(self, dim, in_chans, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.in_chans = in_chans
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, H, W):
        C_in = self.in_chans
        x = self.expand(x)          # (B, H*W*C_in, 2D)
        B, L, C = x.shape
        assert L == H * W * C_in

        # Split 2D into (p1=2, p2=2, D//2) and interleave with spatial dims
        x = x.view(B, H, W, C_in, 2, 2, C // 4)       # (B, H, W, C_in, 2, 2, D//2)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()  # (B, H, 2, W, 2, C_in, D//2)
        x = x.view(B, 2 * H, 2 * W, C_in, C // 4)     # (B, 2H, 2W, C_in, D//2)
        x = x.reshape(B, 2 * H * 2 * W * C_in, C // 4)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# ChannelFinalPatchExpand_X4  (spatial ×4, then aggregate C_in → 1)
# ---------------------------------------------------------------------------

class ChannelFinalPatchExpand_X4(nn.Module):
    """×4 spatial expansion followed by channel aggregation (mean over C_in).

    Resolution-agnostic: no learned parameter depends on H, W.
    """

    def __init__(self, dim, in_chans, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.in_chans = in_chans
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, H, W):
        C_in = self.in_chans
        x = self.expand(x)          # (B, H*W*C_in, 16D)
        B, L, C = x.shape
        assert L == H * W * C_in

        # Split 16D into (p1=4, p2=4, D) and interleave with spatial dims
        x = x.view(B, H, W, C_in, 4, 4, C // 16)        # (B, H, W, C_in, 4, 4, D)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()   # (B, H, 4, W, 4, C_in, D)
        x = x.view(B, 4 * H, 4 * W, C_in, C // 16)       # (B, 4H, 4W, C_in, D)

        # Aggregate over input channels
        x = x.mean(dim=3)                                   # (B, 4H, 4W, D)
        x = x.reshape(B, 4 * H * 4 * W, self.output_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# ChannelBasicLayer  (encoder stage)
# ---------------------------------------------------------------------------

class ChannelBasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, in_chans, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.in_chans = in_chans
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            ChannelSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, in_chans=in_chans,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, in_chans=in_chans, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2
        return x, H, W


# ---------------------------------------------------------------------------
# ChannelBasicLayer_up  (decoder stage)
# ---------------------------------------------------------------------------

class ChannelBasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, in_chans, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.in_chans = in_chans
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            ChannelSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, in_chans=in_chans,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = ChannelPatchExpand(dim=dim, in_chans=in_chans,
                                               dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.upsample is not None:
            x = self.upsample(x, H, W)
            H, W = H * 2, W * 2
        return x, H, W


# ---------------------------------------------------------------------------
# ChannelPatchEmbed  (per-channel Conv + channel position embedding)
# ---------------------------------------------------------------------------

class ChannelPatchEmbed(nn.Module):
    """Each input channel gets its own Conv2d patch projection.

    Returns (B, H'*W'*C_in, D) with token ordering (h, w, c), c varying fastest.
    A learnable channel embedding is added to distinguish channels.

    The exact-size assert is relaxed: any H, W divisible by patch_size works.
    `img_size`/`patches_resolution` are kept as defaults (informational / used to
    size the default shift-mask buffers elsewhere); forward() derives the actual
    patch grid from the real input shape.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=4, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # One Conv2d per input channel
        self.projs = nn.ModuleList([
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(in_chans)])

        # Learnable channel position embedding: (C_in, D)
        self.channel_embed = nn.Parameter(torch.zeros(in_chans, embed_dim))
        trunc_normal_(self.channel_embed, std=.02)

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input image size ({H}*{W}) must be divisible by patch_size {self.patch_size}."
        assert C == self.in_chans

        tokens = []
        for i, proj in enumerate(self.projs):
            t = proj(x[:, i:i + 1])              # (B, D, Ph, Pw)
            t = t.flatten(2).transpose(1, 2)     # (B, Ph*Pw, D)
            t = t + self.channel_embed[i]         # add channel embedding
            tokens.append(t)

        # tokens: list of C_in tensors, each (B, L, D)
        # Stack → (B, L, C_in, D) → reshape to (B, L*C_in, D)  (c varies fastest)
        x = torch.stack(tokens, dim=2)           # (B, L, C_in, D)
        B, L, C_in, D = x.shape
        x = x.reshape(B, L * C_in, D)            # (B, L*C_in, D)

        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# ChannelSwinTransformerSys  (full encoder-decoder)
# ---------------------------------------------------------------------------

class ChannelSwinTransformerSys(nn.Module):
    """Channel-Swin-Unet encoder-decoder with skip connections.

    Token layout throughout: (B, H'*W'*C_in, D).

    Resolution-agnostic at inference time: forward() derives the patch grid (H, W)
    from the actual input tensor shape and threads it through encoder/decoder/up_x4.
    H, W (and every intermediate stage resolution) must be divisible by window_size;
    equivalently, input image H, W must be a multiple of `size_divisor`. The module's
    learned parameters (and therefore checkpoints trained at a fixed img_size) are
    unaffected by this change.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=4, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 2, 2), depths_decoder=(1, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, use_checkpoint=False,
                 final_upsample="expand_first", **kwargs):
        super().__init__()

        print(f"ChannelSwinTransformerSys: depths={depths} depths_decoder={depths_decoder} "
              f"drop_path={drop_path_rate} num_classes={num_classes} in_chans={in_chans}")

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.in_chans = in_chans
        self.patch_size = to_2tuple(patch_size)[0]
        self.window_size = window_size
        # Required divisor of the input image H, W for direct (non-tiled) inference.
        self.size_divisor = self.patch_size * window_size * (2 ** (self.num_layers - 1))

        # Patch embedding
        self.patch_embed = ChannelPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if ape:
            n = self.patch_embed.num_patches * in_chans
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ChannelBasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(patches_resolution[0] // (2 ** i),
                                   patches_resolution[1] // (2 ** i)),
                in_chans=in_chans,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=ChannelPatchMerging if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # Decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_layers):
            lvl = self.num_layers - 1 - i          # 3, 2, 1, 0
            D_lvl = int(embed_dim * 2 ** lvl)
            res = (patches_resolution[0] // (2 ** lvl),
                   patches_resolution[1] // (2 ** lvl))

            concat_linear = nn.Linear(2 * D_lvl, D_lvl) if i > 0 else nn.Identity()

            if i == 0:
                # Bottleneck: just expand spatial, no Swin blocks
                layer_up = ChannelPatchExpand(
                    dim=D_lvl, in_chans=in_chans, dim_scale=2,
                    norm_layer=norm_layer)
            else:
                layer_up = ChannelBasicLayer_up(
                    dim=D_lvl, input_resolution=res, in_chans=in_chans,
                    depth=depths[lvl],
                    num_heads=num_heads[lvl],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:lvl]):sum(depths[:lvl + 1])],
                    norm_layer=norm_layer,
                    upsample=ChannelPatchExpand if (i < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)

        if final_upsample == "expand_first":
            self.up = ChannelFinalPatchExpand_X4(
                dim=embed_dim, in_chans=in_chans, dim_scale=4)
            self.output = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'spatial_rp_bias_table', 'channel_rp_bias_table', 'channel_embed'}

    def forward_features(self, x):
        B, C, H_img, W_img = x.shape
        assert H_img % self.size_divisor == 0 and W_img % self.size_divisor == 0, \
            f"input size ({H_img}x{W_img}) must be a multiple of {self.size_divisor} " \
            f"for direct (non-tiled) inference."

        x = self.patch_embed(x)   # (B, H'*W'*C_in, D)
        H = H_img // self.patch_size
        W = W_img // self.patch_size

        if self.ape:
            assert (H, W) == tuple(self.patches_resolution), \
                "absolute position embedding requires the default resolution"
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x, H, W = layer(x, H, W)

        x = self.norm(x)
        return x, x_downsample, H, W

    def forward_up_features(self, x, x_downsample, H, W):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x, H, W)  # bare ChannelPatchExpand: returns x only
                H, W = H * 2, W * 2
            else:
                x = torch.cat([x, x_downsample[3 - inx]], dim=-1)
                x = self.concat_back_dim[inx](x)
                x, H, W = layer_up(x, H, W)

        x = self.norm_up(x)
        return x, H, W

    def up_x4(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W * self.in_chans

        # ChannelFinalPatchExpand_X4 aggregates C_in → outputs (B, 4H*4W, D)
        x = self.up(x, H, W)        # (B, 4H*4W, D)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # (B, D, 4H, 4W)
        x = self.output(x)          # (B, num_classes, H_orig, W_orig)
        return x

    def forward(self, x):
        x, x_downsample, H, W = self.forward_features(x)
        x, H, W = self.forward_up_features(x, x_downsample, H, W)
        x = self.up_x4(x, H, W)
        return x


# ---------------------------------------------------------------------------
# ChannelSwinUnet  (top-level wrapper, mirrors SwinUnet in vision_transformer.py)
# ---------------------------------------------------------------------------

class ChannelSwinUnet(nn.Module):
    """Top-level wrapper for Channel-Swin-Unet.

    Unlike SwinUnet, no grayscale→RGB repeat is performed; the 4 input channels
    are kept distinct throughout the network.
    """

    def __init__(self, config, img_size=224, num_classes=21843):
        super().__init__()
        self.num_classes = num_classes
        self.config = config

        self.swin_unet = ChannelSwinTransformerSys(
            img_size=img_size,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            depths_decoder=config.MODEL.SWIN.DECODER_DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        # x: (B, C_in, H, W)
        return self.swin_unet(x)
