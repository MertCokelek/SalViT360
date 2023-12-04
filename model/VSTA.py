import torch
import torch.nn as nn
from einops import rearrange, repeat


class SphericalVideoTransformer(nn.Module):
    def __init__(self, emb_dim, sph_input_dim, depth=6, num_heads=4, ffdropout=0.0, attn_dropout=0.0, mlp_mult=4.0,
                 **kwargs):
        super().__init__()
        # print('VSTA Depth:', depth)
        self.layer = nn.ModuleList(
            [VSTABlock(emb_dim, num_heads=num_heads, drop=ffdropout, attn_drop=attn_dropout, mlp_ratio=mlp_mult,
                       **kwargs) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

        self.sph_posemb = nn.Linear(sph_input_dim, emb_dim)
        self.tmp_posemb = nn.Parameter(torch.randn(1, 8, emb_dim))

    def forward(self, x, sph, *args):
        """
        x: img embeddings [BS, F, T, emb_dim]
        sph: flattened spherical coordinates. [T, 980]
        """
        spatial_posemb = rearrange(self.sph_posemb(sph), 't d -> 1 1 t d')  # 1, 1, 18, 512
        temporal_posemb = repeat(self.tmp_posemb, '1 f d -> 1 f 1 d')  # 1, 8, 1, 512
        x = x + spatial_posemb + temporal_posemb
        for i, layer_block in enumerate(self.layer):
            x = layer_block(x, *args)
        x = self.norm(x)
        return x


class VSTABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Temporal Attention Layers
        self.temp_norm_layer = norm_layer(dim)
        self.temp_attn = Attention(dim, pos_emb='temporal', num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop=attn_drop, proj_drop=drop)

        # Spatial Attention
        self.spatial_norm_layer = norm_layer(dim)
        self.spatial_attn = Attention(dim, pos_emb='spatial', num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      attn_drop=attn_drop, proj_drop=drop)

        # Final Feed-Forward-Network
        self.FFN = nn.Sequential(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop),
        )


    def forward(self, x, *args):
        N, F, T, D = x.shape  # number of tangent images.
        x = self.temp_attn(self.temp_norm_layer(x), 'N F T D', '(N T) F D', T=T) + x
        x = self.spatial_attn(self.spatial_norm_layer(x), 'N F T D', '(N F) T D', F=F) + x
        x = self.FFN(x) + x
        return x


class Attention(nn.Module):
    def __init__(self, dim, pos_emb, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.QKV = nn.Linear(dim, dim * 3, bias=qkv_bias)


    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.QKV(x).chunk(3, dim=-1)

        # Divide heads
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=h), (q, k, v))
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        attn = self.attn_drop((q @ k.transpose(-2, -1)).softmax(dim=-1))
        x = attn @ v
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # Merge heads
        x = rearrange(x, '(b h) ... d -> b ... (h d)', h=h)
        x = self.proj_drop(self.proj(x))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layers(x)