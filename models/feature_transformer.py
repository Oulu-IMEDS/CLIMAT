import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class FeatureTransformer(nn.Module):
    def __init__(self, num_patches, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_cls_num=1, with_cls=True,
                 n_outputs=1, dropout=0., emb_dropout=0.):
        super().__init__()
        self.patch_dim = patch_dim
        self.n_outputs = n_outputs
        self.with_cls = with_cls
        if self.with_cls:
            self.cls_token = nn.Parameter(torch.randn(1, num_cls_num, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_num, dim))
        self.type_embedding = nn.Parameter(torch.randn(1, num_patches + num_cls_num, dim))
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        for i in range(self.n_outputs):
            setattr(self, f"mlp_head{i}", nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes)
            ))

    def forward(self, features, mask=None):
        x = self.patch_to_embedding(features)

        if self.with_cls:
            cls_tokens = self.cls_token.expand(features.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding + self.type_embedding
        x = self.dropout(x)

        states, attentions = self.transformer(x, mask)

        x = self.to_cls_token(states[:, 0:self.n_outputs])

        outputs = []
        for i in range(self.n_outputs):
            out = getattr(self, f"mlp_head{i}")(x[:, i])
            outputs.append(out)

        if len(outputs) > 0:
            outputs = torch.stack(outputs, dim=1)

        return outputs, states, attentions


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.depth = depth

        for d in range(depth):
            setattr(self, f"prenorm_0_{d}", nn.LayerNorm(dim))
            setattr(self, f"attn_{d}", Attention(dim, heads=heads, dropout=dropout))

            setattr(self, f"prenorm_1_{d}", nn.LayerNorm(dim))
            setattr(self, f"ff_{d}", FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x, mask=None):
        attentions = []
        for d in range(self.depth):
            o = getattr(self, f"prenorm_0_{d}")(x)
            o, attn = getattr(self, f"attn_{d}")(o, mask)
            attentions.append(attn)
            x = o + x

            ff = getattr(self, f"prenorm_1_{d}")(x)
            ff = getattr(self, f"ff_{d}")(ff)
            x = ff + x

        return x, attentions