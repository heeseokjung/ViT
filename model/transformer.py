import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attention = torch.matmul(q, k.transpose(-2, -1))
    attention = attention / math.sqrt(d_k)
    if mask is not None:
        attention = attention.masked_fill(mask==0, -9e15)
    attention = F.softmax(attention, dim=-1)
    values = torch.matmul(attention, v)
    return values

class PatchEmbedding(nn.Module):
    def __init__(self, channel=3, patch_size=16, img_size=224, d_model=768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channel, d_model, kernel_size=patch_size, stride=patch_size),
            # b 768 14 14 -> b 14x14 768
            Rearrange("b d (h) (w) -> b (h w) d")
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, d_model))

    def forward(self, x):
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position_embedding
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, \
               "Embedding dimension must be divided by the # heads"

        self.d_model = d_model
        self.num_heads = num_heads

        self.w_qkv = nn.Linear(self.d_model, 3*self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

        # reset parameters
        nn.init.xavier_normal_(self.w_qkv.weight)
        self.w_qkv.bias.data.fill_(0)
        nn.init.xavier_normal_(self.w_o.weight)
        self.w_o.bias.data.fill_(0)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.w_qkv(x)

        # separate Q, K, V
        qkv = rearrange(qkv, "b n (h d_k qkv) -> b h n (d_k qkv)", h=self.num_heads, qkv=3)
        q, k, v = qkv.chunk(3, dim=-1)

        # calculate values
        values = scaled_dot_product(q, k, v, mask=mask)
        values = rearrange(values, "b h n d_k -> b n (h d_k)")
        values = self.w_o(values)

        return values

class EncoderBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=8, hidden_dim=768*2, dropout=0.):
        super().__init__()

        # Multi-head Self Attention Layer
        self.attention = MultiheadAttention(d_model, num_heads)

        # MLP Layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head Self Attention
        nx = self.norm1(x)
        nx = self.attention(nx, mask)
        nx = self.dropout(nx)
        y =  x + nx

        # MLP
        ny = self.norm2(y)
        ny = self.mlp(ny)
        ny = self.dropout(ny)
        z = y + ny

        return z

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=12, **block_args):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, d_model=768, n_class=10):
        super().__init__()
        self.d_model = d_model
        self.n_class = n_class
        self.linear = nn.Sequential(
            Reduce("b n d -> b d", reduction="mean"),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.n_class)
        )

    def forward(self, x):
        return self.linear(x)

class VisionTransformer(nn.Sequential):
    def __init__(self,
                 channel=3,
                 patch_size=16,
                 img_size=224,
                 d_model=768,
                 num_layers=12,
                 n_class=10,
                 **kwargs):
        super().__init__(
            PatchEmbedding(channel, patch_size, img_size, d_model),
            TransformerEncoder(num_layers, **kwargs),
            ClassificationHead(d_model, n_class)
        )

'''
class VisionTransformer(nn.Module):
    def __init__(self,
                 channel=3,
                 patch_size=16,
                 img_size=224,
                 d_model=768,
                 num_layers=12,
                 n_class=10,
                 **kwargs):
        super().__init__()
        self.channel = channel
        self.patch_size = patch_size
        self.img_size = img_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_class = n_class
        self.kwargs = kwargs

        self.model = nn.Sequential(
            PatchEmbedding(self.channel, self.patch_size, self.img_size, self.d_model),
            TransformerEncoder(self.num_layers, **self.kwargs),
            ClassificationHead(self.d_model, self.n_class)
        )

    def forward(self, x):
        return self.model(x)
'''