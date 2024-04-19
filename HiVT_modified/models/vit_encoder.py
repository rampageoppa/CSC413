import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
import faulthandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
faulthandler.enable()
# from utils import init_weights

# helpers

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # self.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        # breakpoint()
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale       # b, 16, 101, 101

        attn = self.attend(dots)                                       # b, 16, 101, 101
        attn = self.dropout(attn)                  # b, 16, 101, 101

        out = torch.matmul(attn, v)                # b, 16, 101, 64
        out = rearrange(out, 'b h n d -> b n (h d)')     # b, 101, 1024
        # breakpoint()
        return self.to_out(out)                    # b, 101, 128

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        # self.apply(init_weights)

    def forward(self, x):
        # breakpoint() # x.shape  32, 101, 128
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        # breakpoint()
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        image_height, image_width = self.pair(image_size)
        patch_height, patch_width = self.pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        # self.apply(init_weights)

    
    def pair(self, t):
        return t if isinstance(t, tuple) else (t, t)

    def forward(self, data):
        bev_embed = torch.tensor(data['bev_embed']).reshape(-1, 200, 100, 256)
        # for i in range(len(bev_embed)):
        #     breakpoint()
        #     visualize_bev(bev_embed[i].numpy())
        img = bev_embed.cuda().permute(0, 3, 1, 2)

        # img = data
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)                                  # 32, 101, 128

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]   # 32, 128

        x = self.to_latent(x)                               # 32, 128
        return self.mlp_head(x)                            # 32, 128

def visualize_bev(bev_embed):
    feature_embedding = bev_embed
    # aggregated_embedding = np.mean(feature_embedding, axis=2)

    reshaped_embedding = feature_embedding.reshape(-1, 256)
    # breakpoint()
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(reshaped_embedding)
    pca_image = principal_component.reshape(200, 100)
    aggregated_embedding = pca_image

    normalized_embedding = (aggregated_embedding - np.min(aggregated_embedding)) / (np.max(aggregated_embedding) - np.min(aggregated_embedding))
    normalized_embedding *= 255

    plt.imshow(normalized_embedding, cmap='gray')
    plt.colorbar()
    plt.title("Visualization of Feature Embedding")
    plt.show()


# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# img = torch.randn(1, 3, 256, 256)

# preds = v(img) # (1, 1000)
# breakpoint()

# v = ViT(
#     image_size = (200, 100),
#     patch_size = (20, 10),
#     num_classes = 10,
#     pool = 'mean',
#     channels = 256,
#     dim = 128,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 512,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# img = torch.randn(32, 256, 200, 100)

# preds = v(img) # (1, 1000)
# model = v.cuda()
# pred = model(img.cuda())
# breakpoint()

# v = ViT(
#     num_classes = 1000,
#     image_size = (256, 128),  # image size is a tuple of (height, width)
#     patch_size = (32, 16),    # patch size is a tuple of (height, width)
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# img = torch.randn(1, 3, 256, 128)