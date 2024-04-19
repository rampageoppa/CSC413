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
from utils import init_weights

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
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

# Modified Attention module to focus on center pixel
class PatchCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, patch_indices, data, scale):
        b, n, _ = x.shape[0], x.shape[1], x.shape[2]
        h = self.heads
        v_patch = scale[0]
        h_patch = scale[1]

        new_patch_indices = (patch_indices[:, 0] * v_patch + patch_indices[:, 1]).type(torch.long)
        new_patch_indices = new_patch_indices.unsqueeze(1)

        selected_x = x.gather(1, new_patch_indices.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        q = self.to_q(selected_x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale                                     # num_agents, 16, 1,100

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                                                  # num_agents, 16, 1, 64

        out = rearrange(out, 'b h n d -> b n (h d)')                                                 # num_agents, 1, 1024
        return self.to_out(out)                                                                      # num_agents, 1, 128 -> num_agents, 1, 128


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PatchCrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        self.apply(init_weights)

    def forward(self, x, patch_indices, data, scale):
        # x.shape: num_agents, num_patches, 128
        for attn, ff in self.layers:
            attn_output = attn(x, patch_indices, data, scale)                           # (num_agents, 1, 128)

            new_patch_indices = (patch_indices[:, 0] * scale[0] + patch_indices[:, 1]).unsqueeze(1).type(torch.long)
            expand_attn = attn_output.expand(-1, x.shape[1], -1)
            new_tensor = torch.zeros_like(expand_attn)
            batch_indices = torch.arange(0, x.size(0)).unsqueeze(1)

            for i in range(x.size(0)):
                new_tensor[i, new_patch_indices[i], :] = attn_output[i]
            
            x = new_tensor + x
            x = ff(x) + x

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
        
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.apply(init_weights)
        self.scale = (image_height/patch_height), (image_width/patch_width)

    
    def pair(self, t):
        return t if isinstance(t, tuple) else (t, t)

    def forward(self, data):
        bev_embed = data['bev_embed'].reshape(-1, self.image_height, self.image_width, self.channels)
        av_index = data['av_index']
        av_index = torch.cat((av_index, torch.tensor([data['positions'].shape[0]], device=av_index.device, dtype=av_index.dtype)))

        replication_counts = (av_index[1:] - av_index[:-1]).cpu()
        replicated_bev_embed = torch.cat([bev_embed[i].repeat(count, 1, 1, 1) for i, count in enumerate(replication_counts)])

        # for i in range(len(bev_embed)):
        #     visualize_bev(bev_embed[i].numpy())
        # plot_trajectory(data['positions'][:28].cpu().numpy())

        img = replicated_bev_embed.permute(0, 3, 1, 2)
        patch_indices = get_patch_idx(data, self.patch_height, self.patch_width, self.image_height, self.image_width).long()

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x, patch_indices, data, self.scale)                          # num_agents, num_patches, 128
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]                           # num_agents, 128

        x = self.to_latent(x)
        return self.mlp_head(x)

def get_patch_idx(data, patch_height, patch_width, image_height, image_width):
    coordinates = data['positions']
    rotate_90_degrees_ccw = torch.tensor([[0, -1], [1,  0]], dtype=torch.float).to(coordinates.device)

    reshaped_coordinates = coordinates.view(-1, 2).T 
    rotated_reshaped_coordinates = torch.matmul(rotate_90_degrees_ccw, reshaped_coordinates)
    rotated_coordinates = rotated_reshaped_coordinates.T.view(coordinates.shape)

    x_limit = 30.0 / 2.0 
    y_limit = 60.0 / 2.0  
 
    perception_height, perception_width = 60.0, 30.0

    scale_height = image_height / perception_height
    scale_width = image_width / perception_width

    pixel_coordinates = torch.full(rotated_coordinates.shape, -1.0, device=rotated_coordinates.device)

    scaled_coordinates = torch.clone(rotated_coordinates)
    scaled_coordinates[..., 0] *= scale_width
    scaled_coordinates[..., 1] *= scale_height

    translated_coordinates = torch.clone(scaled_coordinates)
    translated_coordinates[..., 0] += image_width / 2.0
    translated_coordinates[..., 1] = image_height / 2.0 - translated_coordinates[..., 1]

    pixel_coordinates = translated_coordinates
    patch_indices = pixel_coordinates // torch.tensor([patch_width, patch_height]).to(coordinates.device)
    patch_indices = patch_indices[..., [1, 0]]

    h_patch, v_patch = image_height/patch_height, image_width/patch_width 
    mask_1 = (patch_indices == h_patch) 
    mask_2 = (patch_indices == v_patch)

    patch_indices[mask_1] = h_patch - 1
    patch_indices[mask_2] = v_patch - 1

    return patch_indices[:, 19, :]

def plot_trajectory(coordinates):
    # plt.figure(figsize=(10, 6))
    for i in range(len(coordinates)):
        plt.plot(coordinates[i][:20,0], coordinates[i][:20,1], color='red')
        plt.plot(coordinates[i][20:,0], coordinates[i][20:,1], color='blue')
        plt.scatter(coordinates[i][-1, 0], coordinates[i][-1, 1], marker="*")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectory Plot with Direction')
    plt.grid(True)
    plt.show()


def visualize_bev(bev_embed):
    feature_embedding = bev_embed
    # aggregated_embedding = np.mean(feature_embedding, axis=2)

    reshaped_embedding = feature_embedding.reshape(-1, 256)

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
