import torch
import torch.nn as nn

class SpatialCrossAttentionBlock(nn.Module):
    def __init__(self, channels=256, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.visual_to_topology = nn.MultiheadAttention(
            channels, num_heads, batch_first=True)
        self.topology_to_visual = nn.MultiheadAttention(
            channels, num_heads, batch_first=True)
        self.p_norm = nn.LayerNorm(channels)
        self.h_norm = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.p_mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                   nn.Linear(hidden, channels))
        self.h_mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                   nn.Linear(hidden, channels))

    def forward(self, p, h):
        p_update, _ = self.visual_to_topology(p, h, h, need_weights=False)
        h_update, _ = self.topology_to_visual(h, p, p, need_weights=False)
        p = self.p_norm(p + p_update)
        h = self.h_norm(h + h_update)
        p = p + self.p_mlp(p)
        h = h + self.h_mlp(h)
        return p, h

class CrossAttention(nn.Module):
    def __init__(self, channel_a=256, channel_b=256, out_channel=256,
                 dim=None, depth=2, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.proj_h = nn.Conv2d(channel_a, out_channel, 1)
        self.proj_p = nn.Conv2d(channel_b, out_channel, 1)
        self.layers = nn.ModuleList([
            SpatialCrossAttentionBlock(out_channel, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, FI, FP):
        if FI.shape[-2:] != FP.shape[-2:]:
            raise ValueError("Visual and topology maps must share the same spatial grid.")
        hmap, pmap = self.proj_h(FI), self.proj_p(FP)
        b, c, height, width = hmap.shape
        h = hmap.flatten(2).transpose(1, 2)
        p = pmap.flatten(2).transpose(1, 2)
        for block in self.layers:
            p, h = block(p, h)
        h = h.transpose(1, 2).reshape(b, c, height, width)
        p = p.transpose(1, 2).reshape(b, c, height, width)
        return self.fusion_conv(torch.cat([h, p], dim=1))
