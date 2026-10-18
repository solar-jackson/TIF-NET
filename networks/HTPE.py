import torch
import torch.nn as nn
import torch.nn.functional as F

HAND_KINEMATIC_EDGES = (
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
    (15, 16), (17, 18), (18, 19), (19, 20),
)

def scaled_hand_laplacian(joints=21):
    if joints != 21:
        raise ValueError("The predefined graph supports the 21-joint convention only.")
    adj = torch.zeros(joints, joints, dtype=torch.float32)
    for i, j in HAND_KINEMATIC_EDGES:
        adj[i, j] = adj[j, i] = 1.0
    deg = adj.sum(dim=1)
    invsqrt = deg.clamp_min(1).rsqrt()
    return -(invsqrt[:, None] * adj * invsqrt[None, :])

class ChebyshevGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, order=2):
        super().__init__()
        if order < 2:
            raise ValueError("order >= 2 is required to propagate neighbor information.")
        self.order = order
        self.linear = nn.Linear(order * in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, laplacian):
        cheb = [x]
        if self.order > 1:
            cheb.append(torch.einsum("ij,bjf->bif", laplacian, x))
        for _ in range(2, self.order):
            cheb.append(2 * torch.einsum("ij,bjf->bif", laplacian, cheb[-1]) - cheb[-2])
        y = self.linear(torch.cat(cheb, dim=-1))
        b, j, d = y.shape
        return F.relu(self.norm(y.reshape(b * j, d)).reshape(b, j, d))

class KnowledgeGuidedModule(nn.Module):
    def __init__(self, in_channel=2, out_channel=1024, order=2,
                 widths=(1024, 2048, 2048)):
        super().__init__()
        self.register_buffer("laplacian", scaled_hand_laplacian(), persistent=True)
        features = (in_channel,) + tuple(widths) + (out_channel,)
        self.layers = nn.ModuleList([
            ChebyshevGraphConv(features[i], features[i+1], order=order)
            for i in range(len(features)-1)
        ])

    def forward(self, joints_2d):
        if joints_2d.ndim != 3 or joints_2d.shape[1:] != (21, 2):
            raise ValueError("Expected [batch,21,2] joint coordinates.")
        x = joints_2d
        for layer in self.layers:
            x = layer(x, self.laplacian)
        return x
