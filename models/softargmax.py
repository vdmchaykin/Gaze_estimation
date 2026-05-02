import torch
import torch.nn.functional as F

def soft_argmax_2d(heatmaps):
    B, C, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, C, -1)
    probs = F.softmax(heatmaps, dim=-1)

    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    ys = torch.linspace(0, 1, H, device=heatmaps.device)
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    x = (probs * xs).sum(dim=-1)
    y = (probs * ys).sum(dim=-1)

    return torch.stack([x, y], dim=-1)
