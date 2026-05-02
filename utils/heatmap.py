import torch

def generate_heatmap(size, center, sigma=2):
    H, W = size
    y = torch.arange(0, H).float()
    x = torch.arange(0, W).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    cx, cy = center
    heatmap = torch.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
    )
    return heatmap
