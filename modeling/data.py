"""
Dataset over the image cache: per-crop contrast stretch, grey -> 3 channels, ImageNet
normalisation (as MOMO does for pretrained models), and flips / 90-degree rotations for training.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from modeling import cache

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
LOW, HIGH = 2, 98                                # percentile stretch, same as the paper figures


def normalise(img: np.ndarray) -> torch.Tensor:
    """uint16 crop -> (3, H, W) float tensor: 2-98% stretch to [0, 1], copied to 3 channels, ImageNet mean/std."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [LOW, HIGH])
    x = torch.from_numpy(np.clip((img - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32))
    return (x.expand(3, *x.shape) - MEAN) / STD


def augment(x: torch.Tensor) -> torch.Tensor:
    """One of the 8 flip/rotation variants, uniformly: craters have no preferred orientation."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    return torch.rot90(x, int(torch.randint(4, (1,))), dims=(-2, -1))


class CraterDataset(Dataset):
    """Returns (image, class index 0-3, row number in `table`)."""

    def __init__(self, table: pd.DataFrame, size: int = 224, train: bool = False):
        self.table = table.reset_index(drop=True)
        self.size, self.train = size, train
        self.images = None                                       # opened lazily, once per worker
        _, rows = cache.load(size)
        self.rows = rows.loc[self.table.crater_id].to_numpy()
        self.labels = self.table.label.to_numpy() - 1

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        if self.images is None:
            self.images, _ = cache.load(self.size)
        x = normalise(np.asarray(self.images[self.rows[i]]))
        if self.train:
            x = augment(x)
        return x, int(self.labels[i]), i