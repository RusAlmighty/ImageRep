import os
import random
from glob import glob

import cv2
import torch
from torch.utils.data import Dataset


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def collate_1d(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    x, y, meta = zip(*data)
    return torch.concat(x, dim=0), torch.concat(y, dim=0), meta


class ImageFitting(Dataset):
    def __init__(self, path, limit=None, cach=True):
        super().__init__()
        self.sample_list = list(glob(os.path.join(path, "*.png")))
        if cach:
            self._cached_samples = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in self.sample_list]
        else:
            self._cached_samples = None
        self.path = path
        self.limit = limit

        categories = [os.path.basename(s.split("-")[0]) for s in self.sample_list]
        cats = sorted(set(categories))
        self.map = {c: n / len(cats) for n, c in enumerate(cats)}

        self.mapping = list(map(lambda x: self.map[x], categories))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # img = transform(cv2.imread(self.sample_list[idx]))
        if self._cached_samples:
            img = (torch.tensor(self._cached_samples[idx]) / 255. - 0.5) * 2
        else:
            img = (torch.tensor(cv2.imread(self.sample_list[idx]), cv2.IMREAD_UNCHANGED) / 255. - 0.5) * 2
        pixels = img.view(-1, 4)
        # assume square samples only
        coords = get_mgrid(img.shape[1], 2)
        coords = torch.concat((coords, self.mapping[idx] * torch.ones([coords.shape[0], 1])), dim=-1)
        if self.limit:
            len_coord = len(coords)
            l = list(range(len_coord))
            random.shuffle(l)
            indxs = l[:min(self.limit, len_coord)]
            return coords[indxs], pixels[indxs], 0
        else:
            return coords, pixels, 0
