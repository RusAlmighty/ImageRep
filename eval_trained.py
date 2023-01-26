import argparse
import os
from glob import glob
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ModelConfig, add_model
from data_pipeline import get_mgrid, read_im
from model import Siren


def size_interpolation_example():
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    res = 256
    coords = get_mgrid(res, dim=2)
    encoded_index = 0.85
    coords = torch.concat((coords, encoded_index * torch.ones([coords.shape[0], 1])), dim=-1)
    model_output_256 = model(coords)
    img_256 = cv2.cvtColor(np.clip(model_output_256.cpu().view(res, res, 3).detach().numpy() / 2 + 0.5, 0, 1),
                           cv2.COLOR_RGB2BGR)
    axes[0].imshow(img_256)
    axes[0].title.set_text("Interpolated 256x256")

    res = 96
    coords = get_mgrid(res, dim=2)
    coords = torch.concat((coords, encoded_index * torch.ones([coords.shape[0], 1])), dim=-1)
    model_output_96 = model(coords)
    img_96 = cv2.cvtColor(np.clip(model_output_96.cpu().view(res, res, 3).detach().numpy() / 2 + 0.5, 0, 1),
                          cv2.COLOR_RGB2BGR)
    axes[1].imshow(img_96)
    axes[1].title.set_text("Trained sample 96x96")

    plt.savefig("size_interpolation_example.jpg")


def sort_by_similarity(path: str) -> List[int]:
    sample_list = list(glob(os.path.join(path, "*.png")))
    sample_list = sorted(sample_list)
    _cached_samples = [read_im(f, set_transparent_to_white=True) for f in sample_list]
    # sort by MSE similarity
    indxs = np.argsort(np.mean(np.square(np.reshape(np.diff(_cached_samples, axis=0), (-1, 96 ** 2 * 3))), axis=1))
    return indxs


def image_interpolation_example(indxs: List[int]):
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    # interpolate between most similar images
    labels = ["source 1", "interpolated", "source 2"]
    for j in range(3):
        axes[0, j].title.set_text(labels[j])
    for p, idx in enumerate(indxs[:3]):
        for j in range(3):
            res = 96
            encoded_index = (idx + j / 2) / 100
            coords = get_mgrid(res, dim=2)
            coords = torch.concat((coords, encoded_index * torch.ones([coords.shape[0], 1])), dim=-1)
            model_output_96 = model(coords)
            img_96 = cv2.cvtColor(np.clip(model_output_96.cpu().view(res, res, 3).detach().numpy() / 2 + 0.5, 0, 1),
                                  cv2.COLOR_RGB2BGR)
            axes[p, j].imshow(img_96)
    plt.savefig("image_interpolation_example.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")  # positional argument
    add_model(parser, ModelConfig)
    args = parser.parse_args()
    ckpt = args.checkpoint  # r"output/run17/ckpt-epoch=892.ckpt"
    d = vars(args)
    del d["checkpoint"]
    config = ModelConfig(**d)

    model = Siren.load_from_checkpoint(ckpt, config=config)

    # Size interpolation example
    size_interpolation_example()

    # Image interpolation example
    dataset = r"Data\96"
    indxs = sort_by_similarity(dataset)

    image_interpolation_example(indxs)
