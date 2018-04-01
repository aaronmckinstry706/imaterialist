"""
This module contains a set of functions used for calculating the mean and standard deviation of a dataset. The
dataset is assumed to be composed of k-channel images, for some k. Multiprocessing is used to speed up the calculations.
"""


import multiprocessing
import multiprocessing.pool as pool
import typing

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def channel_sum_and_size(image: torch.FloatTensor) -> (torch.FloatTensor, int):
    """Takes an image in the form of a float tensor with dimensions k-by-m-by-n, and returns the per-channel pixel
    sum, as well as the number of pixels in each channel."""
    return torch.sum(torch.sum(image, 2), 1), image.shape[1] * image.shape[2]


def mean_per_channel(dataset: datasets.ImageFolder, num_workers: int = 8):
    """Takes a dataset of `(image, label)` pairs, where `image` is a float tensor with dimensions k-by-m-by-n. It
    returns the per-channel mean."""

    thread_pool = pool.ThreadPool(num_workers)
    asyncresults: typing.Union[multiprocessing.pool.AsyncResult, None] = [None for _ in range(len(dataset))]
    channel_sums = [None for _ in range(len(dataset))]
    channel_sizes = [None for _ in range(len(dataset))]
    for i in range(len(dataset)):
        asyncresults[i] = thread_pool.apply_async(channel_sum_and_size, (dataset[i][0],))
    for i in range(len(dataset)):
        channel_sums[i], channel_sizes[i] = asyncresults[i].get()

    return sum(channel_sums) / sum(channel_sizes)


def test_mean_per_channel():
    print(
        mean_per_channel(
            datasets.ImageFolder('data/validation', transform=transforms.ToTensor()),
            num_workers=4))


if __name__ == '__main__':
    test_mean_per_channel()
