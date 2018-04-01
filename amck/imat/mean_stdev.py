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


def channel_sum_and_size(dataset, i) -> (torch.FloatTensor, int):
    """Takes an image in the form of a float tensor with dimensions k-by-m-by-n, and returns the per-channel pixel
    sum, as well as the number of pixels in each channel."""
    image = dataset[i][0]
    return torch.sum(torch.sum(image, 2), 1), image.shape[1] * image.shape[2]


def mean_per_channel(dataset, num_workers: int = 8):
    """Takes a dataset of tuples `(image, ...)`, where `image` is a float tensor with dimensions k-by-m-by-n and the
    elipses can be anything. It returns the per-channel mean."""

    thread_pool = pool.ThreadPool(num_workers)
    asyncresults: typing.Union[multiprocessing.pool.AsyncResult, None] = [None for _ in range(len(dataset))]
    channel_sums = [None for _ in range(len(dataset))]
    channel_sizes = [None for _ in range(len(dataset))]
    for i in range(len(dataset)):
        asyncresults[i] = thread_pool.apply_async(channel_sum_and_size, (dataset, i))
    for i in range(len(dataset)):
        channel_sums[i], channel_sizes[i] = asyncresults[i].get()

    return sum(channel_sums) / sum(channel_sizes)


def test_mean_per_channel():
    assert torch.equal(
        mean_per_channel([(torch.zeros(3, 224, 224),), (torch.zeros(3, 224, 224),)], num_workers=1),
        torch.zeros(3))
    assert torch.equal(
        mean_per_channel([(torch.zeros(3, 224, 224),), (torch.ones(3, 224, 224),)], num_workers=1),
        torch.zeros(3) + 0.5)
    assert mean_per_channel([(torch.zeros(3, 224, 224),), (torch.ones(3, 224, 224),)], num_workers=1).shape == (3,)


def channel_sum_of_squares_and_size(dataset, i, means) -> (torch.FloatTensor, int):
    """Takes an image in the form of a float tensor with dimensions k-by-m-by-n, and returns the per-channel pixel
    sum of squared pixels, as well as the number of pixels in each channel."""
    diffs = dataset[i][0] - means
    return torch.sum(torch.sum(diffs*diffs, 2), 1), diffs.shape[1] * diffs.shape[2]


def means_and_stdev_per_channel(dataset, num_workers: int = 8):
    """Takes a dataset of tuples `(image, ...)`, where `image` is a float tensor with dimensions k-by-m-by-n and the
    elipses can be anything. It returns the per-channel standard deviation."""

    means = mean_per_channel(dataset, num_workers).unsqueeze(1).unsqueeze(2)

    thread_pool = pool.ThreadPool(num_workers)
    asyncresults: typing.Union[multiprocessing.pool.AsyncResult, None] = [None for _ in range(len(dataset))]
    channel_sums_of_squares = [None for _ in range(len(dataset))]
    channel_sizes = [None for _ in range(len(dataset))]
    for i in range(len(dataset)):
        asyncresults[i] = thread_pool.apply_async(channel_sum_of_squares_and_size, (dataset, i, means))
    for i in range(len(dataset)):
        channel_sums_of_squares[i], channel_sizes[i] = asyncresults[i].get()

    return means.squeeze(), torch.sqrt(sum(channel_sums_of_squares) / (sum(channel_sizes) - 1))


def test_means_and_stdev_per_channel():
    mean, stdev = means_and_stdev_per_channel([(torch.zeros(3, 2, 2),), (torch.ones(3, 2, 2),)], num_workers=1)
    assert mean.shape == (3,)
    assert stdev.shape == (3,)
    assert torch.equal(mean, torch.zeros(3) + 0.5)
    assert torch.equal(stdev, torch.sqrt(torch.zeros(3) + (2.0/7.0)))


if __name__ == '__main__':
    test_mean_per_channel()
    test_means_and_stdev_per_channel()

    print(means_and_stdev_per_channel(datasets.ImageFolder('data/training', transform=transforms.ToTensor())))

