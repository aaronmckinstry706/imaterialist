"""
This module contains a set of functions used for calculating the mean and standard deviation of a dataset. The
dataset is assumed to be composed of k-channel images, for some k. Multiprocessing is used to speed up the calculations.
"""


import multiprocessing


import torch
import torch.utils.data as data
import torchvision.datasets as datasets


def mean_per_channel(dataset: datasets.ImageFolder):
    def channel_sum_and_size(image: torch.FloatTensor) -> (torch.FloatTensor, int):
        """Takes an image in the form of a float tensor with dimensions k-by-m-by-n, and returns the per-channel pixel
        sum, as well as the number of pixels in each channel."""
        return torch.sum(torch.sum(image, 2), 1), image.shape[1] * image.shape[2]

    pool = multiprocessing.Pool()
    pool.imap_unordered(channel_sum_and_size, dataset)