"""
This module contains a set of functions used for calculating the mean and standard deviation of a dataset. The
dataset is assumed to be composed of k-channel images, for some k. Multiprocessing is used to speed up the calculations.
"""


import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def channel_sum_and_size(batch: torch.FloatTensor) -> (torch.FloatTensor, int):
    return torch.sum(torch.sum(torch.sum(batch, 3), 2), 0), batch.shape[0] * batch.shape[2] * batch.shape[3]


def mean_per_channel(dataloader: data.DataLoader):
    channel_sums = torch.zeros(3).float()
    channel_sizes = torch.zeros(3).float()
    for (images, labels) in dataloader:
        sums, sizes = channel_sum_and_size(images)
        channel_sums += sums
        channel_sizes += sizes

    return channel_sums / channel_sizes


def test_mean_per_channel():
    half0_half1 = data.DataLoader(
        data.TensorDataset(
            torch.cat([torch.zeros(2, 3, 2, 2), torch.ones(2, 3, 2, 2)]), torch.Tensor([0 for _ in range(4)])),
        num_workers=1)
    all1 = data.DataLoader(
        data.TensorDataset(torch.ones(2, 3, 2, 2), torch.Tensor([0 for _ in range(2)])),
        num_workers=1)

    assert torch.equal(
        mean_per_channel(all1),
        torch.ones(3))
    assert torch.equal(
        mean_per_channel(half0_half1),
        torch.zeros(3) + 0.5)
    assert mean_per_channel(all1).shape == (3,)


def channel_sum_of_squares_and_size(batch: torch.FloatTensor, means: torch.FloatTensor) -> (torch.FloatTensor, int):
    diffs = batch - means.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return torch.sum(torch.sum(torch.sum(diffs*diffs, 3), 2), 0), diffs.shape[0] * diffs.shape[2] * diffs.shape[3]


def means_and_stdev_per_channel(dataloader: data.DataLoader):
    channel_means = mean_per_channel(dataloader)

    channel_sums_of_squares = torch.zeros(3)
    channel_sizes = torch.zeros(3)
    for (images, labels) in dataloader:
        sums, sizes = channel_sum_of_squares_and_size(images, channel_means)
        channel_sums_of_squares += sums
        channel_sizes += sizes

    return channel_means, torch.sqrt(channel_sums_of_squares / (channel_sizes - 1))


def test_means_and_stdev_per_channel():
    half0_half1 = data.DataLoader(
        data.TensorDataset(
            torch.cat([torch.zeros(2, 3, 2, 2), torch.ones(2, 3, 2, 2)]), torch.Tensor([0 for _ in range(4)])),
        num_workers=1)
    mean, stdev = means_and_stdev_per_channel(half0_half1)
    assert mean.shape == (3,)
    assert stdev.shape == (3,)
    assert torch.equal(mean, torch.zeros(3) + 0.5)
    assert torch.equal(stdev, torch.sqrt(torch.zeros(3) + (4.0/15.0)))


if __name__ == '__main__':
    test_mean_per_channel()
    test_means_and_stdev_per_channel()

    dataloader = data.DataLoader(
        datasets.ImageFolder(
            'data/training',
            transform=transforms.ToTensor()),
        num_workers=8,
        batch_size=1)
    print(means_and_stdev_per_channel(dataloader))
