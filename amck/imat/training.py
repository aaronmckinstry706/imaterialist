import functools
import typing


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data


def train(dataset: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
          network: nn.Module,
          optimizer: optim.Optimizer,
          loss_function: typing.Callable[[autograd.Variable, autograd.Variable], autograd.Variable],
          cuda=False):
    """
    Trains the given network for a single epoch.

    :param dataset: An iterable that yields a (data, labels) tuple. 
    :param network: The neural network to be trained.
    :param optimizer: The optimizer with which to train the network.
    :param loss_function: A loss function (outputs, labels) -> scalar.
    :param cuda: If true, then it is assumed that network is on GPU.
    :return: The loss history over several batches.
    """
    network.train()
    loss_history = []
    for i, (images, labels) in enumerate(dataset):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()
        loss = loss_function(network(images), labels)
        loss_history.append(loss.data[0])
        network.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history


def test_train():
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    optimizer = optim.SGD(network.parameters(), lr=0.1)
    loss_function = functional.cross_entropy
    dataset = data.TensorDataset(torch.rand(200, 3), (torch.rand(200)*3).long())
    dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)
    loss_history = train(dataloader, network, optimizer, loss_function)
    assert isinstance(loss_history, list)
    assert len(loss_history) == 67
    for loss in loss_history:
        assert isinstance(loss, float)


def test_train_cuda():
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    network.cuda()
    optimizer = optim.SGD(network.parameters(), lr=0.1)
    loss_function = functional.cross_entropy
    dataset = data.TensorDataset(torch.rand(200, 3), (torch.rand(200)*3).long())
    dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)
    loss_history = train(dataloader, network, optimizer, loss_function, cuda=True)
    assert isinstance(loss_history, list)
    assert len(loss_history) == 67
    for loss in loss_history:
        assert isinstance(loss, float)


def validate(dataset: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
             network: nn.Module,
             loss_function: typing.Callable[[autograd.Variable, autograd.Variable], autograd.Variable],
             cuda=False):
    """
    Computes the network's average loss over the entire dataset. Note that the loss function must *sum* over each
    batch, rather than *average* over each batch.

    :param dataset: An iterable that yields a (data, labels) tuple.
    :param network: The neural network to be trained.
    :param loss_function: A loss function (outputs, labels) -> scalar.
    :param cuda: If true, then it is assumed that network is on GPU.
    :return: The averaged loss over the whole dataset.
    """
    network.eval()
    loss = torch.zeros(1).float()
    total_samples = 0
    for i, (images, labels) in enumerate(dataset):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()
        loss += loss_function(network(images), labels).data[0]
        total_samples += len(images)
    return (loss / total_samples)[0]


def test_validate():
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    loss_function = functools.partial(functional.cross_entropy, size_average=False)
    dataset = data.TensorDataset(torch.rand(200, 3), (torch.rand(200)*3).long())
    dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)
    assert isinstance(validate(dataloader, network, loss_function), float)


def test_validate_cuda():
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    network.cuda()
    loss_function = functools.partial(functional.cross_entropy, size_average=False)
    dataset = data.TensorDataset(torch.rand(200, 3), (torch.rand(200)*3).long())
    dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)
    assert isinstance(validate(dataloader, network, loss_function, cuda=True), float)


if __name__ == '__main__':
    test_train()
    test_train_cuda()
    test_validate()
    test_validate_cuda()
