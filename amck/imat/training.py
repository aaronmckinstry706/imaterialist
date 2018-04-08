import functools
import typing


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data


def train(data: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
          network: nn.Module,
          optimizer: optim.Optimizer,
          loss_function: typing.Callable[[autograd.Variable, autograd.Variable], autograd.Variable]):
    """
    Trains the given network for a single epoch.

    :param data: An iterable that yields a (data, labels) tuple. 
    :param network: The neural network to be trained.
    :param optimizer: The optimizer with which to train the network.
    :param loss_function: A loss function (outputs, labels) -> scalar.
    :return: The loss history over several batches.
    """
    loss_history = []
    for i, (images, labels) in enumerate(data):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
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


def validate(data: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
          network: nn.Module,
          loss_function: typing.Callable[[autograd.Variable, autograd.Variable], autograd.Variable]):
    """
    Computes the network's average loss over the entire dataset. Note that the loss function must *sum* over each
    batch, rather than *average* over each batch.

    :param data: An iterable that yields a (data, labels) tuple.
    :param network: The neural network to be trained.
    :param loss_function: A loss function (outputs, labels) -> scalar.
    :return: The averaged loss over the whole dataset.
    """
    loss = torch.zeros(1).float()
    total_samples = 0
    for i, (images, labels) in enumerate(data):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        loss += loss_function(network(images), labels).data
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


if __name__ == '__main__':
    test_train()
    test_validate()
