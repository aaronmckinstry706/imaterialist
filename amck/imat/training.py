import functools
import logging
import math
import sys
import typing


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data


LOGGING_LEVEL = logging.DEBUG
LOGGER = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(LOGGING_LEVEL)
LOGGER.addHandler(stream_handler)
LOGGER.setLevel(LOGGING_LEVEL)


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
    accuracy_history = []
    for i, (images, labels) in enumerate(dataset):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()
        network_output = network(images)
        _, predicted_indices = torch.max(network_output.data, 1)
        accuracy = torch.sum(torch.eq(predicted_indices, labels.data))/float(len(images.data))
        accuracy_history.append(accuracy)
        loss = loss_function(network_output, labels)
        loss_history.append(loss.data[0])
        network.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history, accuracy_history


def test_train(cuda=False):
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    if cuda:
        network.cuda()
    optimizer = optim.SGD(network.parameters(), lr=0.1)
    loss_function = functional.cross_entropy
    dataset = data.TensorDataset(torch.rand(200, 3), (torch.rand(200)*3).long())
    dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)
    loss_history, accuracy_history = train(dataloader, network, optimizer, loss_function, cuda=cuda)

    assert isinstance(loss_history, list)
    assert len(loss_history) == 67
    for loss in loss_history:
        assert isinstance(loss, float) and loss >= 0.0

    assert isinstance(accuracy_history, list)
    assert len(accuracy_history) == 67
    for accuracy in accuracy_history:
        assert isinstance(accuracy, float) and 0.0 <= accuracy <= 1.0


def evaluate_loss(dataset: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
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


def test_evaluate_loss(cuda=False):
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    if cuda:
        network.cuda()
    loss_function = functools.partial(functional.cross_entropy, size_average=False)
    dataset = data.TensorDataset(torch.rand(200, 3), (torch.rand(200)*3).long())
    dataloader = data.DataLoader(dataset, batch_size=3, num_workers=1)
    assert isinstance(evaluate_loss(dataloader, network, loss_function, cuda=cuda), float)


def evaluate_accuracy(dataset: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
                      network: nn.Module,
                      cuda=False):
    network.eval()
    total_num_correct = 0
    total_samples = 0
    for i, (images, labels) in enumerate(dataset):

        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()
        _, predicted_indices = torch.max(network(images).data, 1)
        total_num_correct += torch.sum(torch.eq(predicted_indices, labels.data))
        total_samples += len(images)
    return total_num_correct/total_samples


def test_evaluate_accuracy(cuda=False):
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)
            self.fc.weight.data.fill_(0)
            self.fc.bias.data = torch.FloatTensor([1, 0, 0])

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    if cuda:
        network.cuda()
    dataset = data.TensorDataset(torch.rand(12, 3), torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]))
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    accuracy = evaluate_accuracy(dataloader, network, cuda=cuda)
    assert isinstance(accuracy, float) and accuracy == 0.5


def evaluate_loss_and_accuracy(dataset: typing.Iterable[typing.Tuple[torch.FloatTensor, torch.LongTensor]],
                               network: nn.Module,
                               loss_function: typing.Callable[[autograd.Variable, autograd.Variable], autograd.Variable],
                               cuda=False):
    network.eval()
    loss = torch.zeros(1).float()
    total_num_correct = 0
    total_samples = 0
    for i, (images, labels) in enumerate(dataset):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        if cuda:
            images = images.cuda()
            labels = labels.cuda()
        network_output = network(images)
        loss += loss_function(network_output, labels).data[0]
        _, predicted_indices = torch.max(network_output.data, 1)
        total_num_correct += torch.sum(torch.eq(predicted_indices, labels.data))
        total_samples += len(images)
    return (loss / total_samples)[0], total_num_correct/total_samples


def test_evaluate_loss_and_accuracy(cuda=False):
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)
            self.fc.weight.data.fill_(0)
            self.fc.bias.data = torch.FloatTensor([math.log(0.5), math.log(0.25), math.log(0.25)])

        def forward(self, x):
            return self.fc(x)

    network = SimpleLinear()
    if cuda:
        network.cuda()
    dataset = data.TensorDataset(torch.rand(12, 3), torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]))
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    loss_function = functools.partial(functional.cross_entropy, size_average=False)
    loss, accuracy = evaluate_loss_and_accuracy(dataloader, network, loss_function, cuda=cuda)

    loss_values_by_label = -1.0 * torch.log(torch.FloatTensor([0.5, 0.25, 0.25]))
    expected_sample_losses = torch.FloatTensor([loss_values_by_label[0]]*6 + [loss_values_by_label[1]]*6)
    expected_loss = torch.sum(expected_sample_losses)/len(expected_sample_losses)

    assert isinstance(loss, float) and loss == expected_loss
    assert isinstance(accuracy, float) and accuracy == 0.5


if __name__ == '__main__':
    test_train()
    test_train(cuda=True)
    test_evaluate_loss()
    test_evaluate_loss(cuda=True)
    test_evaluate_accuracy()
    test_evaluate_accuracy(cuda=True)
    test_evaluate_loss_and_accuracy()
    test_evaluate_loss_and_accuracy(cuda=True)
