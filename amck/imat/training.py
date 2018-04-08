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
    :return:
    """
    for i, (images, labels) in enumerate(data):
        images = autograd.Variable(images)
        labels = autograd.Variable(labels)
        loss = loss_function(network(images), labels)
        network.zero_grad()
        loss.backward()
        optimizer.step()


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
    train(dataloader, network, optimizer, loss_function)


def validate():
    pass


def test():
    pass


if __name__ == '__main__':
    test_train()
