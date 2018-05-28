import logging
import sys


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as functional


LOGGER = logging.getLogger(__name__)


class NCropAverage(nn.Module):
    """This class wraps a nn.Module instance. The forward function expects a tensor of dimension
    (batch size)-by-(num crops)-by-(remaining dims). It computes the class probabilities for each crop, averages over
    the set of crops for each sample in the batch, and then returns the log of these averaged probabilities. Note
    that the base module must produce linear class scores as output (i.e., pre-softmax)."""

    def __init__(self, base_module: nn.Module):
        super().__init__()
        self.base_module = base_module

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        num_crops = x.size(1)
        remaining_dims = x.size()[2:]
        flattened_dims = (batch_size * num_crops,) + remaining_dims
        x = x.view(*flattened_dims)
        x = self.base_module(x)
        x = functional.softmax(x, dim=1)
        unflattened_output_dims = (batch_size, num_crops, x.size(-1))
        x = x.view(unflattened_output_dims)
        x = torch.mean(x, dim=1)
        return x


class TestNet(nn.Module):
    """
    This class is a class used for testing the NCropAverage class. It is a linear network with weights set to the
    identity matrix and biases set to 0. The number of classes in the output of this network is 2, and the input to
    the network is a
    2-dimensional vector.
    """

    def __init__(self):
        super().__init__()
        self.fixed_linear = nn.Linear(2, 2)
        self.fixed_linear.weight.data = torch.eye(2)
        self.fixed_linear.bias.data.fill_(0)

    def forward(self, x):
        return self.fixed_linear(x)


def test_ncrop_average():
    network_input = torch.log(torch.tensor(
        [[[0.25, 0.75],
          [0.75, 0.25]],
         [[0.25, 0.75],
          [0.75, 0.25]]]
    ))
    expected_output = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    network = NCropAverage(TestNet())
    network.eval()
    actual_output = network(network_input).detach()

    assert (numpy.nextafter(1.0, 2.0, dtype=numpy.float32) - 1.0
            >=
            torch.abs(1.0 - expected_output/actual_output)).all()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    test_ncrop_average()
