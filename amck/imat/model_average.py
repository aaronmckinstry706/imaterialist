import logging
import typing


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as functional


LOGGER = logging.getLogger(__name__)


class ModelAverage(nn.Module):
    """
    This class works by averaging the outputs of existing models. The models are expected to have linear outputs (i.e.,
    *before* converting to probabilities via softmax) of two dimensions, where the first dimension is the batch index
    and the second dimension is the class index. The output of this model is the log of the average of the softmax of
    each model's output.
    """

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.base_modules = nn.ModuleList(modules=modules)

    def forward(self, x):
        linear_outputs: typing.List[torch.Tensor] = []
        for module in self.base_modules:
            linear_outputs.append(functional.softmax(module(x), dim=1))
        # Make a len(self.modules)-by-(batch size)-by-(num classes) Tensor from the set of outputs.
        concatenated_outputs = torch.cat([torch.unsqueeze(linear_output, dim=0) for linear_output in linear_outputs])
        return torch.log(torch.mean(concatenated_outputs, 0))


class TestNet(nn.Module):
    """
    This class is a class used for testing the ModelAverage class. It is a linear network with fixed outputs that are
    set to `torch.log(probabilities)`, via setting the weights to 0 and the biases to the log probabilities. The number
    of classes in the output of this network is 2, and the input to the network is a 3-dimensional vector.
    """

    def __init__(self, *probabilities: float):
        super().__init__()
        self.fixed_linear = nn.Linear(3, 2)
        self.fixed_linear.weight.data.fill_(0)
        self.fixed_linear.bias.data = torch.log(torch.tensor(probabilities))

    def forward(self, x):
        return self.fixed_linear(x)


def test_model_average():
    model1 = TestNet(0.25, 0.75)
    model2 = TestNet(0.75, 0.25)
    model_average = ModelAverage(model1, model2)
    model_average.eval()
    tensor_input = torch.randn(2, 3)
    expected_output = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    assert (
        numpy.nextafter(1.0, 2.0, dtype=numpy.float32) >=
        torch.abs(1.0 - model_average(tensor_input).detach() / expected_output)
    ).all()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_model_average()
