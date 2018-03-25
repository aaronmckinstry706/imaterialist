import enum
import numpy
import numpy.testing as testing
import torch
import torch.autograd as autograd
import torchvision.models as models
import torch.nn as nn


class ResNetCopy(nn.Module):

    class AllowedLayers(enum.Enum):
        PENULTIMATE = 'penultimate'
        ULTIMATE = 'utlimate'

    def __init__(self):
        super().__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.layer1 = resnet34.layer1
        self.layer2 = resnet34.layer2
        self.layer3 = resnet34.layer3
        self.layer4 = resnet34.layer4
        self.avgpool = resnet34.avgpool
        self.fc = resnet34.fc
        self.resnet34 = resnet34
        self._layer = ResNetCopy.AllowedLayers.PENULTIMATE

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, value: AllowedLayers):
        if not isinstance(value, ResNetCopy.AllowedLayers):
            raise TypeError('ResNetCopy.layer must be an instance of AllowedLayers.')
        self._layer = value

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.layer == ResNetCopy.AllowedLayers.PENULTIMATE:
            x = self.fc(x.view(-1, 512))
        return x


def test_resnet34_output_same_as_output_of_copy():
    penult_resnet = ResNetCopy()
    penult_resnet.layer = ResNetCopy.AllowedLayers.PENULTIMATE
    network_input = autograd.Variable(torch.rand(1, 3, 224, 224))
    my_output = penult_resnet(network_input).data.numpy()
    original_output = penult_resnet.resnet34(network_input).data.numpy()
    assert my_output.shape == original_output.shape, (
        "My class's output shape {} does not match original output shape {}".format(
            my_output.shape, original_output.shape))
    testing.assert_almost_equal(my_output / original_output, numpy.ones(my_output.shape))


test_resnet34_output_same_as_output_of_copy()