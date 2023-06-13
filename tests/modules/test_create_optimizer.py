import sys

sys.path.append(".")

import pytest

from mindspore import nn

from mindyolo.optim import create_optimizer


class SimpleCNN(nn.Cell):
    def __init__(self, in_channels=1):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        return x


@pytest.mark.parametrize("optimizer", ['momentum', 'sgd'])
@pytest.mark.parametrize("nesterov", [True, False])
def test_create_optimizer(optimizer, nesterov):
    model = SimpleCNN()
    model.set_train(True)
    optimizer = create_optimizer(params=model.trainable_params(), optimizer=optimizer, nesterov=nesterov)
    assert optimizer is not None


if __name__ == '__main__':
    test_create_optimizer()
