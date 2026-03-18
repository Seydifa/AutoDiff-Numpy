# Third-party
import pytest
import numpy as np

# Local imports
import dnp
from dnp.core.tensor import Tensor


def test_linear_layer():
    lin = dnp.Linear(10, 5)
    x = Tensor(np.random.randn(8, 10))
    y = lin(x)
    assert y.shape == (8, 5)
    assert len(list(lin.parameters())) == 2  # W and b


def test_conv2d_layer():
    conv = dnp.Conv2d(3, 16, kernel_size=3, padding="same")
    x = Tensor(np.random.randn(1, 3, 32, 32))
    y = conv(x)
    assert y.shape == (1, 16, 32, 32)
    assert len(list(conv.parameters())) == 2  # weight and bias


def test_pooling_layers():
    x = Tensor(np.random.randn(1, 1, 4, 4))
    maxpool = dnp.MaxPool2d(kernel_size=2)
    avgpool = dnp.AvgPool2d(kernel_size=2)

    assert maxpool(x).shape == (1, 1, 2, 2)
    assert avgpool(x).shape == (1, 1, 2, 2)


def test_sequential():
    model = dnp.Sequential(dnp.Linear(10, 20), dnp.ReLU(), dnp.Linear(20, 5))
    x = Tensor(np.random.randn(8, 10))
    y = model(x)
    assert y.shape == (8, 5)
    assert len(list(model.parameters())) == 4  # (W, b) * 2


def test_normalization():
    bn = dnp.BatchNorm1d(10)
    x = Tensor(np.random.randn(8, 10))
    y = bn(x)
    assert y.shape == (8, 10)

    bn2d = dnp.BatchNorm2d(3)
    x2d = Tensor(np.random.randn(8, 3, 32, 32))
    y2d = bn2d(x2d)
    assert y2d.shape == (8, 3, 32, 32)


def test_dropout():
    dropout = dnp.Dropout(p=0.5)
    x = Tensor(np.ones((10, 10)))
    # In eval mode, dropout is identity
    dropout.eval()
    assert np.all(np.array(dropout(x)) == 1.0)
    # In train mode, some elements should be zeroed
    dropout.train()
    y = dropout(x)
    assert np.any(np.array(y) == 0.0)


def test_activations():
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(np.array(dnp.ReLU()(x)), [0.0, 0.0, 1.0])
    assert np.allclose(np.array(dnp.Sigmoid()(Tensor(np.array([0.0])))), [0.5])


def test_flatten():
    flatten = dnp.Flatten()
    x = Tensor(np.random.randn(8, 3, 32, 32))
    y = flatten(x)
    assert y.shape == (8, 3 * 32 * 32)
