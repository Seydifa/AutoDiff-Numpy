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


def test_global_pooling_layers():
    x1d = Tensor(np.array([[[1.0, 3.0, 2.0], [4.0, 6.0, 8.0]]]))
    x2d = Tensor(
        np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                ]
            ]
        )
    )

    global_avg_1d = dnp.GlobalAvgPool1d()
    global_max_1d = dnp.GlobalMaxPool1d()
    global_avg_2d = dnp.GlobalAvgPool2d()
    global_max_2d = dnp.GlobalMaxPool2d(keepdims=True)

    assert global_avg_1d(x1d).shape == (1, 2)
    assert global_max_1d(x1d).shape == (1, 2)
    assert np.allclose(np.array(global_avg_1d(x1d)), np.array([[2.0, 6.0]]))
    assert np.allclose(np.array(global_max_1d(x1d)), np.array([[3.0, 8.0]]))

    assert global_avg_2d(x2d).shape == (1, 2)
    assert global_max_2d(x2d).shape == (1, 2, 1, 1)
    assert np.allclose(np.array(global_avg_2d(x2d)), np.array([[2.5, 25.0]]))
    assert np.allclose(np.array(global_max_2d(x2d)), np.array([[[[4.0]], [[40.0]]]]))


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


def test_utility_shape_layers():
    x = Tensor(np.arange(6).reshape(2, 3))

    reshape = dnp.Reshape(3, 2)
    expand = dnp.ExpandDims(axis=1)
    squeeze = dnp.Squeeze(axis=1)
    repeat = dnp.Repeat(repeats=2, axis=1)

    assert reshape(x).shape == (3, 2)
    assert expand(x).shape == (2, 1, 3)
    assert squeeze(expand(x)).shape == (2, 3)
    assert repeat(x).shape == (2, 6)
    assert np.allclose(
        np.array(repeat(x)), np.array([[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5]])
    )


def test_stack_and_concatenate_layers():
    x1 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    x2 = Tensor(np.array([[10.0, 20.0], [30.0, 40.0]]))

    concat = dnp.Concatenate(axis=0)
    stack = dnp.Stack(axis=1)

    concatenated = concat(x1, x2)
    stacked = stack(x1, x2)

    assert concatenated.shape == (4, 2)
    assert stacked.shape == (2, 2, 2)
    assert np.allclose(
        np.array(concatenated),
        np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]]),
    )
    assert np.allclose(
        np.array(stacked),
        np.array([[[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [30.0, 40.0]]]),
    )


def test_conv1d():
    import numpy as np
    from dnp.core.tensor import Tensor
    from dnp.layers import Conv1d

    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 10))  # (batch, in_channels, L)
    conv = Conv1d(in_channels=3, out_channels=4, kernel_size=3, padding=1)

    out = conv(x)
    assert out.shape == (2, 4, 10)  # L_out = L with padding=1, kL=3

    # Check backward pass mathematically
    from dnp.core import ops

    loss = ops.sum(out)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 3, 10)
    assert conv.W.grad is not None
    assert conv.W.grad.shape == (4, 3, 3)


def test_rnn_layers():
    import numpy as np
    from dnp.core.tensor import Tensor
    from dnp.core import ops
    from dnp.layers.advanced import RNN, LSTM, GRU

    np.random.seed(42)
    x = Tensor(np.random.randn(2, 5, 10))  # B, T, d_in

    rnn = RNN(10, 20)
    out_rnn = rnn(x)
    assert out_rnn.shape == (2, 5, 20)
    ops.sum(out_rnn).backward()
    assert x.grad is not None

    x.zero_grad()  # Reset grad for next tests if it cumulates

    lstm = LSTM(10, 20)
    out_lstm = lstm(x)
    assert out_lstm.shape == (2, 5, 20)
    ops.sum(out_lstm).backward()
    assert x.grad is not None

    x.zero_grad()

    gru = GRU(10, 20)
    out_gru = gru(x)
    assert out_gru.shape == (2, 5, 20)
    ops.sum(out_gru).backward()
    assert x.grad is not None


def test_additional_loss_layers():
    y_pred = Tensor(np.array([1.0, 3.0, 2.0]))
    y_true = Tensor(np.array([2.0, 1.0, 2.0]))

    mae = dnp.MAELoss(reduction="sum")
    l1 = dnp.L1Loss(reduction="mean")
    huber = dnp.HuberLoss(delta=1.0, reduction="sum")
    smooth_l1 = dnp.SmoothL1Loss(beta=1.0, reduction="sum")
    log_cosh = dnp.LogCoshLoss(reduction="sum")
    hinge = dnp.HingeLoss(reduction="sum")
    squared_hinge = dnp.SquaredHingeLoss(reduction="sum")

    assert np.isclose(float(mae(y_pred, y_true)), 3.0)
    assert np.isclose(float(l1(y_pred, y_true)), 1.0)
    assert np.isclose(float(huber(y_pred, y_true)), 2.0)
    assert np.isclose(float(smooth_l1(y_pred, y_true)), 2.0)
    assert np.isclose(
        float(log_cosh(y_pred, y_true)),
        float(np.log(np.cosh(np.array([-1.0, 2.0, 0.0]))).sum()),
    )

    margin_pred = Tensor(np.array([2.0, -0.5, 0.0]))
    margin_true = Tensor(np.array([1.0, -1.0, 1.0]))
    assert np.isclose(float(hinge(margin_pred, margin_true)), 1.5)
    assert np.isclose(float(squared_hinge(margin_pred, margin_true)), 1.25)


def test_probabilistic_loss_layers():
    log_probs = Tensor(np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])))
    targets = np.array([0, 2])
    nll = dnp.NLLLoss()
    expected_nll = -0.5 * (np.log(0.7) + np.log(0.6))
    assert np.isclose(float(nll(log_probs, targets)), expected_nll)

    p = Tensor(np.array([[0.6, 0.4], [0.5, 0.5]]))
    q = Tensor(np.array([[0.5, 0.5], [0.4, 0.6]]))
    kl = dnp.KLDivLoss(reduction="sum")
    expected_kl = np.sum(
        np.array([[0.6, 0.4], [0.5, 0.5]])
        * (
            np.log(np.array([[0.6, 0.4], [0.5, 0.5]]))
            - np.log(np.array([[0.5, 0.5], [0.4, 0.6]]))
        )
    )
    assert np.isclose(float(kl(p, q)), expected_kl)

    focal = dnp.FocalLoss(gamma=2.0, alpha=0.25)
    logits = Tensor(np.array([0.0, 2.0, -2.0]))
    labels = Tensor(np.array([1.0, 1.0, 0.0]))
    assert focal(logits, labels).shape == ()
    assert float(focal(logits, labels)) >= 0.0
