import numpy as np
import time
import dnp
from dnp.core.layers import Conv2d, MaxPool2d, AvgPool2d
from dnp.core import Tensor

def benchmark_conv2d():
    print("--- Benchmarking Conv2d ---")
    batch_size = 4
    in_channels = 3
    out_channels = 8
    H, W = 64, 64
    kH, kW = 3, 3
    
    x_np = np.random.randn(batch_size, in_channels, H, W).astype(np.float32)
    x = Tensor(x_np)
    
    conv = Conv2d(in_channels, out_channels, kernel_size=(kH, kW), padding=1)
    
    # Warmup
    _ = conv(x)
    
    start = time.time()
    for _ in range(10):
        out = conv(x)
    end = time.time()
    print(f"Numba Conv2d forward time (10 runs): {end - start:.4f}s")
    
    # Verify shape
    print(f"Output shape: {out.shape}")
    assert out.shape == (batch_size, out_channels, H, W)

def benchmark_pooling():
    print("\n--- Benchmarking Pooling Backward ---")
    batch_size = 4
    channels = 16
    H, W = 32, 32
    
    x_np = np.random.randn(batch_size, channels, H, W).astype(np.float32)
    x = Tensor(x_np)
    
    # MaxPool
    pool = MaxPool2d(kernel_size=2, stride=2)
    y = pool(x)
    
    # Warmup
    y.backward(np.ones_like(y))
    x.grad.fill(0)
    
    start = time.time()
    for _ in range(10):
        y.backward(np.ones_like(y))
        x.grad.fill(0)
    end = time.time()
    print(f"Numba MaxPool backward time (10 runs): {end - start:.4f}s")

    # AvgPool
    pool_avg = AvgPool2d(kernel_size=2, stride=2)
    y_avg = pool_avg(x)
    
    # Warmup
    y_avg.backward(np.ones_like(y_avg))
    x.grad.fill(0)
    
    start = time.time()
    for _ in range(10):
        y_avg.backward(np.ones_like(y_avg))
        x.grad.fill(0)
    end = time.time()
    print(f"Numba AvgPool backward time (10 runs): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_conv2d()
    benchmark_pooling()
    print("\n✅ Verification complete!")
