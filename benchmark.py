"""
Compares NumPy (CPU) and CuPy (GPU) matrix multiplication
performance on large square matrices.
"""
import numpy as np
import cupy as cp
import time

N = 10000
iterations = 30

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

A_gpu = cp.asarray(A)
B_gpu = cp.asarray(B)


def matrix_mult_cpu():
    _ = A @ B  # warm-up

    times = []
    for _ in range(iterations):
        start = time.time()
        A @ B
        times.append(time.time() - start)

    return times


def matrix_mult_gpu():
    _ = A_gpu @ B_gpu  # warm-up
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(iterations):
        start = time.time()
        A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()
        times.append(time.time() - start)

    return times


def compare():
    cpu_runtimes = matrix_mult_cpu()
    gpu_runtimes = matrix_mult_gpu()

    avg_cpu = sum(cpu_runtimes) / len(cpu_runtimes)
    avg_gpu = sum(gpu_runtimes) / len(gpu_runtimes)

    speedup = avg_cpu / avg_gpu
    print(f"Matrix size: {N} x {N}")
    print(f"Iterations: {iterations}")    
    print(f"CPU Average: {avg_cpu:.4f}s")
    print(f"GPU Average: {avg_gpu:.4f}s")
    print(f"GPU is {speedup:.2f}x faster than CPU")


compare()