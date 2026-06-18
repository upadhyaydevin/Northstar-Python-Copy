"""
Robust NumPy (CPU) vs CuPy (GPU) matrix multiplication benchmark.
"""

import numpy as np
import cupy as cp
import time
import statistics
import os

# ================= CONFIG =================
N = 12000  
ITERATIONS = 10
WARMUP = 3
DTYPE = np.float32
# ================= UTIL =================
def report(label, times):
    print(f"\n{label}")
    print(f"  Mean:   {statistics.mean(times):.4f}s")
    print(f"  Median: {statistics.median(times):.4f}s")
    print(f"  Min:    {min(times):.4f}s")


def print_hardware_info():
    print("===== Hardware Info =====")
    print("CPU threads (NumPy may use):", os.cpu_count())
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print("GPU:", props["name"].decode())
        print("GPU memory (GB):", props["totalGlobalMem"] / 1e9)
    except Exception:
        print("No GPU detected")
    print("=========================\n")


# ================= BENCHMARKS =================
def cpu_benchmark(A, B):
    # warmup
    for _ in range(WARMUP):
        C = A @ B

    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        times.append(end - start)
    return times


def gpu_compute_only(A_gpu, B_gpu):

    """
    Benchmark GPU compute ONLY (no data transfer).
    Assumes A_gpu and B_gpu are already on device.
    """

    # Warm-up: loads kernels, initializes GPU context
    for _ in range(WARMUP):
        C = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(ITERATIONS):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        C = A_gpu @ B_gpu
        end.record()

        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1000)
    return times


def gpu_transfer_only(A, B):
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return times


def gpu_end_to_end(A, B):
    times = []
    for _ in range(WARMUP):
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()
        _ = cp.asnumpy(C)

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C = A_gpu @ B_gpu
        _ = cp.asnumpy(C)

        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    return times

# ================= MAIN =================
if __name__ == "__main__":
    print_hardware_info()

    print(f"\n===== Matrix Size: {N} x {N} =====")

    A = np.random.rand(N, N).astype(DTYPE)
    B = np.random.rand(N, N).astype(DTYPE)

    # CPU
    cpu_times = cpu_benchmark(A, B)

    # GPU (reuse memory for compute-only)
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    gpu_times = gpu_compute_only(A_gpu, B_gpu)

    # transfer-only
    transfer_times = gpu_transfer_only(A, B)

    # end-to-end
    e2e_times = gpu_end_to_end(A, B)

    # REPORT
    report("CPU (NumPy)", cpu_times)
    report("GPU compute only (CuPy)", gpu_times)
    report("GPU transfer only", transfer_times)
    report("GPU end-to-end", e2e_times)

    print("\nSpeedups:")
    print(f"  Compute only: {statistics.mean(cpu_times)/statistics.mean(gpu_times):.2f}x")
    print(f"  End-to-end:   {statistics.mean(cpu_times)/statistics.mean(e2e_times):.2f}x")