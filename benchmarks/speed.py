"""Throughput of the ssmix mixers vs. parameter-free softmax self-attention.

    python benchmarks/speed.py [--device cuda] [--C 64]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmix import ScaleSpaceMixer, SWTCrossScaleMixer  # noqa: E402


def timeit(fn, x, reps=5):
    """Best-of-reps wall time in ms (the first call warms caches and FFT plans)."""
    fn(x)
    best = float("inf")
    for _ in range(reps):
        if x.is_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def softmax_attention(x):  # Q = K = V = x, no parameters, O(T^2 C)
    a = torch.softmax(x @ x.transpose(1, 2) / x.shape[-1] ** 0.5, dim=-1)
    return a @ x


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--C", type=int, default=64)
    a = p.parse_args()
    dev = torch.device(a.device)
    print(f"device={dev}  threads={torch.get_num_threads()}  B=1  C={a.C}  fp32\n")
    print(f"{'T':>8} {'frame db4 J=8':>14} {'cross-scale k=2':>16} {'softmax attn':>13}   (ms)")
    for T in (1024, 4096, 16384, 65536):
        x = torch.randn(1, T, a.C, device=dev)
        tf = timeit(ScaleSpaceMixer(order=4, levels=8, heads=8), x)
        tc = timeit(SWTCrossScaleMixer(order=4, levels=8, heads=8, window=2), x, reps=3)
        ta = timeit(softmax_attention, x, reps=2) if T <= 16384 else float("nan")
        print(f"{T:>8d} {tf:>14.1f} {tc:>16.1f} {ta:>13.1f}")
    T = 16384
    x = torch.randn(1, T, a.C, device=dev)
    print(f"\nT={T}: frame cost vs Daubechies order (support 2N-1), J=8")
    for N in (1, 4, 16):
        print(f"  db{N:<3d} support={2 * N - 1:<4d} {timeit(ScaleSpaceMixer(order=N, levels=8, heads=8), x):7.1f} ms")


if __name__ == "__main__":
    main()
