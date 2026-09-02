# MXFP8 quantization overhead in a linear layer

The MXFP8 counterpart to [`README_nvfp4_quant_overhead.md`](README_nvfp4_quant_overhead.md).
Same scripts, method and shapes — only the recipe differs:

```bash
python benchmarks/linear/benchmark_nvfp4_quant_overhead.py --recipe mxfp8 ...
python benchmarks/linear/benchmark_nvfp4_grouped_quant_overhead.py --recipe mxfp8 ...
```

## What differs from NVFP4

**No global amax.** MXFP8 scales are per-32-element E8M0 values computed inside
the block, so nothing has to be reduced over the whole tensor first. The cast is
one kernel, swizzle included:

| recipe | usage | kernels | total |
| --- | --- | --- | --- |
| NVFP4 | rowwise | `zero_amax` + `amax` + `quantize_transpose` + `swizzle` | 55.5 µs |
| NVFP4 | both | `zero_amax` + `amax` + `quantize_transpose` + 2x `swizzle` | 76.6 µs |
| **MXFP8** | rowwise | `mxfp8::quantize_kernel` | **26.0 µs** |
| **MXFP8** | both | `mxfp8::quantize_kernel` | **41.4 µs** |

There is no `amax us` column below; the scripts drop it when it would be all
zeros.

**More bytes per pass, half as many passes.** MXFP8 writes 8-bit data plus an
E8M0 scale per 32 elements (1.03125 B/elem/usage) against NVFP4's 4-bit plus an
fp8 scale per 16 (0.5625) — but reads once instead of twice:

    NVFP4 single-usage:  2 reads + 0.5625 write  = 4.5625 B/elem
    MXFP8 single-usage:  1 read  + 1.03125 write = 3.03125 B/elem

**Half the GEMM throughput.** MXFP8 runs on FP8 tensor cores, NVFP4 on FP4, so
every GEMM here is ~2x slower. That dominates the percentages.

Grouped MXFP8 has **no RHT requirement**, unlike grouped NVFP4.

## Environment

`NVIDIA B200` (148 SMs, cc 10.0, max SM clock 1965 MHz), TE
`2.20.0.dev0+1ff9c372`, torch `2.11.0+cu130`, CUDA 13.2, cuBLAS `13.4.1.3`.
Weight amortized, `--iters 100 --repeats 7`. Times in µs. `cast GB/s` reference:
a bf16 copy sustains **6490 GB/s** here.

Every number on this page and every Part 1 number in the NVFP4 README were
measured back-to-back in one session, so they are directly comparable. (The NVFP4
*grouped* table is from an earlier clock state — see the note there.)

---

# Part 1 — Dense

```bash
python benchmarks/linear/benchmark_nvfp4_quant_overhead.py --recipe mxfp8 \
    --layers fc1,dense_fc1 --iters 100 --repeats 7 --amortize-weight --step-total --breakdown
```

### Per-GEMM

| layer | (M,N,K) | gemm | gemm us | cast us | cast GB/s | total us | overhead us | cast % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | fprop | 152.9 | 26.9 | 6624 | 197.5 | 44.6 | 22.6 |
| fc1 (MoE) | (8192,4096,7168) | dgrad | 154.4 | 18.3 | 5555 | 189.0 | 34.6 | 18.3 |
| fc1 (MoE) | (8192,4096,7168) | wgrad | 158.0 | 53.2 | 5259 | 221.1 | 63.1 | 28.6 |
| fc1 (MoE) | (16384,4096,7168) | fprop | 303.3 | 52.7 | 6757 | 387.4 | 84.1 | 21.7 |
| fc1 (MoE) | (16384,4096,7168) | dgrad | 309.7 | 35.0 | 5817 | 375.4 | 65.7 | 17.5 |
| fc1 (MoE) | (16384,4096,7168) | wgrad | 331.9 | 84.8 | 6595 | 448.9 | 116.9 | 26.1 |
| fc1 (dense) | (8192,36864,7168) | fprop | 1489.3 | 30.3 | 5880 | 1535.2 | 45.9 | 3.0 |
| fc1 (dense) | (8192,36864,7168) | dgrad | 1539.7 | 152.1 | 6019 | 1701.0 | 161.3 | 9.5 |
| fc1 (dense) | (8192,36864,7168) | wgrad | 1502.5 | 160.7 | 6806 | 1658.9 | 156.4 | 9.4 |
| fc1 (dense) | (16384,36864,7168) | fprop | 3024.1 | 56.7 | 6281 | 3074.3 | 50.2 | 1.6 |
| fc1 (dense) | (16384,36864,7168) | dgrad | 3213.8 | 258.9 | 7070 | 3475.8 | 262.0 | 7.5 |
| fc1 (dense) | (16384,36864,7168) | wgrad | 3104.5 | 312.4 | 7001 | 3427.3 | 322.8 | 9.4 |

### One training pass

| layer | (M,N,K) | gemm us | cast us | cast GB/s | total us | overhead us | cast % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | 494.9 | 80.1 | 4681 | 591.0 | 96.1 | 16.3 | 2916 |
| fc1 (MoE) | (16384,4096,7168) | 1057.0 | 125.1 | 5993 | 1171.4 | 114.5 | 9.8 | 2731 |
| fc1 (dense) | (8192,36864,7168) | 4612.9 | 236.0 | 6208 | 4865.1 | 252.2 | 5.2 | 2816 |
| fc1 (dense) | (16384,36864,7168) | 9347.1 | 460.5 | 6364 | 9838.9 | 491.8 | 5.0 | 2779 |

---

# Part 2 — Grouped MoE

Same recipe as Part 1 — no RHT needed. Only the fused dual-usage (`both`) cast.

```bash
python benchmarks/linear/benchmark_nvfp4_grouped_quant_overhead.py --recipe mxfp8 \
    --experts 8 --tokens-per-expert 512,1024,2048 \
    --iters 100 --repeats 7 --amortize-weight --step-total --breakdown
```

| E | tok/E | (M,N,K) | gemm us | cast us | cast GB/s | total us | overhead us | cast % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 512 | (4096,4096,7168) | 362.2 | 53.2 | 3526 | 443.4 | 81.2 | 18.3 | 1992 |
| 8 | 1024 | (8192,4096,7168) | 631.7 | 78.2 | 4791 | 735.2 | 103.5 | 14.1 | 2285 |
| 8 | 2048 | (16384,4096,7168) | 1128.9 | 136.0 | 5513 | 1288.0 | 159.1 | 12.4 | 2557 |

---

## Observations

**Quantization overhead is roughly half NVFP4's**, one training pass:

| shape | MXFP8 | NVFP4 |
| --- | --- | --- |
| fc1 (MoE) M=8192 | 16.3% | 36.4% |
| fc1 (MoE) M=16384 | 9.8% | 31.4% |
| fc1 (dense) M=8192 | 5.2% | 14.2% |
| fc1 (dense) M=16384 | 5.0% | 13.3% |

**But that is not a free win.** Two effects pull the same way: the cast is
cheaper (one pass) *and* the GEMM is slower (~2800 vs ~5600 TFLOP/s), so the same
cast hides inside a longer GEMM. In absolute terms a dense fc1 step at M=16384
takes **9839 µs under MXFP8 against 5348 µs under NVFP4** — MXFP8 buys a smaller
quantization *fraction* by enlarging the denominator.

**The cast is bandwidth-saturated** — 5259–7070 GB/s against the 6490 GB/s copy
reference, several rows at or above a plain copy. No headroom worth chasing,
unlike NVFP4 where amax alone is a third of cast cost and is removable.

**wgrad is still the worst GEMM** (26–29% MoE, 9.4% dense): both operands cast
columnwise, neither is the weight.

**Grouped is the exception to saturation** — 3526 GB/s at 512 tok/E rising to
5513 at 2048, the same occupancy limit the NVFP4 grouped path shows.
