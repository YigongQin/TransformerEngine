# NVFP4 quantization overhead in a linear layer

How much of a quantized linear layer is quantization rather than math?

| script | path measured |
| --- | --- |
| `benchmark_quant_overhead.py` | dense `general_gemm`, one shape at a time |
| `benchmark_grouped_quant_overhead.py` | `group_quantize` + `general_grouped_gemm_for_grouped_tensor` across experts |

MXFP8 results — one-pass cast, no global amax — are in
[`README_mxfp8_quant_overhead.md`](README_mxfp8_quant_overhead.md).

## Method

Three loops per GEMM, sharing one set of quantized operands: `gemm`
(pre-quantized), `quant` (casts alone), `quant+gemm`. Then
`overhead = (quant+gemm) − gemm`, and `amax+cast %` is that overhead as a
fraction of the full path. Both loops use identical operands and a
byte-identical GEMM, so the difference isolates the cast.

An NVFP4 cast is **two passes** — a full-tensor amax, then the cast itself —
reported separately via `--breakdown`. Everything that is not amax is bucketed as
`cast`, so the two columns sum to the total cast cost. `cast GB/s` covers the cast
pass alone, against `elems * (2 + n_usages * 0.5625)`.

`--amortize-weight` models gradient accumulation: the weight is cast once per
optimizer step, so its cost is excluded.

`--step-total` measures one full training pass — all three GEMMs with X, W and dY
each cast **once**, measured directly rather than summed. Summing the per-GEMM
rows would be wrong twice over: the fused cast+cast-transpose reads a tensor once
and writes both layouts, so three GEMMs need 6 single-usage casts where a step
needs 3 dual-usage ones; and each per-GEMM row carries a fixed ~10–25 µs penalty
(the `gemm` baseline runs cache-warm in a way production never is) that a step
pays once, not three times. **Read the per-GEMM rows for the ordering of
fprop/dgrad/wgrad, and the step rows for the actual fraction.**

## Environment

`NVIDIA B200` (148 SMs, cc 10.0, **max SM clock 1965 MHz**), TE
`2.20.0.dev0+1ff9c372`, torch `2.11.0+cu130`, CUDA 13.2, cuBLAS `13.4.1.3`.
Weight amortized, `--iters 100 --repeats 7`. Times in µs. Reference for
`cast GB/s`: a trivial bf16 copy sustains **6490 GB/s** here.

`gemm us` is reproducible to ±0.5% back-to-back.

Requires Blackwell (SM100+); grouped GEMM additionally needs cuBLAS 13.3+.

---

# Part 1 — Dense

Plain 1D NVFP4: no RHT, no stochastic rounding, no 2D scaling.

```bash
python benchmarks/linear/benchmark_quant_overhead.py \
    --layers fc1,dense_fc1 --iters 100 --repeats 7 --amortize-weight --step-total --breakdown
```

`fc1 (MoE)` is a DeepSeek-V3 MoE expert (58 of 61 layers); `fc1 (dense)` is the
dense MLP in the first 3. DeepSeek-V3's config lists `moe_intermediate_size = 2048`
and declares `gate_proj`/`up_proj` separately, but TE concatenates them
(`fc1_output_features = 2 * ffn_hidden_size`), so the dispatched GEMM is `N = 4096`.
These are the GEMMs TE issues, not the projections the config names.

`optimize_for_gemm` is on by default. On this RHT-off path it *relocates* the
swizzle rather than fusing it — 3 kernels and no swizzle with the flag off, 4
including a separate 5.5 µs `swizzle_row_scaling_kernel` with it on — which is why
enabling it is roughly net-neutral. It is genuinely fused only on the RHT path
(Part 2). `te.Linear` does not set it today, so Linear still re-swizzles inside
every GEMM call; `--no-fused-swizzle` reproduces that.

### Per-GEMM

| layer | (M,N,K) | gemm | gemm us | amax us | cast us | total us | overhead us | amax+cast % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | fprop | 78.8 | 25.8 | 28.8 | 144.9 | 66.0 | 45.6 |
| fc1 (MoE) | (8192,4096,7168) | dgrad | 83.7 | 13.6 | 16.4 | 125.4 | 41.7 | 33.3 |
| fc1 (MoE) | (8192,4096,7168) | wgrad | 82.1 | 43.1 | 72.9 | 200.3 | 118.2 | 59.0 |
| fc1 (MoE) | (16384,4096,7168) | fprop | 153.1 | 46.3 | 50.1 | 264.3 | 111.2 | 42.1 |
| fc1 (MoE) | (16384,4096,7168) | dgrad | 163.2 | 30.2 | 32.8 | 240.0 | 76.8 | 32.0 |
| fc1 (MoE) | (16384,4096,7168) | wgrad | 160.8 | 72.4 | 126.5 | 367.5 | 206.7 | 56.2 |
| fc1 (dense) | (8192,36864,7168) | fprop | 737.1 | 25.4 | 29.6 | 832.3 | 95.2 | 11.4 |
| fc1 (dense) | (8192,36864,7168) | dgrad | 770.9 | 110.4 | 116.6 | 1009.4 | 238.5 | 23.6 |
| fc1 (dense) | (8192,36864,7168) | wgrad | 725.6 | 124.6 | 231.3 | 1120.0 | 394.4 | 35.2 |
| fc1 (dense) | (16384,36864,7168) | fprop | 1538.4 | 46.5 | 49.8 | 1622.5 | 84.1 | 5.2 |
| fc1 (dense) | (16384,36864,7168) | dgrad | 1569.5 | 212.5 | 226.2 | 1987.9 | 418.4 | 21.0 |
| fc1 (dense) | (16384,36864,7168) | wgrad | 1498.8 | 229.2 | 447.1 | 2154.4 | 655.5 | 30.4 |

### One training pass

| layer | (M,N,K) | gemm us | amax us | cast us | total us | overhead us | amax+cast % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | 250.6 | 43.9 | 84.4 | 393.8 | 143.2 | 36.4 | 5759 |
| fc1 (MoE) | (16384,4096,7168) | 503.5 | 75.5 | 135.8 | 734.4 | 230.9 | 31.4 | 5733 |
| fc1 (dense) | (8192,36864,7168) | 2335.5 | 130.6 | 246.5 | 2723.4 | 387.9 | 14.2 | 5561 |
| fc1 (dense) | (16384,36864,7168) | 4639.1 | 240.9 | 467.3 | 5348.0 | 708.9 | 13.3 | 5599 |

### Observations

**MoE shapes pay ~2.5x the dense overhead, purely from shape** — 31–36% vs 13–14%
for a training pass. Same kernel, same code path; only `N` differs (4096 vs
36864). The shorter GEMM cannot hide a cast whose cost scales with bytes.

**amax is 33–36% of cast cost** in the step rows, and it is a full read that
produces no output. Removing it — fusing amax into the producer, or a
delayed/cached amax as FP8 recipes use — is worth more than tuning the cast.

**The cast kernel is near bandwidth-saturated.** `cast GB/s` runs 3246–6842
against the 6490 GB/s copy rate. wgrad is weakest (3246–4135), matching a
standalone columnwise cast — strided writes cannot reach copy bandwidth.

**wgrad is the worst GEMM everywhere** (30–59%): both operands cast columnwise,
and neither is the weight, so amortization cannot help it.

### Caveats

- The `fc1 (MoE)` rows here are a **shape**, not TE's MoE path — a dense
  `general_gemm` at one expert's dimensions, with `M` as tokens *per expert*.
  Part 2 measures the real path.
- Quant-side columns are a difference of two end-to-end measurements; on small
  shapes they can swing several points between runs. Raise `--repeats`.
- Rows where `total <= gemm` are physically impossible and are dropped with a
  warning.

---

# Part 2 — Grouped MoE

`tex.group_quantize` into packed `GroupedTensor` storage, then
`general_grouped_gemm_for_grouped_tensor` across all local experts in one launch.
Weights cast per expert (discrete list); activations and gradients packed and
group-cast, matching `ops/fused/grouped_mlp.py`. Same fc1 shape, `E=8` local
experts (DeepSeek-V3 at EP=32).

```bash
python benchmarks/linear/benchmark_grouped_quant_overhead.py \
    --experts 8 --tokens-per-expert 512,1024,2048 \
    --iters 100 --repeats 7 --amortize-weight --step-total --breakdown
```

> **RHT is forced on.** Grouped NVFP4 has no non-RHT kernel — `group_quantize`
> raises *"graph safe grouped quant kernel for non-RHT path is not ready yet"*
> (`pytorch/csrc/extensions/cast.cpp`). Part 1 runs RHT off, so the two are not
> directly comparable.

### One training pass

Only the fused dual-usage (`both`) cast is reported: it is what a real step
performs, and the configuration where the swizzle is genuinely fused into the
cast kernel.

| E | tok/E | (M,N,K) | gemm us | amax us | cast us | total us | overhead us | amax+cast % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 512 | (4096,4096,7168) | 269.0 | 28.3 | 65.3 | 365.0 | 96.0 | 26.3 | 2682 |
| 8 | 1024 | (8192,4096,7168) | 363.7 | 47.5 | 107.9 | 526.1 | 162.5 | 30.9 | 3968 |
| 8 | 2048 | (16384,4096,7168) | 629.6 | 74.8 | 184.1 | 890.5 | 260.9 | 29.3 | 4584 |

### Observations

**The swizzle really is fused here** — 3 kernels with `optimize_for_gemm` on or
off (72.1 vs 71.6 µs), versus Part 1 where the flag adds a separate kernel.

**GEMM efficiency climbs sharply with tokens per expert** (2682 → 4584 TFLOP/s):
at 512 tok/E the `M=4096` problem is split across 8 small expert GEMMs. The cast
scales alongside, so `amax+cast %` stays roughly flat at 26–31%.

### Caveats

- **`cast us` includes allocation.** `tex.group_quantize` allocates its output
  every call; the `output=` reuse path requires uniform shapes
  (`first_dims=None`), and that path dies with an illegal memory access in
  `group_row_col_rht_gemm_ntt_w_sfc_graph_safe`, reproducible on a single call.
  Part 1 preallocates via `update_quantized` and does not carry this.
- Token counts are uniform across experts; real routing is imbalanced.
