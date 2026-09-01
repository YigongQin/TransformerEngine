# NVFP4 quantization overhead in a linear layer

How much of a quantized linear layer is quantization rather than math?

Two scripts, both in this directory:

| script | path measured | recipe |
| --- | --- | --- |
| `benchmark_nvfp4_quant_overhead.py` | dense `general_gemm`, one shape at a time | plain 1D NVFP4, **RHT off** |
| `benchmark_nvfp4_grouped_quant_overhead.py` | `group_quantize` + `general_grouped_gemm_for_grouped_tensor` across experts | 1D NVFP4, **RHT forced on** |

They use different recipes and are **not directly comparable**; see
[Comparing the two](#comparing-the-two).

## Environment

All numbers below: `NVIDIA B200` (148 SMs, 178 GiB, cc 10.0), TE
`2.20.0.dev0+1ff9c372`, torch `2.11.0+cu130`, cuDNN 9.19, CUDA 13.2, cuBLAS
`13.4.1.3`. Weight amortized, `--iters 100 --repeats 7`. Times in µs.

cuBLAS 13.3+ is required for grouped GEMM, and the cuBLAS version moves `gemm us`
noticeably, so every number here was measured on 13.4. Requires Blackwell
(SM100+).

## How it's measured

Both scripts time three loops per GEMM, sharing one set of quantized operands:

| loop | what it runs |
| --- | --- |
| `gemm` | pre-quantized operands into the GEMM — raw NVFP4 math |
| `quant` | the cast kernels alone |
| `quant+gemm` | cast, then that same GEMM |

`overhead = (quant+gemm) − gemm`, and `quant %` is that overhead as a fraction of
the full path. Both loops use identical operands and a byte-identical GEMM, so
the difference isolates the cast.

`--amortize-weight` models gradient accumulation: the weight is cast once per
optimizer step and reused across microbatches, so its cost is excluded. Without
it, the weight is re-cast every step (microbatch = 1).

`--step-total` measures one full training pass — all three GEMMs with X, W and dY
each cast **once** using a fused rowwise+columnwise kernel. Measured directly, not
summed from the per-GEMM rows: X and dY are each consumed by two GEMMs, so summing
charges them twice through two single-usage casts instead of one dual-usage cast,
and overstates quantization badly (42% vs 38.3% for dense MoE below).

### Prefer the step rows over the per-GEMM rows

Every per-GEMM row carries a **fixed ~10–25 µs penalty** that a real training pass
pays roughly once rather than three times. It shows up as `overhead us` exceeding
`quant us` in essentially every per-GEMM row, which is impossible if
`total = quant + gemm`.

The cause is GPU-side, not CPU dispatch (measured: 66 µs CPU vs 113 µs GPU for a
full loop, so the loop is GPU-bound). About half is **L2 cold-cache**: in the
`gemm`-only loop, consecutive identical GEMMs keep their operands resident in
B200's ~126 MB L2, but in `quant+gemm` the cast evicts them and the GEMM re-reads
from HBM. Measured directly on the grouped fc1 at 512 tok/E, the same GEMM takes
61.5 µs back-to-back and 104.2 µs after an explicit L2 flush (flush itself
36.8 µs) — a +5.8 µs cold penalty. The remainder is kernel-boundary effects.

So the `gemm` baseline is measured cache-warm in a way production never is, which
understates it and inflates `overhead`. The penalty is roughly fixed, so it
distorts short GEMMs most — exactly the MoE shapes.

Concretely, at 512 tok/E the three grouped rows sum to 179.8 µs of overhead
against a measured step of 94.6 µs. Of that 85 µs gap, 54 µs is the double-cast
above and ~33 µs is this fixed penalty counted three times.

**Read the per-GEMM tables for the relative ordering of fprop/dgrad/wgrad, and the
step tables for the actual quantization fraction.**

---

# Part 1 — Dense

`benchmark_nvfp4_quant_overhead.py`. Plain 1D NVFP4: no RHT, no stochastic
rounding, no 2D scaling. Issues a dense `general_gemm` per shape.

```bash
# every DeepSeek-V3 linear shape, M in {8192, 16384}
python benchmarks/linear/benchmark_nvfp4_quant_overhead.py

# the numbers below
python benchmarks/linear/benchmark_nvfp4_quant_overhead.py \
    --layers fc1,dense_fc1 --iters 100 --repeats 7 --amortize-weight --step-total
```

The scale-factor swizzle is fused into the cast by default (`optimize_for_gemm`),
so no standalone swizzle pass exists in the timed path. This is *not* what
`te.Linear` does today — `optimize_for_gemm` defaults to `False` and only the
attention paths enable it, so Linear still re-swizzles inside every GEMM call.
`--no-fused-swizzle` reproduces that behaviour; measured directly, the swizzle is
only ~8–13 µs on these shapes.

Two shapes are reported: `fc1 (MoE)` is a DeepSeek-V3 MoE expert (58 of 61
layers), `fc1 (dense)` is the dense MLP used in the first 3 layers.

**On the fc1 `N`.** DeepSeek-V3's config lists `moe_intermediate_size = 2048`, and
its HF reference declares `gate_proj` and `up_proj` as two separate `(7168, 2048)`
linears. TE concatenates them: for any gated activation,
`fc1_output_features = 2 * ffn_hidden_size` (`pytorch/module/layernorm_mlp.py`),
so the GEMM actually dispatched is `N = 4096`. The shapes below are the GEMMs TE
issues, not the projections the config names — hence fc1 at `N=4096` while fc2
takes `K=2048`, since SwiGLU halves 4096 back down before `down_proj`. Same for
the dense MLP: `intermediate_size = 18432`, fc1 `N = 36864`.

### Per-GEMM

| layer | (M,N,K) | gemm | gemm us | quant us | total us | overhead us | quant % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | fprop | 83.1 | 54.1 | 156.7 | 73.5 | 46.9 |
| fc1 (MoE) | (8192,4096,7168) | dgrad | 87.9 | 30.6 | 133.1 | 45.2 | 34.0 |
| fc1 (MoE) | (8192,4096,7168) | wgrad | 85.4 | 115.1 | 205.7 | 120.3 | 58.5 |
| fc1 (MoE) | (16384,4096,7168) | fprop | 164.2 | 96.2 | 288.7 | 124.5 | 43.1 |
| fc1 (MoE) | (16384,4096,7168) | dgrad | 172.0 | 65.1 | 258.7 | 86.7 | 33.5 |
| fc1 (MoE) | (16384,4096,7168) | wgrad | 168.0 | 198.4 | 385.6 | 217.6 | 56.4 |
| fc1 (dense) | (8192,36864,7168) | fprop | 814.4 | 56.7 | 882.3 | 67.9 | 7.7 |
| fc1 (dense) | (8192,36864,7168) | dgrad | 852.0 | 233.7 | 1075.2 | 223.2 | 20.8 |
| fc1 (dense) | (8192,36864,7168) | wgrad | 796.4 | 356.3 | 1191.9 | 395.5 | 33.2 |
| fc1 (dense) | (16384,36864,7168) | fprop | 1668.2 | 96.1 | 1764.8 | 96.6 | 5.5 |
| fc1 (dense) | (16384,36864,7168) | dgrad | 1695.2 | 455.0 | 2168.3 | 473.0 | 21.8 |
| fc1 (dense) | (16384,36864,7168) | wgrad | 1602.4 | 684.5 | 2342.0 | 739.6 | 31.6 |

### One training pass

| layer | (M,N,K) | gemm us | quant us | total us | overhead us | quant % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | 263.9 | 128.5 | 427.8 | 164.0 | 38.3 | 5469 |
| fc1 (MoE) | (16384,4096,7168) | 505.8 | 210.8 | 785.0 | 279.2 | 35.6 | 5706 |
| fc1 (dense) | (8192,36864,7168) | 2522.6 | 391.1 | 2919.1 | 396.5 | 13.6 | 5149 |
| fc1 (dense) | (16384,36864,7168) | 5000.4 | 748.5 | 5778.0 | 777.6 | 13.5 | 5195 |

### Observations

**MoE-shaped GEMMs pay ~2.7x the dense overhead — purely from shape.** ~36–38% vs
~13–14% for a full training pass. Same kernel, same code path; the only difference
is `N` (4096 vs 36864). The shorter GEMM cannot hide a cast whose cost scales with
bytes moved.

**wgrad is the worst GEMM everywhere** — 31–58%. It is the only one casting both
operands columnwise, and unlike fprop/dgrad it cannot benefit from weight
amortization, since neither of its operands is the weight.

**The cast runs well below memory-bandwidth roofline.** Quantization is pure
streaming: read bf16, write fp4 + fp8 scales = 2.5625 bytes/element. Against
B200's ~8 TB/s HBM the measured cast lands at ~2.2–2.8 TB/s, roughly 30% of peak,
and it stays near 30% across a 40x–1900 MB range. That flatness argues for a
genuine kernel inefficiency rather than fixed per-launch overhead — closing it
would cut the overhead figures by roughly 3x. Confirming that properly wants `ncu`
on the cast kernel rather than wall-clock arithmetic.

**Raising M does not help.** Dense holds at 13.6% -> 13.5% from M=8192 to 16384.
Doubling M doubles GEMM and cast alike, so the ratio is scale-invariant.

### Caveats

- **The `fc1 (MoE)` rows here are a shape, not TE's MoE path.** They are a plain
  dense `general_gemm` at one expert's dimensions, with `M` as tokens *per
  expert* — so `M=8192` implies a very large batch, since DeepSeek-V3 routes
  top-8 of 256 experts (roughly total/32 per expert). Part 2 measures the real
  path.
- **Run-to-run variance on the quant columns.** `gemm us` is reproducible to a few
  percent, but the quant side is a difference of two end-to-end measurements.
  Where that residual is small relative to the GEMM it moves several points
  between runs — the MoE step figure has been observed between 28% and 38% across
  runs, while dense stays within 13–15%. Raise `--repeats` when it matters, and do
  not read single-digit differences between small-shape rows as signal.
- Rows where `total <= gemm` are physically impossible; the script drops them with
  a warning rather than reporting a negative overhead.

---

# Part 2 — Grouped MoE

`benchmark_nvfp4_grouped_quant_overhead.py`. Drives the path TE actually uses for
MoE: `tex.group_quantize` into packed `GroupedTensor` storage, then
`general_grouped_gemm_for_grouped_tensor` across all local experts in one launch.
Weights are cast per expert (discrete list); activations and gradients are packed
and group-cast, matching `ops/fused/grouped_mlp.py`.

Same fc1 shape as Part 1, `E=8` local experts (DeepSeek-V3 with EP=32).

```bash
python benchmarks/linear/benchmark_nvfp4_grouped_quant_overhead.py \
    --experts 8 --tokens-per-expert 512,1024,2048 \
    --iters 100 --repeats 7 --amortize-weight --step-total
```

> **RHT is forced on here.** Grouped NVFP4 quantization has no non-RHT kernel —
> `group_quantize` raises *"graph safe grouped quant kernel for non-RHT path is
> not ready yet"* (`pytorch/csrc/extensions/cast.cpp`), and post-RHT amax is
> likewise required.

### Per-GEMM

| E | tok/E | (M,N,K) | gemm | gemm us | quant us | total us | overhead us | quant % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 512 | (4096,4096,7168) | fprop | 61.0 | 38.7 | 113.4 | 52.4 | 46.2 |
| 8 | 512 | (4096,4096,7168) | dgrad | 71.1 | 27.8 | 108.1 | 37.0 | 34.2 |
| 8 | 512 | (4096,4096,7168) | wgrad | 126.9 | 80.1 | 217.3 | 90.4 | 41.6 |
| 8 | 1024 | (8192,4096,7168) | fprop | 102.8 | 69.9 | 185.4 | 82.6 | 44.5 |
| 8 | 1024 | (8192,4096,7168) | dgrad | 113.8 | 42.8 | 175.4 | 61.6 | 35.1 |
| 8 | 1024 | (8192,4096,7168) | wgrad | 145.1 | 132.9 | 285.2 | 140.0 | 49.1 |
| 8 | 2048 | (16384,4096,7168) | fprop | 185.8 | 113.6 | 322.3 | 136.5 | 42.3 |
| 8 | 2048 | (16384,4096,7168) | dgrad | 196.5 | 84.1 | 293.4 | 96.9 | 33.0 |
| 8 | 2048 | (16384,4096,7168) | wgrad | 227.7 | 219.9 | 469.1 | 241.4 | 51.5 |

### One training pass

| E | tok/E | (M,N,K) | gemm us | quant us | total us | overhead us | quant % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 512 | (4096,4096,7168) | 268.9 | 92.6 | 363.5 | 94.6 | 26.0 | 2683 |
| 8 | 1024 | (8192,4096,7168) | 384.1 | 153.8 | 564.7 | 180.6 | 32.0 | 3757 |
| 8 | 2048 | (16384,4096,7168) | 635.8 | 258.6 | 945.5 | 309.6 | 32.7 | 4539 |

### Observations

**The grouped GEMM is what degrades at low occupancy, not the cast.** At 512
tokens/expert the grouped GEMM manages 2683 TFLOP/s versus 4539 at 2048 — the
whole `M=4096` problem is spread across 8 separate expert GEMMs, so each is small.
`quant %` is *lowest* at 512 tok/E precisely because the GEMM is inefficient
there, which inverts the shape effect seen in Part 1. Read the absolute
`overhead us` rather than the percentage when comparing across `tok/E`.

**wgrad is again the worst per-GEMM**, 42–52%, for the same reason as Part 1.

### Caveats

- **`quant us` includes allocation.** `tex.group_quantize` allocates its output on
  every call. Its `output=` reuse path requires uniform shapes
  (`first_dims=None`), and that path currently dies with an illegal memory access
  in `group_row_col_rht_gemm_ntt_w_sfc_graph_safe`
  (`common/hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu`),
  reproducible with NVFP4 + RHT on a single call in a fresh process. Part 1
  preallocates and reuses via `update_quantized`, so its `quant us` does not carry
  this. These grouped figures therefore overstate pure kernel time.
- Grouped GEMM requires cuBLAS 13.3+; older runtimes raise from
  `check_grouped_gemm_requirements`.
- Token counts are uniform across experts. Real routing is imbalanced, and the
  grouped GEMM's efficiency depends on the distribution.

---

## Comparing the two

Grouped lands at **26–33%** for a training pass against dense's **36–38%** at the
same total row count, so the dense-shape approximation is a reasonable proxy for
this configuration rather than the loose lower bound it might appear to be.

That comparison is confounded and should not be read as a measurement of the
grouped path's cost on its own:

- **Different recipes.** Part 1 runs RHT off, Part 2 cannot. Any gap mixes the
  grouped path with the cost of the RHT cast itself, and these scripts cannot
  separate the two. Running Part 1 with RHT on would isolate it.
- **Different allocation behaviour** in the `quant` column, as noted above.
- **Different work decomposition.** One dense `M=8192` GEMM is not the same
  problem as eight `M=1024` expert GEMMs, even at equal FLOPs.

All numbers are specific to this GPU, TE build and cuBLAS version.
