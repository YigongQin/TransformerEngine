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

`overhead = (quant+gemm) − gemm`, and `amax+cast %` is that overhead as a fraction of
the full path. Both loops use identical operands and a byte-identical GEMM, so
the difference isolates the cast.

`--amortize-weight` models gradient accumulation: the weight is cast once per
optimizer step and reused across microbatches, so its cost is excluded. Without
it, the weight is re-cast every step (microbatch = 1).

`--step-total` measures one full training pass — all three GEMMs with X, W and dY
each cast **once**. Measured directly, never summed from the per-GEMM rows.

The reason summing is wrong is the **fused cast + cast-transpose**: one kernel
reads a tensor once and writes both the rowwise and columnwise layouts. The
per-GEMM bench needs two quantized operands per GEMM, so across three GEMMs it
performs **6 single-usage casts**; a training pass performs **3 fused dual-usage
casts** over X, W and dY. With the weight amortized that is 4 casts versus 2.

The saving is entirely in *reads*, not writes — both produce the same two output
layouts. For the grouped fc1 at 512 tok/E:

| | casts | read | write | total traffic |
| --- | --- | --- | --- | --- |
| per-GEMM (summed) | 4 single-usage | 184.5 MB | 51.9 MB | 236.5 MB |
| one training pass | 2 fused dual-usage | 92.3 MB | 51.9 MB | 144.2 MB |

That predicts the step doing 39% less cast traffic; measured `quant us` is 37%
lower (92.6 vs 146.6 µs). The agreement is close enough to conclude the cast is
bandwidth-bound and that this gap is real physics rather than a measurement
artifact — unlike the penalty described next.

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
against a measured step of 94.6 µs. That 85 µs gap splits cleanly: **54 µs is the
redundant input reads** described above (real, and correctly absent from a
training pass) and **~33 µs is this fixed penalty** counted three times (a
measurement artifact).

**Read the per-GEMM tables for the relative ordering of fprop/dgrad/wgrad, and the
step tables for the actual quantization fraction.**

### The `amax us` / `cast us` / `cast GB/s` columns

An NVFP4 cast is a **two-pass** operation, so `quant us` is split into its parts
(`--breakdown`, via `torch.profiler`). Everything that is not amax is bucketed as
`cast`, so the two columns sum to `quant us`:

| config | amax pass | cast pass | separate swizzle? |
| --- | --- | --- | --- |
| rowwise, RHT off | 25.1 µs | 22.9 µs | yes, 6.0 µs |
| columnwise, RHT off | 24.9 µs | 39.7 µs | yes, 6.5 µs |
| both, RHT off | 26.4 µs | 36.4 µs | yes, 11.8 µs |
| both, RHT on (Part 2) | 27.0 µs | 44.6 µs | no — fused into the cast |

`torch.profiler` adds ~13% per-kernel overhead, so the split is reported as
*proportions* rescaled to the unprofiled `quant us`.

**`cast GB/s`** is the cast pass alone — the amax pass is excluded, since it is a
separate read that produces no output. The cast reads each tensor once and writes
one layout per usage:

    bytes = elems * (2 + n_usages * 0.5625)     # read bf16; write fp4 + fp8 scale

so 2.5625 B/elem single-usage and 3.125 fused dual-usage. Amortized rows exclude
the weight. Reference: a trivial bf16 copy sustains **6490 GB/s** on this GPU
(measured), which is what these should be judged against rather than the 8 TB/s
spec figure.

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

`optimize_for_gemm` is on by default. On this RHT-off path it does **not** fuse
the swizzle into the cast — it *relocates* it: profiling one `update_quantized`
shows 3 kernels and no swizzle with the flag off, versus 4 kernels including a
separate 5.5 µs `swizzle_row_scaling_kernel` with it on. The swizzle moves out of
the GEMM and into the cast call, which is why enabling it measures as roughly
net-neutral. It is genuinely fused only on the RHT path (Part 2).

This is also *not* what `te.Linear` does today — `optimize_for_gemm` defaults to
`False` and only the attention paths enable it, so Linear still re-swizzles inside
every GEMM call. `--no-fused-swizzle` reproduces that.

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

| layer | (M,N,K) | gemm | gemm us | quant us | amax us | cast us | cast GB/s | total us | overhead us | amax+cast % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | fprop | 84.8 | 54.5 | 25.3 | 29.1 | 5163 | 162.4 | 77.7 | 47.8 |
| fc1 (MoE) | (8192,4096,7168) | dgrad | 89.4 | 30.2 | 13.7 | 16.5 | 5200 | 135.4 | 46.1 | 34.0 |
| fc1 (MoE) | (8192,4096,7168) | wgrad | 85.9 | 115.8 | 42.4 | 73.4 | 3221 | 206.0 | 120.2 | 58.3 |
| fc1 (MoE) | (16384,4096,7168) | fprop | 166.9 | 96.5 | 46.3 | 50.3 | 5986 | 287.1 | 120.2 | 41.9 |
| fc1 (MoE) | (16384,4096,7168) | dgrad | 174.3 | 69.9 | 33.2 | 36.7 | 4686 | 259.0 | 84.8 | 32.7 |
| fc1 (MoE) | (16384,4096,7168) | wgrad | 169.4 | 199.9 | 71.2 | 128.7 | 3676 | 396.5 | 227.1 | 57.3 |
| fc1 (dense) | (8192,36864,7168) | fprop | 853.6 | 56.2 | 25.8 | 30.4 | 4942 | 913.7 | 60.1 | 6.6 |
| fc1 (dense) | (8192,36864,7168) | dgrad | 867.1 | 234.8 | 114.6 | 120.3 | 6434 | 1126.1 | 258.9 | 23.0 |
| fc1 (dense) | (8192,36864,7168) | wgrad | 825.3 | 356.7 | 122.8 | 233.9 | 3952 | 1200.5 | 375.2 | 31.3 |
| fc1 (dense) | (16384,36864,7168) | fprop | 1742.4 | 100.0 | 48.6 | 51.4 | 5859 | 1838.6 | 96.2 | 5.2 |
| fc1 (dense) | (16384,36864,7168) | dgrad | 1798.1 | 456.1 | 218.9 | 237.2 | 6525 | 2269.1 | 470.9 | 20.8 |
| fc1 (dense) | (16384,36864,7168) | wgrad | 1693.3 | 689.8 | 231.3 | 458.5 | 4032 | 2365.0 | 671.6 | 28.4 |

### One training pass

| layer | (M,N,K) | gemm us | quant us | amax us | cast us | cast GB/s | total us | overhead us | amax+cast % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | 276.8 | 128.3 | 43.0 | 85.3 | 3379 | 426.4 | 149.7 | 35.1 | 5214 |
| fc1 (MoE) | (16384,4096,7168) | 579.0 | 212.3 | 73.7 | 138.6 | 4161 | 789.4 | 210.3 | 26.6 | 4985 |
| fc1 (dense) | (8192,36864,7168) | 2665.1 | 391.2 | 129.5 | 261.7 | 4308 | 3035.3 | 370.1 | 12.2 | 4873 |
| fc1 (dense) | (16384,36864,7168) | 5295.2 | 754.1 | 247.0 | 507.1 | 4446 | 6097.9 | 802.6 | 13.2 | 4906 |

### Observations

**MoE-shaped GEMMs pay ~2.7x the dense overhead — purely from shape.** ~36–38% vs
~13–14% for a full training pass. Same kernel, same code path; the only difference
is `N` (4096 vs 36864). The shorter GEMM cannot hide a cast whose cost scales with
bytes moved.

**wgrad is the worst GEMM everywhere** — 31–58%. It is the only one casting both
operands columnwise, and unlike fprop/dgrad it cannot benefit from weight
amortization, since neither of its operands is the weight.

**The cast kernel is near bandwidth-saturated; the cost is the extra pass.**
`cast GB/s` runs 3221–6525 against a 6490 GB/s copy rate — 50–100%. There is no
slow kernel to fix.

**The amax pass is the lever.** It is a separate full read of the tensor that
buys no output, and `amax us` is **33–49% of `quant us`** in every row — for the
dense step rows, 43.0 of 128.3 µs and 247.0 of 754.1 µs.
Eliminating it (fusing amax into whatever produces the tensor, or a
delayed/cached amax as FP8 recipes use) would cut quantization cost far more than
any tuning of the cast itself.

wgrad is weakest at 56–74%, matching the standalone columnwise cast at 58% —
strided writes cannot reach copy bandwidth. fprop/dgrad on large dense shapes
reach 86–93%.

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

### One training pass

Only the fused dual-usage (`both`) cast is reported here: it is the one a real
step performs, and it is the configuration where the scale swizzle is genuinely
fused into the cast kernel rather than run as a separate pass.

| E | tok/E | (M,N,K) | gemm us | quant us | amax us | cast us | cast GB/s | total us | overhead us | amax+cast % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 512 | (4096,4096,7168) | 269.4 | 95.5 | 28.4 | 67.1 | 2149 | 365.7 | 96.3 | 26.3 | 2679 |
| 8 | 1024 | (8192,4096,7168) | 371.4 | 156.0 | 45.9 | 110.1 | 2618 | 567.4 | 196.0 | 34.5 | 3886 |
| 8 | 2048 | (16384,4096,7168) | 681.3 | 259.3 | 72.8 | 186.5 | 3092 | 958.8 | 277.5 | 28.9 | 4237 |

`cast us` here also absorbs the ~8 µs of per-call allocation and device-to-device
copy that `tex.group_quantize` incurs (see caveats), so `cast GB/s` on this path
is understated by a few percent.

### Observations

**The swizzle really is fused here.** Profiling `update_quantized` on the RHT
path gives 3 kernels with `optimize_for_gemm` either on or off (72.1 vs 71.6 µs),
and the cast kernel itself carries the swizzle. Contrast Part 1, where the flag
adds a separate 5.5 µs swizzle kernel.

**amax is ~28–30% of `quant us`**, 28–73 µs — the same story as Part 1: a
full-tensor read that produces no output.

**Cast bandwidth is well below dense** — 2149–3092 GB/s (33–48% of the 6490 GB/s
copy rate) versus 3221–6525 (50–100%) in Part 1, and it climbs steadily with
tokens per expert. Part of that is the per-call allocation folded into `cast us`;
the rest is the grouped cast being occupancy-limited at small per-expert counts.

**`amax+cast %` is roughly flat at 26–35%** across a 4x range of tokens per
expert, because both the GEMM and the cast scale with token count. GEMM efficiency
does climb sharply (2679 -> 4237 TFLOP/s) as experts get more tokens, but the cast
improves alongside it.

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
