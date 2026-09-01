# NVFP4 quantization overhead in a linear layer

How much of a quantized linear layer is quantization rather than math?

`benchmark_nvfp4_quant_overhead.py` measures the NVFP4 cast cost as a fraction of
cast + GEMM, reporting the three training GEMMs (fprop, dgrad, wgrad) separately
instead of rolling them into one `te.Linear` number. Recipe is plain 1D NVFP4 —
no RHT, no stochastic rounding, no 2D scaling.

## Quick start

```bash
# every DeepSeek-V3 linear shape, M in {8192, 16384}
python benchmarks/linear/benchmark_nvfp4_quant_overhead.py

# the numbers below
python benchmarks/linear/benchmark_nvfp4_quant_overhead.py \
    --layers fc1,dense_fc1 --iters 100 --repeats 7 --amortize-weight --step-total
```

Requires Blackwell (SM100+).

## Method

Three loops are timed per GEMM, all sharing one set of quantized operands:

| loop | what it runs |
| --- | --- |
| `gemm` | pre-quantized operands into `general_gemm` — the raw NVFP4 GEMM |
| `quant` | the cast kernels alone |
| `quant+gemm` | cast, then that same GEMM |

`overhead = (quant+gemm) − gemm`, and `quant %` is that overhead as a fraction of
the full path. Both loops use identical operands and a byte-identical GEMM, so
the difference isolates the cast.

The scale-factor swizzle is fused into the cast by default (`optimize_for_gemm`),
so no standalone swizzle pass exists in the timed path. Note this is *not* what
`te.Linear` does today — `optimize_for_gemm` defaults to `False` and only the
attention paths enable it, so Linear still re-swizzles inside every GEMM call.
`--no-fused-swizzle` reproduces that behaviour; measured directly, the swizzle is
only ~8–13 µs on these shapes.

`--amortize-weight` models gradient accumulation: the weight is quantized once
per optimizer step and reused across microbatches, so its cast cost is excluded.
Without it, the weight is re-quantized every step (microbatch = 1).

See the module docstring for the per-GEMM operand/layout table.

## Results

`NVIDIA B200` (148 SMs, 178 GiB, cc 10.0), TE `2.20.0.dev0+1ff9c372`,
torch `2.11.0+cu130`, cuDNN 9.19, CUDA 13.2. Weight amortized, fused swizzle,
`--iters 100 --repeats 7`. Times in µs.

`fc1 (MoE)` is a DeepSeek-V3 MoE expert (58 of 61 layers); `fc1 (dense)` is the
dense MLP used in the first 3 layers.

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
| fc1 (MoE) | (8192,4096,7168) | fprop | 84.4 | 54.5 | 179.6 | 95.2 | 53.0 |
| fc1 (MoE) | (8192,4096,7168) | dgrad | 91.9 | 33.6 | 138.8 | 46.9 | 33.8 |
| fc1 (MoE) | (8192,4096,7168) | wgrad | 86.6 | 116.6 | 205.0 | 118.4 | 57.8 |
| fc1 (MoE) | (16384,4096,7168) | fprop | 164.3 | 96.3 | 288.4 | 124.2 | 43.0 |
| fc1 (MoE) | (16384,4096,7168) | dgrad | 172.9 | 68.2 | 259.6 | 86.7 | 33.4 |
| fc1 (MoE) | (16384,4096,7168) | wgrad | 166.0 | 198.7 | 381.4 | 215.4 | 56.5 |
| fc1 (dense) | (8192,36864,7168) | fprop | 803.4 | 56.2 | 870.5 | 67.1 | 7.7 |
| fc1 (dense) | (8192,36864,7168) | dgrad | 855.1 | 233.6 | 1067.9 | 212.8 | 19.9 |
| fc1 (dense) | (8192,36864,7168) | wgrad | 789.0 | 354.7 | 1176.0 | 387.0 | 32.9 |
| fc1 (dense) | (16384,36864,7168) | fprop | 1611.9 | 95.5 | 1744.9 | 133.1 | 7.6 |
| fc1 (dense) | (16384,36864,7168) | dgrad | 1773.2 | 456.2 | 2230.0 | 456.8 | 20.5 |
| fc1 (dense) | (16384,36864,7168) | wgrad | 1620.6 | 687.0 | 2341.2 | 720.6 | 30.8 |

### One training pass

All three GEMMs with X, W and dY each quantized **once** via a fused
rowwise+columnwise cast. Measured directly, not summed from the rows above — X
and dY are each consumed by two GEMMs, so summing charges them twice through two
single-usage casts instead of one dual-usage cast, and overstates quantization
badly (42% vs 36.5% for MoE here).

| layer | (M,N,K) | gemm us | quant us | total us | overhead us | quant % | GEMM TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fc1 (MoE) | (8192,4096,7168) | 271.1 | 126.6 | 427.1 | 156.0 | 36.5 | 5322 |
| fc1 (MoE) | (16384,4096,7168) | 576.7 | 211.6 | 800.5 | 223.8 | 28.0 | 5004 |
| fc1 (dense) | (8192,36864,7168) | 2506.6 | 388.4 | 2905.2 | 398.6 | 13.7 | 5182 |
| fc1 (dense) | (16384,36864,7168) | 5050.8 | 746.0 | 5814.1 | 763.3 | 13.1 | 5143 |

## Observations

**MoE experts pay ~2× the dense overhead — purely from shape.** ~28–37% vs
~13–14% for a full training pass. Same kernel, same code path; the only
difference is `N` (4096 vs 36864). The MoE GEMM is too short to hide a cast whose
cost scales with bytes moved.

**wgrad is the worst GEMM everywhere** — 31–58%. It is the only one quantizing
both operands columnwise, and unlike fprop/dgrad it cannot benefit from weight
amortization, since neither of its operands is the weight.

**The cast runs well below memory-bandwidth roofline.** Quantization is pure
streaming: read bf16, write fp4 + fp8 scales = 2.5625 bytes/element. Against
B200's ~8 TB/s HBM the measured cast lands at ~2.2–2.8 TB/s, roughly 30% of peak,
and it stays near 30% across a 40×–1900 MB range. That flatness argues for a
genuine kernel inefficiency rather than fixed per-launch overhead — closing it
would cut the overhead figures by roughly 3×. Confirming that properly wants
`ncu` on the cast kernel rather than wall-clock arithmetic.

**Raising M does not help.** Dense holds at 13.7% → 13.1% from M=8192 to 16384.
Doubling M doubles GEMM and cast alike, so the ratio is roughly scale-invariant.

## Caveats

- **The MoE rows are shapes, not TE's MoE path.** They are a plain dense
  `general_gemm` at one expert's dimensions. Real MoE goes through
  `general_grouped_gemm_for_grouped_tensor` with grouped tensors and batched
  scale swizzling. `M` here is tokens *per expert*, so `M=8192` implies a very
  large batch — DeepSeek-V3 routes top-8 of 256 experts, giving roughly
  total/32 tokens per expert. At a realistic per-expert `M` the GEMM shrinks
  while the cast does not, so these figures are a **lower bound** on MoE
  overhead. See `benchmark_grouped_linear.py` for the grouped path.
- **Run-to-run variance on the quant columns.** `gemm us` is reproducible to a
  few percent, but the quant side is a difference of two end-to-end measurements.
  Where that residual is small relative to the GEMM it moves several points
  between runs — the MoE step figure has been observed between 28% and 37% across
  runs, while dense stays within 13–15%. Raise `--repeats` when it matters, and
  do not read single-digit differences between small-shape rows as signal.
- Rows where `total <= gemm` are physically impossible; the script drops them
  with a warning rather than reporting a negative overhead.
- Numbers are specific to this GPU, TE build, and cuBLAS version. cuBLAS
  upgrades in particular move `gemm us`, and therefore every percentage.
