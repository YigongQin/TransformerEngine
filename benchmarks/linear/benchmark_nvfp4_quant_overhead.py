# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Measure NVFP4 quantization cost as a fraction of GEMM time, per training GEMM.

Answers "how much of my quantized linear layer is quantization, not math?" using
plain 1D NVFP4 -- no RHT, no stochastic rounding, no 2D scaling -- and reports the
three training GEMMs separately rather than rolled into one ``te.Linear`` number.

Method
------
For ``Y = X @ W^T`` with ``X:(M,K)``, ``W:(N,K)``, ``dY:(M,N)``, Transformer
Engine issues three GEMMs (see ``transformer_engine/pytorch/module/linear.py``).
Each operand is consumed in exactly one direction, and NVFP4 blocks are 1D along
the contraction axis, so the required quantizer usage is fixed per GEMM:

    gemm    call                    A (usage)      B (usage)      layout  out
    ------  ----------------------  -------------  -------------  ------  ------
    fprop   general_gemm(W,  X)     W  rowwise     X  rowwise     TN      (M,N)
    dgrad   general_gemm(W,  dY)    W  columnwise  dY rowwise     NN      (M,K)
    wgrad   general_gemm(X,  dY)    X  columnwise  dY columnwise  NT      (N,K)

Three loops are timed per GEMM, all sharing one set of quantized operands:

  * ``gemm``       -- baseline: pre-quantized operands into ``general_gemm``.
  * ``quant``      -- the cast kernels alone.
  * ``quant+gemm`` -- cast, then that same GEMM.

``overhead = (quant+gemm) - gemm``, and ``quant_pct`` is that overhead as a
fraction of the full path. Because both loops use identical operands and the
GEMM is byte-identical between them, the difference isolates the cast.

By default the scale-factor swizzle is fused into the cast (``optimize_for_gemm``
on the quantizer), so no standalone swizzle pass exists anywhere in the timed
path: the cast emits GEMM-swizzled scales and ``swizzle_scales_for_gemm``
early-returns inside the GEMM. Pass ``--no-fused-swizzle`` to instead match what
``te.Linear`` does today, where every ``general_gemm`` call re-swizzles.

``--step-total`` additionally measures one full training pass: all three GEMMs
with X, W and dY each quantized *once* using a fused rowwise+columnwise cast.
This is measured directly rather than summed from the per-GEMM rows, because X
and dY are each consumed by two GEMMs -- summing would charge them twice, through
two separate single-usage casts instead of one dual-usage cast, and overstate
quantization substantially.

Usage
-----
::

    # all DeepSeek-V3 linear shapes, M in {8192, 16384}, all 3 GEMMs
    python benchmarks/linear/benchmark_nvfp4_quant_overhead.py

    # FFN only, weight amortized across microbatches, plus the per-step total
    python benchmarks/linear/benchmark_nvfp4_quant_overhead.py \\
        --layers ffn --amortize-weight --step-total

    # one shape, tensor-parallel sharded, results to CSV
    python benchmarks/linear/benchmark_nvfp4_quant_overhead.py \\
        --layers fc1 -m 8192 --tp 8 -o fc1.csv

    # match te.Linear's current unfused-swizzle behaviour
    python benchmarks/linear/benchmark_nvfp4_quant_overhead.py --no-fused-swizzle

    # profile a single shape
    nsys profile --trace=cuda,nvtx,cublas -o nvfp4_quant_overhead \\
        python benchmarks/linear/benchmark_nvfp4_quant_overhead.py \\
        --layers fc1 -m 8192 --iters 20

Interpreting the output
-----------------------
``gemm_us`` is reproducible to a few percent. The quant-side columns are a
difference of two end-to-end measurements, so on small shapes -- where the
residual is a small fraction of the GEMM -- they can swing by several points
between runs; raise ``--repeats`` when that matters. Rows where
``total <= gemm`` are physically impossible and are dropped with a warning
rather than reported.

Requires Blackwell (SM100+) for NVFP4.
"""

import argparse

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

import transformer_engine.pytorch as te  # noqa: F401  must be first per te-python-import-order
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
# DeepSeek-V3 linear layers as (K, N) at TP=1, plus the axis tensor parallelism
# shards. Config: hidden 7168, dense FFN 18432, MoE expert 2048, 128 heads,
# q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v 128.
DEEPSEEK_V3_LAYERS = {
    #                  K,     N,   shard axis
    "q_a_proj": (7168, 1536, None),
    "q_b_proj": (1536, 24576, "n"),  # 128 heads * (128 nope + 64 rope)
    "kv_a_proj": (7168, 576, None),  # 512 kv_lora + 64 rope
    "kv_b_proj": (512, 32768, "n"),  # 128 heads * (128 nope + 128 v)
    "o_proj": (16384, 7168, "k"),  # 128 heads * 128 v
    "mlp_gate_up": (7168, 36864, "n"),  # dense FFN, first 3 layers; 2 * 18432
    "mlp_down": (18432, 7168, "k"),
    "moe_gate_up": (7168, 4096, "n"),  # per expert; 2 * 2048
    "moe_down": (2048, 7168, "k"),
}

# Convenience names. DeepSeek-V3 has two FFN variants: MoE experts (58 of 61
# layers) and a dense MLP (first 3). "fc1"/"fc2" follow TE's LayerNormMLP naming
# and default to the MoE expert, which is what almost every layer uses.
LAYER_ALIASES = {
    "fc1": "moe_gate_up",
    "fc2": "moe_down",
    "dense_fc1": "mlp_gate_up",
    "dense_fc2": "mlp_down",
    "ffn": "moe_gate_up,moe_down",
    "dense_ffn": "mlp_gate_up,mlp_down",
}

LAYER_DISPLAY = {
    "moe_gate_up": "fc1 (MoE)",
    "moe_down": "fc2 (MoE)",
    "mlp_gate_up": "fc1 (dense)",
    "mlp_down": "fc2 (dense)",
}

# Per-GEMM general_gemm layout and accumulator mode. split_acc mirrors
# _2X_ACC_{FPROP,DGRAD,WGRAD} in transformer_engine/pytorch/module/base.py.
GEMM_SPECS = {
    "fprop": {"layout": "TN", "split_acc": False, "grad": False},
    "dgrad": {"layout": "NN", "split_acc": True, "grad": True},
    "wgrad": {"layout": "NT", "split_acc": True, "grad": True},
}

RESULT_COLUMNS = [
    "layer",
    "M",
    "K",
    "N",
    "gemm",
    "quant_us",
    "gemm_us",
    "total_us",
    "overhead_us",
    "quant_pct",
    "gemm_tflops",
    "eff_tflops",
]


def shard(layer, tp):
    """Return (K, N) for ``layer``, sharded across ``tp`` tensor-parallel ranks."""
    K, N, axis = DEEPSEEK_V3_LAYERS[layer]
    if tp > 1:
        if axis == "n":
            N //= tp
        elif axis == "k":
            K //= tp
    return K, N


def gemm_operands(gemm, M, K, N):
    """Return ((A_role, A_shape, A_usage), (B_role, B_shape, B_usage), out_shape)."""
    if gemm == "fprop":
        return ("weight", (N, K), "rowwise"), ("input", (M, K), "rowwise"), (M, N)
    if gemm == "dgrad":
        return ("weight", (N, K), "columnwise"), ("grad_output", (M, N), "rowwise"), (M, K)
    if gemm == "wgrad":
        return ("input", (M, K), "columnwise"), ("grad_output", (M, N), "columnwise"), (N, K)
    raise ValueError(f"unknown gemm '{gemm}'")


# Bytes written per element, per usage: quantized datum + its block scale.
# NVFP4 = 4-bit data with an fp8 scale per 16 elements; MXFP8 = 8-bit data with
# an E8M0 scale per 32. MXFP8 has no global amax, so its cast is a single pass.
RECIPES = {
    "nvfp4": {"write_bytes": 0.5 + 1 / 16, "has_amax": True},
    "mxfp8": {"write_bytes": 1.0 + 1 / 32, "has_amax": False},
}


def make_quantizer(usage, fused_swizzle=True, recipe="nvfp4"):
    """Build a plain 1D NVFP4 quantizer: no RHT, stochastic rounding, or 2D scaling.

    ``usage`` is "rowwise", "columnwise", or "both".

    ``fused_swizzle`` sets ``optimize_for_gemm``, making the cast kernel emit
    GEMM-swizzled scale factors directly so ``general_gemm`` skips its internal
    swizzle pass. Note this is not what ``te.Linear`` does today --
    ``optimize_for_gemm`` defaults to False and only the attention paths enable
    it -- so Linear still pays a separate swizzle inside every GEMM call.
    """
    rowwise = usage in ("rowwise", "both")
    columnwise = usage in ("columnwise", "both")
    if recipe == "mxfp8":
        quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=rowwise,
            columnwise=columnwise,
            with_2d_quantization=False,
        )
    else:
        quantizer = NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=rowwise,
            columnwise=columnwise,
            with_amax_reduction=False,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )
    quantizer.optimize_for_gemm = fused_swizzle
    return quantizer


def quantize_into(quantizer, tensor, shape, device, fused_swizzle):
    """Allocate a quantized destination and fill it, returning the destination.

    Destinations are preallocated and reused by the timed loops;
    ``quantizer.quantize()`` would allocate a fresh tensor per iteration, and
    that allocator churn is large enough to swamp the signal being measured.
    """
    dst = quantizer.make_empty(shape, dtype=torch.bfloat16, device=device, requires_grad=False)
    quantizer.update_quantized(tensor, dst)
    # pylint: disable=protected-access
    if dst._with_gemm_swizzled_scales != fused_swizzle:
        raise RuntimeError(
            "Quantized tensor swizzle state does not match the requested mode"
            f" (optimize_for_gemm={fused_swizzle}). With fused swizzle the cast must"
            " emit GEMM-swizzled scales, otherwise general_gemm silently adds a"
            " swizzle pass to the GEMM loop alone and the measured difference no"
            " longer isolates the cast. Try --no-fused-swizzle."
        )
    return dst


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_us(fn, iters, warmup, repeats, lead_fn=None):
    """Microseconds per call: min over ``repeats`` timed loops of ``iters`` calls.

    The SM clock idles far below its boost ceiling, so a short burst measures the
    ramp rather than steady state. Warm up long enough to reach boost, then take
    the min across repeats -- min is the robust estimator here, since any
    transient (clock dip, allocator growth, background work) only inflates a
    sample.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best = float("inf")
    for _ in range(repeats):
        # Enqueue a chunky kernel first so CPU dispatch for the first timed
        # iteration hides behind GPU work already in flight. Events are recorded
        # in stream order, so this kernel's own duration is excluded.
        if lead_fn is not None:
            lead_fn()

        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / iters)
    return best


def quant_breakdown(run_quant, iters=20):
    """Split one cast into amax vs cast GPU time, in us per call.

    An NVFP4 cast is a full-tensor amax pass followed by a cast pass. Everything
    that is not amax (the cast kernel, plus any scale-swizzle or copy the path
    happens to emit) is bucketed as ``cast``, so the two columns sum to the
    measured ``quant us``.
    """
    for _ in range(5):
        run_quant()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run_quant()
        torch.cuda.synchronize()

    totals = {"amax": 0.0, "cast": 0.0}
    for evt in prof.key_averages():
        if evt.device_type.name != "CUDA" or evt.self_device_time_total <= 0:
            continue
        phase = "amax" if "amax" in evt.key.lower() else "cast"
        totals[phase] += evt.self_device_time_total
    return {k: v / iters for k, v in totals.items()}


def add_cast_bandwidth(run_quant, row, parts, breakdown, recipe="nvfp4"):
    """Return the amax/cast split plus the cast pass's achieved bandwidth."""
    if not breakdown:
        return {}
    out = scaled_breakdown(run_quant, row["quant_us"])
    mb = cast_bytes(parts, recipe) / 1e6
    cast_us = out.get("cast_us", 0.0)
    out["cast_mb"] = mb
    out["cast_gbps"] = mb / 1e3 / (cast_us * 1e-6) if cast_us > 0 else 0.0
    return out


def scaled_breakdown(run_quant, quant_us):
    """Phase split rescaled so the parts sum to the measured ``quant_us``.

    torch.profiler adds per-kernel overhead, so raw profiled times run ~10-15%
    high and would otherwise exceed the wall-clock total they decompose. The
    *proportions* are what the profiler measures reliably; this reports those
    proportions against the unprofiled total.
    """
    raw = quant_breakdown(run_quant)
    total = sum(raw.values())
    if total <= 0:
        return {f"{k}_us": 0.0 for k in raw}
    return {f"{k}_us": v / total * quant_us for k, v in raw.items()}


# Cast-phase traffic: the cast reads each tensor once and writes one layout per
# usage. The amax pass is a separate read and is excluded -- this is the
# bandwidth of the cast kernel itself.
CAST_READ_BYTES = 2.0  # bf16 input


def cast_bytes(parts, recipe):
    """Bytes moved by the cast pass. ``parts`` is [(elems, n_usages), ...]."""
    write = RECIPES[recipe]["write_bytes"]
    return sum(e * (CAST_READ_BYTES + n * write) for e, n in parts)


def summarize(label, M, K, N, quant_us, gemm_us, total_us, num_gemms=1):
    """Assemble one result row from the three timed loops."""
    overhead_us = total_us - gemm_us
    flops = 2 * M * N * K * num_gemms
    return {
        "gemm": label,
        "M": M,
        "K": K,
        "N": N,
        "quant_us": quant_us,
        "gemm_us": gemm_us,
        "total_us": total_us,
        "overhead_us": overhead_us,
        "quant_pct": 100.0 * overhead_us / total_us if total_us > 0 else float("nan"),
        "gemm_tflops": flops / (gemm_us * 1e-6) / 1e12 if gemm_us > 0 else 0.0,
        "eff_tflops": flops / (total_us * 1e-6) / 1e12 if total_us > 0 else 0.0,
    }


def benchmark_one(gemm, M, K, N, args, lead_fn):
    """Time gemm / quant / quant+gemm for a single training GEMM."""
    device = "cuda"
    (a_role, a_shape, a_usage), (b_role, b_shape, b_usage), out_shape = gemm_operands(gemm, M, K, N)
    spec = GEMM_SPECS[gemm]

    a_hp = torch.randn(a_shape, dtype=torch.bfloat16, device=device)
    b_hp = torch.randn(b_shape, dtype=torch.bfloat16, device=device)
    a_quantizer = make_quantizer(a_usage, args.fused_swizzle, args.recipe)
    b_quantizer = make_quantizer(b_usage, args.fused_swizzle, args.recipe)
    a_q = quantize_into(a_quantizer, a_hp, a_shape, device, args.fused_swizzle)
    b_q = quantize_into(b_quantizer, b_hp, b_shape, device, args.fused_swizzle)

    # A weight quantized once per optimizer step (is_first_microbatch) costs
    # nothing per microbatch, so drop it from the timed work when amortizing.
    quant_a = not (args.amortize_weight and a_role == "weight")
    quant_b = not (args.amortize_weight and b_role == "weight")

    out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)

    def run_gemm():
        general_gemm(
            a_q,
            b_q,
            out_dtype=torch.bfloat16,
            layout=spec["layout"],
            out=out,
            use_split_accumulator=spec["split_acc"],
            grad=spec["grad"],
        )

    def run_quant():
        if quant_a:
            a_quantizer.update_quantized(a_hp, a_q)
        if quant_b:
            b_quantizer.update_quantized(b_hp, b_q)

    def run_full():
        run_quant()
        run_gemm()

    def timed(fn):
        return time_us(fn, args.iters, args.warmup, args.repeats, lead_fn)

    row = summarize(gemm, M, K, N, timed(run_quant), timed(run_gemm), timed(run_full))
    parts = []
    if quant_a:
        parts.append((a_shape[0] * a_shape[1], 1))
    if quant_b:
        parts.append((b_shape[0] * b_shape[1], 1))
    row.update(add_cast_bandwidth(run_quant, row, parts, args.breakdown, args.recipe))
    return row


def benchmark_step(M, K, N, args, lead_fn):
    """Time one full training pass: all 3 GEMMs, each tensor quantized once.

    Not the sum of the three :func:`benchmark_one` rows. X, W and dY are each
    consumed by two GEMMs in opposite directions, so a real step quantizes each
    exactly once with rowwise+columnwise usage -- a single fused cast that reads
    the input once and writes both layouts.
    """
    device = "cuda"
    fused = args.fused_swizzle

    x_hp = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    w_hp = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    dy_hp = torch.randn((M, N), dtype=torch.bfloat16, device=device)

    xq, wq, dyq = (make_quantizer("both", fused, args.recipe) for _ in range(3))
    x_q = quantize_into(xq, x_hp, (M, K), device, fused)
    w_q = quantize_into(wq, w_hp, (N, K), device, fused)
    dy_q = quantize_into(dyq, dy_hp, (M, N), device, fused)

    y = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    dx = torch.empty((M, K), dtype=torch.bfloat16, device=device)
    dw = torch.empty((N, K), dtype=torch.bfloat16, device=device)

    def gemm(a, b, name, out):
        spec = GEMM_SPECS[name]
        general_gemm(
            a,
            b,
            out_dtype=torch.bfloat16,
            layout=spec["layout"],
            out=out,
            use_split_accumulator=spec["split_acc"],
            grad=spec["grad"],
        )

    def run_gemms():
        gemm(w_q, x_q, "fprop", y)
        gemm(w_q, dy_q, "dgrad", dx)
        gemm(x_q, dy_q, "wgrad", dw)

    def run_quant():
        # The weight is quantized once per optimizer step, so with gradient
        # accumulation its cost vanishes per microbatch; X and dY never do.
        if not args.amortize_weight:
            wq.update_quantized(w_hp, w_q)
        xq.update_quantized(x_hp, x_q)
        dyq.update_quantized(dy_hp, dy_q)

    def run_full():
        run_quant()
        run_gemms()

    def timed(fn):
        return time_us(fn, args.iters, args.warmup, args.repeats, lead_fn)

    row = summarize(
        "step(3 gemms)",
        M,
        K,
        N,
        timed(run_quant),
        timed(run_gemms),
        timed(run_full),
        num_gemms=3,
    )
    parts = [(M * K, 2), (M * N, 2)]
    if not args.amortize_weight:
        parts.append((N * K, 2))
    row.update(add_cast_bandwidth(run_quant, row, parts, args.breakdown, args.recipe))
    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(frame, title, per_gemm=True):
    """Print a results frame as layer / (M,N,K) / timings.

    ``per_gemm`` tables break out fprop/dgrad/wgrad; the step table has a single
    constant label, so it drops that column and shows GEMM throughput instead.
    """
    table = frame.copy()
    table["layer"] = table.layer.map(lambda name: LAYER_DISPLAY.get(name, name))
    table["(M,N,K)"] = [f"({m},{n},{k})" for m, n, k in zip(table.M, table.N, table.K)]

    columns = ["layer", "(M,N,K)"]
    if per_gemm:
        columns.append("gemm")
    columns += ["gemm_us", "quant_us"]
    if "amax_us" in table:
        # MXFP8 has no global amax pass, so the column would be all zeros.
        if table["amax_us"].abs().max() > 0:
            columns.append("amax_us")
        columns += ["cast_us", "cast_gbps"]
    columns += ["total_us", "overhead_us", "quant_pct"]
    if not per_gemm:
        columns.append("gemm_tflops")

    table = table[columns].rename(
        columns={
            "gemm_us": "gemm us",
            "quant_us": "quant us",
            "amax_us": "amax us",
            "cast_us": "cast us",
            "cast_gbps": "cast GB/s",
            "total_us": "total us",
            "overhead_us": "overhead us",
            "quant_pct": "amax+cast %",
            "gemm_tflops": "GEMM TFLOP/s",
        }
    )
    print(f"\n=== {title} ===")
    print(table.round(1).to_string(index=False))


def collect(configs, measure, describe):
    """Run ``measure`` over ``configs``, dropping and reporting impossible rows."""
    rows = []
    for layer, M, extra in configs:
        # Release the previous config's buffers. Without this the caching
        # allocator carries fragmentation across configs and later shapes
        # measure slower than they do in isolation.
        torch.cuda.empty_cache()
        try:
            row = measure(layer, M, extra)
        except Exception as exc:  # noqa: BLE001  a bad shape shouldn't kill the sweep
            print(f"  skip {describe(layer, M, extra)}: {exc}")
            continue
        if row["overhead_us"] <= 0:
            # quant+gemm came out no slower than gemm alone, which is physically
            # impossible -- the row is noise, not a result.
            print(f"  WARNING {describe(layer, M, extra)}: total <= gemm, discarding row")
            continue
        row["layer"] = layer
        rows.append(row)
        print(
            f"  {describe(layer, M, extra):<34}"
            f" quant={row['quant_us']:8.1f}us gemm={row['gemm_us']:8.1f}us"
            f" total={row['total_us']:8.1f}us  quant={row['quant_pct']:5.1f}%"
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--token-dims",
        type=str,
        default="8192,16384",
        help="Comma-separated M values (tokens per GEMM).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help=(
            f"Comma-separated subset of {sorted(DEEPSEEK_V3_LAYERS)}, or 'all', or an"
            f" alias from {sorted(LAYER_ALIASES)}."
        ),
    )
    parser.add_argument(
        "--gemms",
        type=str,
        default="fprop,dgrad,wgrad",
        help="Comma-separated subset of fprop,dgrad,wgrad.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel size for sharding K/N.")
    parser.add_argument(
        "--amortize-weight",
        action="store_true",
        help=(
            "Exclude weight quantization from the timed work, modelling a weight"
            " quantized once per optimizer step and reused across microbatches."
        ),
    )
    parser.add_argument(
        "--step-total",
        action="store_true",
        help=(
            "Also measure one full training pass per (layer, M): all 3 GEMMs with"
            " X/W/dY each quantized once via a fused rowwise+columnwise cast."
            " Measured directly, not summed from the per-GEMM rows."
        ),
    )
    parser.add_argument(
        "--no-fused-swizzle",
        dest="fused_swizzle",
        action="store_false",
        help=(
            "Disable optimize_for_gemm, so the cast emits compact scales and every"
            " general_gemm call re-swizzles them, matching te.Linear today. The"
            " swizzle then sits inside both the gemm and total measurements."
        ),
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help=(
            "Split the quant column into its amax / cast / swizzle kernels via"
            " torch.profiler. An NVFP4 cast is a full-tensor amax pass followed by"
            " a cast pass, and amax alone is roughly half the traffic."
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default="nvfp4",
        help=(
            "Quantization recipe. nvfp4 is a two-pass cast (global amax, then cast);"
            " mxfp8 has per-block E8M0 scales and no global amax, so it is one pass."
        ),
    )
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per loop.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations per loop; long enough to reach boost clocks.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed loops per measurement; the min is kept."
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Write results to CSV.")
    return parser.parse_args()


def extra_cols(rows):
    """Breakdown columns, present only when --breakdown was passed."""
    return [c for c in ("amax_us", "cast_us", "cast_mb", "cast_gbps") if c in rows[0]]


def resolve_layers(spec):
    """Expand ``--layers`` (names, aliases, or 'all') into concrete layer names."""
    if spec == "all":
        return list(DEEPSEEK_V3_LAYERS)
    layers = []
    for name in (n.strip() for n in spec.split(",") if n.strip()):
        layers.extend(LAYER_ALIASES.get(name, name).split(","))
    for name in layers:
        if name not in DEEPSEEK_V3_LAYERS:
            raise SystemExit(
                f"unknown layer '{name}'; choose from {sorted(DEEPSEEK_V3_LAYERS)} or"
                f" an alias from {sorted(LAYER_ALIASES)}"
            )
    return layers


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise SystemExit("NVFP4 requires Blackwell (SM100+).")

    token_dims = [int(x) for x in args.token_dims.split(",") if x.strip()]
    gemms = [g.strip() for g in args.gemms.split(",") if g.strip()]
    layers = resolve_layers(args.layers)

    print(f"Device: {torch.cuda.get_device_name()} (SM{major}0)")
    if args.recipe == "mxfp8":
        print("Recipe: MXFP8 (E4M3 data, E8M0 scale per 32) -- single-pass cast, no amax")
    else:
        print(
            "Recipe: NVFP4 1D block scaling -- no RHT, no stochastic rounding, no 2D quantization"
        )
    print(
        f"TP={args.tp}, amortize_weight={args.amortize_weight},"
        f" fused_swizzle={args.fused_swizzle}, warmup={args.warmup},"
        f" iters={args.iters}, repeats={args.repeats} (min)\n"
    )

    # Leading kernel, shared across every measurement so the dispatch-hiding
    # trick costs the same everywhere.
    lead_a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    lead_b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

    def lead_fn():
        torch.matmul(lead_a, lead_b)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", None)

    per_gemm_configs = [(layer, M, g) for layer in layers for M in token_dims for g in gemms]
    rows = collect(
        per_gemm_configs,
        lambda layer, M, g: benchmark_one(g, M, *shard(layer, args.tp), args, lead_fn),
        lambda layer, M, g: f"{layer} M={M} {g}",
    )
    if not rows:
        raise SystemExit("No successful measurements.")

    frame = pd.DataFrame(rows)[RESULT_COLUMNS + extra_cols(rows)]
    report(frame, "Per-GEMM")

    if args.step_total:
        step_configs = [(layer, M, None) for layer in layers for M in token_dims]
        step_rows = collect(
            step_configs,
            lambda layer, M, _: benchmark_step(M, *shard(layer, args.tp), args, lead_fn),
            lambda layer, M, _: f"{layer} M={M} step",
        )
        if step_rows:
            step_frame = pd.DataFrame(step_rows)[RESULT_COLUMNS + extra_cols(step_rows)]
            report(step_frame, "One training pass: 3 GEMMs, each tensor quantized once", False)
            if args.output:
                step_out = args.output.replace(".csv", "_step.csv")
                step_frame.to_csv(step_out, index=False)
                print(f"\nWrote {step_out}")

    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
