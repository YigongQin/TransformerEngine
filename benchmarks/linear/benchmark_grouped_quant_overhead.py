# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped-quantize + grouped-GEMM overhead for an MoE expert FC1.

The MoE counterpart to ``benchmark_quant_overhead.py``; ``--recipe`` selects
NVFP4 or MXFP8. That script issues a
dense ``general_gemm`` at one expert's shape; this one drives the path TE actually
uses for MoE -- ``tex.group_quantize`` into packed ``GroupedTensor`` storage, then
``general_grouped_gemm_for_grouped_tensor`` across all local experts in one launch.

Operand layout per GEMM, matching ``pytorch/ops/fused/grouped_mlp.py``. Weights are
quantized per expert (discrete list); activations and gradients are packed and
group-quantized:

    gemm    A (usage)              B (usage)            layout  out
    ------  ---------------------  -------------------  ------  --------------------
    fprop   W discrete, rowwise    X grouped, rowwise   TN      grouped (rows, N)
    dgrad   W discrete, columnwise dY grouped, rowwise  NN      grouped (rows, K)
    wgrad   X grouped, columnwise  dY grouped, colwise  NT      discrete list (N, K)

Timing follows the dense script: three loops per GEMM sharing one set of quantized
operands -- ``gemm`` (pre-quantized), ``quant`` (casts alone), ``quant+gemm`` --
with ``overhead = (quant+gemm) - gemm``. ``--step-total`` measures one full training
pass with X, W and dY each quantized once using a fused rowwise+columnwise cast.

RHT is mandatory here
---------------------
Grouped NVFP4 quantization on TE requires the Random Hadamard Transform: the
non-RHT kernel is unimplemented and ``group_quantize`` raises

    "graph safe grouped quant kernel for non-RHT path is not ready yet"

(``pytorch/csrc/extensions/cast.cpp``), and post-RHT amax is likewise required.
The dense script defaults to a plain 1D recipe with RHT off, so its numbers and
these are **not** directly comparable -- any gap mixes the grouped path with the
cost of the RHT cast itself. Use ``--no-rht`` only to confirm the guard still
fires on a newer build.

Requires Blackwell (SM100+) and cuBLAS 13.3+ (TE gates grouped GEMM on it; older
runtimes raise from ``check_grouped_gemm_requirements``).

Known limitation: quant timings include allocation
--------------------------------------------------
The dense script preallocates its destinations and reuses them via
``update_quantized``, keeping allocator churn out of the timed loop.
``tex.group_quantize`` allocates its output instead, and its ``output=`` reuse
path is unusable here: reuse requires uniform shapes (``first_dims=None``), but
that path currently dies with an illegal memory access inside
``group_row_col_rht_gemm_ntt_w_sfc_graph_safe``
(``common/hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu``),
reproducible with NVFP4 + RHT even for a single ``group_quantize`` call. Passing
explicit ``first_dims`` works but forbids reuse. So the ``quant`` column here
carries per-call allocation that the dense script's does not, and is an
overestimate of pure kernel time.

Usage::

    # default: 8 local experts, sweep tokens/expert
    python benchmarks/linear/benchmark_grouped_quant_overhead.py --step-total

    # DeepSeek-V3 with EP=32 (256 routed experts / 32 ranks = 8 local)
    python benchmarks/linear/benchmark_grouped_quant_overhead.py \\
        --experts 8 --tokens-per-expert 512,1024,2048 --amortize-weight --step-total
"""

import argparse

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

import transformer_engine.pytorch as te  # noqa: F401  must be first per te-python-import-order
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor
from transformer_engine.pytorch.tensor import GroupedTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

# DeepSeek-V3 MoE expert FC1. TE fuses gate and up for gated activations
# (fc1_output_features = 2 * ffn_hidden_size in pytorch/module/layernorm_mlp.py),
# so the dispatched GEMM is N = 2 * moe_intermediate_size = 2 * 2048.
HIDDEN_SIZE = 7168
FC1_OUT = 4096

# layout and accumulator mode per GEMM; split_acc mirrors _2X_ACC_{FPROP,DGRAD,WGRAD}
# in transformer_engine/pytorch/module/base.py.
GEMM_SPECS = {
    "fprop": {"layout": "TN", "split_acc": False},
    "dgrad": {"layout": "NN", "split_acc": True},
    "wgrad": {"layout": "NT", "split_acc": True},
}

RESULT_COLUMNS = [
    "experts",
    "tokens_per_expert",
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


# Bytes written per element, per usage: quantized datum + its block scale.
RECIPES = {
    "nvfp4": {"write_bytes": 0.5 + 1 / 16, "has_amax": True},
    "mxfp8": {"write_bytes": 1.0 + 1 / 32, "has_amax": False},
}


def make_quantizer(usage, with_rht=True, recipe="nvfp4"):
    """Quantizer for the grouped path. ``usage`` is rowwise/columnwise/both.

    For NVFP4, RHT and post-RHT amax are on by default because grouped NVFP4
    quantization has no non-RHT kernel; see the module docstring. MXFP8 has no
    RHT and no global amax, so ``with_rht`` does not apply to it.
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
            with_rht=with_rht,
            with_post_rht_amax=with_rht,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )
    quantizer.optimize_for_gemm = True
    return quantizer


def group_quantize(packed, quantizer, first_dims, num_experts):
    """Cast a packed activation/gradient into grouped NVFP4 storage."""
    return tex.group_quantize(packed, quantizer, num_experts, first_dims)


def empty_grouped(first_dims, num_experts, rows, last_dim):
    """Allocate packed bf16 GroupedTensor output storage."""
    return GroupedTensor.make_grouped_tensor(
        num_tensors=num_experts,
        first_dims=first_dims,
        last_dims=None,
        logical_first_dim=rows,
        logical_last_dim=last_dim,
        quantizer=None,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


def grouped_gemm(a, b, out, name):
    spec = GEMM_SPECS[name]
    general_grouped_gemm_for_grouped_tensor(
        a, b, out, layout=spec["layout"], use_split_accumulator=spec["split_acc"]
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_us(fn, iters, warmup, repeats, lead_fn=None):
    """Microseconds per call: min over ``repeats`` timed loops of ``iters`` calls.

    The SM clock idles far below boost, so a short burst measures the ramp rather
    than steady state. Warm up to reach boost, then take the min -- any transient
    only inflates a sample.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(repeats):
        # Leading kernel hides CPU dispatch for the first timed iteration; events
        # record in stream order so its own duration is excluded.
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


def summarize(label, experts, tokens, K, N, quant_us, gemm_us, total_us, num_gemms=1):
    """Assemble one result row from the three timed loops."""
    rows = experts * tokens
    overhead_us = total_us - gemm_us
    flops = 2 * rows * N * K * num_gemms
    return {
        "experts": experts,
        "tokens_per_expert": tokens,
        "M": rows,
        "K": K,
        "N": N,
        "gemm": label,
        "quant_us": quant_us,
        "gemm_us": gemm_us,
        "total_us": total_us,
        "overhead_us": overhead_us,
        "quant_pct": 100.0 * overhead_us / total_us if total_us > 0 else float("nan"),
        "gemm_tflops": flops / (gemm_us * 1e-6) / 1e12 if gemm_us > 0 else 0.0,
        "eff_tflops": flops / (total_us * 1e-6) / 1e12 if total_us > 0 else 0.0,
    }


class Operands:
    """High-precision inputs and their grouped/discrete NVFP4 destinations."""

    def __init__(self, experts, tokens, K, N, with_rht, recipe="nvfp4"):
        self.experts, self.tokens, self.K, self.N = experts, tokens, K, N
        self.rows = experts * tokens
        self.first_dims = torch.tensor([tokens] * experts, dtype=torch.int64, device="cuda")
        self.x_hp = torch.randn(self.rows, K, dtype=torch.bfloat16, device="cuda")
        self.dy_hp = torch.randn(self.rows, N, dtype=torch.bfloat16, device="cuda")
        self.w_hp = [torch.randn(N, K, dtype=torch.bfloat16, device="cuda") for _ in range(experts)]
        self.with_rht = with_rht
        self.recipe = recipe

    def quantizer(self, usage):
        return make_quantizer(usage, self.with_rht, self.recipe)

    def grouped(self, packed, usage):
        return group_quantize(packed, self.quantizer(usage), self.first_dims, self.experts)

    def discrete_weights(self, usage):
        """Weights are quantized per expert, as grouped_mlp does for fprop/dgrad."""
        quantizer = self.quantizer(usage)
        return [quantizer(w) for w in self.w_hp]

    def out_grouped(self, last_dim):
        return empty_grouped(self.first_dims, self.experts, self.rows, last_dim)

    def out_discrete(self):
        return [
            torch.empty(self.N, self.K, dtype=torch.bfloat16, device="cuda")
            for _ in range(self.experts)
        ]


def benchmark_one(gemm, experts, tokens, K, N, args, lead_fn):
    """Time gemm / quant / quant+gemm for one grouped training GEMM."""
    ops = Operands(experts, tokens, K, N, args.with_rht, args.recipe)
    quantize_weight = not args.amortize_weight

    if gemm == "fprop":
        w_q = ops.discrete_weights("rowwise")
        x_q = ops.grouped(ops.x_hp, "rowwise")
        out = ops.out_grouped(N)
        a, b = w_q, x_q
        w_quantizer, act_quantizer = ops.quantizer("rowwise"), ops.quantizer("rowwise")

        def run_quant():
            if quantize_weight:
                for i, w in enumerate(ops.w_hp):
                    w_q[i] = w_quantizer(w)
            ops.grouped(ops.x_hp, "rowwise")

    elif gemm == "dgrad":
        w_q = ops.discrete_weights("columnwise")
        dy_q = ops.grouped(ops.dy_hp, "rowwise")
        out = ops.out_grouped(K)
        a, b = w_q, dy_q
        w_quantizer = ops.quantizer("columnwise")

        def run_quant():
            if quantize_weight:
                for i, w in enumerate(ops.w_hp):
                    w_q[i] = w_quantizer(w)
            ops.grouped(ops.dy_hp, "rowwise")

    elif gemm == "wgrad":
        x_q = ops.grouped(ops.x_hp, "columnwise")
        dy_q = ops.grouped(ops.dy_hp, "columnwise")
        out = ops.out_discrete()
        a, b = x_q, dy_q

        def run_quant():
            # No weight operand in wgrad, so amortization cannot help it.
            ops.grouped(ops.x_hp, "columnwise")
            ops.grouped(ops.dy_hp, "columnwise")

    else:
        raise ValueError(f"unknown gemm '{gemm}'")

    def run_gemm():
        grouped_gemm(a, b, out, gemm)

    def run_full():
        run_quant()
        run_gemm()

    def timed(fn):
        return time_us(fn, args.iters, args.warmup, args.repeats, lead_fn)

    return summarize(
        gemm, experts, tokens, K, N, timed(run_quant), timed(run_gemm), timed(run_full)
    )


def benchmark_step(experts, tokens, K, N, args, lead_fn):
    """One full training pass: 3 grouped GEMMs, each tensor quantized once.

    Not the sum of the per-GEMM rows: X and dY are each consumed by two GEMMs, so
    a real step group-quantizes each once with rowwise+columnwise usage rather
    than twice with a single usage.
    """
    ops = Operands(experts, tokens, K, N, args.with_rht, args.recipe)
    w_quantizer = ops.quantizer("both")
    w_q = [w_quantizer(w) for w in ops.w_hp]
    x_q = ops.grouped(ops.x_hp, "both")
    dy_q = ops.grouped(ops.dy_hp, "both")

    y = ops.out_grouped(N)
    dx = ops.out_grouped(K)
    dw = ops.out_discrete()

    def run_gemms():
        grouped_gemm(w_q, x_q, y, "fprop")
        grouped_gemm(w_q, dy_q, dx, "dgrad")
        grouped_gemm(x_q, dy_q, dw, "wgrad")

    def run_quant():
        # The weight is cast once per optimizer step, so with gradient
        # accumulation its cost vanishes per microbatch; X and dY never do.
        if not args.amortize_weight:
            for i, w in enumerate(ops.w_hp):
                w_q[i] = w_quantizer(w)
        ops.grouped(ops.x_hp, "both")
        ops.grouped(ops.dy_hp, "both")

    def run_full():
        run_quant()
        run_gemms()

    def timed(fn):
        return time_us(fn, args.iters, args.warmup, args.repeats, lead_fn)

    row = summarize(
        "step(3 gemms)",
        experts,
        tokens,
        K,
        N,
        timed(run_quant),
        timed(run_gemms),
        timed(run_full),
        num_gemms=3,
    )
    # X and dY are packed across experts; the weight is E separate (N, K) tensors.
    parts = [(ops.rows * K, 2), (ops.rows * N, 2)]
    if not args.amortize_weight:
        parts.append((experts * N * K, 2))
    row.update(add_cast_bandwidth(run_quant, row, parts, args.breakdown, args.recipe))
    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(frame, title, per_gemm=True):
    """Print a results frame as experts / tokens / (M,N,K) / timings."""
    table = frame.copy()
    table["(M,N,K)"] = [f"({m},{n},{k})" for m, n, k in zip(table.M, table.N, table.K)]
    table["E"] = table.experts
    table["tok/E"] = table.tokens_per_expert

    columns = ["E", "tok/E", "(M,N,K)"]
    if per_gemm:
        columns.append("gemm")
    columns += ["gemm_us", "quant_us"]
    if "amax_us" in table:
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
    for config in configs:
        # Release the previous config's buffers, or allocator fragmentation
        # carries across configs and later shapes measure slower than in isolation.
        torch.cuda.empty_cache()
        try:
            row = measure(*config)
        except Exception as exc:  # noqa: BLE001  a bad shape shouldn't kill the sweep
            print(f"  skip {describe(*config)}: {exc}")
            continue
        if row["overhead_us"] <= 0:
            # quant+gemm no slower than gemm alone is physically impossible.
            print(f"  WARNING {describe(*config)}: total <= gemm, discarding row")
            continue
        rows.append(row)
        print(
            f"  {describe(*config):<34}"
            f" quant={row['quant_us']:8.1f}us gemm={row['gemm_us']:8.1f}us"
            f" total={row['total_us']:8.1f}us  quant={row['quant_pct']:5.1f}%"
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experts", type=int, default=8, help="Local experts per rank (after expert parallelism)."
    )
    parser.add_argument(
        "--tokens-per-expert",
        type=str,
        default="512,1024,2048",
        help="Comma-separated token counts per expert.",
    )
    parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE, help="Hidden size (K).")
    parser.add_argument(
        "--fc1-out", type=int, default=FC1_OUT, help="FC1 output (N), gate and up fused."
    )
    parser.add_argument(
        "--gemms",
        type=str,
        default="fprop,dgrad,wgrad",
        help="Comma-separated subset of fprop,dgrad,wgrad.",
    )
    parser.add_argument(
        "--amortize-weight",
        action="store_true",
        help=(
            "Exclude weight quantization from the timed work, modelling a weight cast"
            " once per optimizer step and reused across microbatches."
        ),
    )
    parser.add_argument(
        "--step-total",
        action="store_true",
        help=(
            "Also measure one full training pass per config: all 3 grouped GEMMs with"
            " X/W/dY each quantized once. Measured directly, not summed."
        ),
    )
    parser.add_argument(
        "--no-rht",
        dest="with_rht",
        action="store_false",
        help=(
            "Disable RHT. Grouped NVFP4 quantization has no non-RHT kernel on current"
            " TE, so this is expected to raise; use it to check a newer build."
        ),
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help=(
            "Split the quant column into its amax / cast kernels via torch.profiler."
            " On this path the scale swizzle is genuinely fused into the cast kernel,"
            " so no separate swizzle time appears."
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default="nvfp4",
        help=(
            "Quantization recipe. nvfp4 is a two-pass cast (global amax, then cast)"
            " and requires RHT on this path; mxfp8 has per-block E8M0 scales, no"
            " global amax, and no RHT requirement."
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


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise SystemExit("NVFP4 grouped GEMM requires Blackwell (SM100+).")

    token_counts = [int(t) for t in args.tokens_per_expert.split(",") if t.strip()]
    gemms = [g.strip() for g in args.gemms.split(",") if g.strip()]
    K, N = args.hidden, args.fc1_out

    print(f"Device: {torch.cuda.get_device_name()} (SM{major}0)")
    print(
        f"Recipe: {args.recipe.upper()}, grouped quantize + grouped GEMM"
        + (
            f", RHT={'on' if args.with_rht else 'off'} (grouped NVFP4 requires RHT)"
            if args.recipe == "nvfp4"
            else " (no amax, no RHT)"
        )
    )
    print(
        f"experts={args.experts}, K={K}, N={N},"
        f" amortize_weight={args.amortize_weight}, warmup={args.warmup},"
        f" iters={args.iters}, repeats={args.repeats} (min)\n"
    )

    lead_a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    lead_b = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

    def lead_fn():
        torch.matmul(lead_a, lead_b)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", None)

    rows = collect(
        [(g, t) for t in token_counts for g in gemms],
        lambda g, t: benchmark_one(g, args.experts, t, K, N, args, lead_fn),
        lambda g, t: f"E={args.experts} tok/E={t} {g}",
    )
    if not rows:
        raise SystemExit("No successful measurements.")

    frame = pd.DataFrame(rows)[RESULT_COLUMNS]
    report(frame, "Per-GEMM (grouped)")

    if args.step_total:
        step_rows = collect(
            [(t,) for t in token_counts],
            lambda t: benchmark_step(args.experts, t, K, N, args, lead_fn),
            lambda t: f"E={args.experts} tok/E={t} step",
        )
        if step_rows:
            extra = [c for c in ("amax_us", "cast_us", "cast_mb", "cast_gbps") if c in step_rows[0]]
            step_frame = pd.DataFrame(step_rows)[RESULT_COLUMNS + extra]
            report(step_frame, "One training pass: 3 grouped GEMMs, each tensor cast once", False)
            if args.output:
                step_out = args.output.replace(".csv", "_step.csv")
                step_frame.to_csv(step_out, index=False)
                print(f"\nWrote {step_out}")

    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
