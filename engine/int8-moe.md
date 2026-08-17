# Native HIP INT8 MoE — W8A16 `fdot2` + W8A8 `sdot4`

Date: 2026-08-18. **Spec / Todo.** Engine contract. Silicon: [kernels/int8-moe.md](../kernels/int8-moe.md). Target gfx1030 / wave32 / vLLM V1. This is one dispatch family with two numeric routes, not one kernel pretending they are equivalent.

## Two routes

| Route | Storage | Activation | Inner op | Accum | First use |
|---|---|---|---|---|---|
| **W8A16** | signed i8 weights + scale/zero | fp16 | i8→fp16, then `V_DOT2_F32_F16` | fp32 | decode / tiny expert buckets |
| **W8A8** | signed i8 weights | signed i8 + scale | `V_DOT4_I32_I8` / `sdot4` | i32 through K, scale epilogue | prefill / fat expert buckets |

W8A16 is **storage-INT8**, not integer compute. W8A8 is the DP4A-like path. Never label both “INT8 GEMM” without the A dtype/opcode.

## Dispatch

Choose after routing/alignment from:

```text
(rows_per_expert, K, N, activation_scale_scheme, layout, graph_mode)
```

No single global M threshold. Seed policy only:

- W8A16 for rows/expert `{1,2,4,8}`.
- W8A8 for grouped prefill / verify after A8 exists.
- W8A8 decode only if A is already int8/static-scaled, or a fused per-token quant + GEMV beats W8A16 end to end.

Autotune/store a small shape key outside capture; graph replay must not benchmark or allocate.

## Shared vLLM pipeline

Reuse existing sort/alignment metadata: `sorted_token_ids`, `expert_ids`, expert offsets, `topk_weights`. Keep the same two-stage MoE pipeline:

1. `w13` grouped GEMM.
2. Owner-local split gate/up + SiLU×up.
3. `w2` grouped GEMM.
4. Fuse route scale only where TP/EP semantics allow; TP>1 writes unreduced routed rows before the collective.

Weights pack once in `process_weights_after_loading`. No repack/copy per forward.

## W8A16 path

Existing foundation: `moe_w8a16_rdna2.cu`; incomplete, not a new from-zero backend.

- Logical weight `[E,N,K]` int8; K-contiguous native packed view.
- Load i8, convert/dequant to packed fp16, issue explicit `fdot2`, fp32 accum.
- Decode seed from silicon: WG128, N512, K-stage256, rows/expert `1/2/4/8`; dequant in VGPR.
- Same gate/up adjacency and epilogues as FP16 MoE.
- Benchmark against current in-tree W8A16 MoE before replacing dispatch.

## W8A8 path

- Symmetric signed i8 A/W; pack 4 K bytes/dword; explicit `__builtin_amdgcn_sdot4(..., false)`.
- i32 accumulator remains unscaled through K. Epilogue: `fp16(acc_i32 * a_scale[row] * w_scale[col])`.
- Per-token A scale uses `[M,1]`; **`(1,1)` is per-token**, not per-tensor. Scalar/broadcast is distinguished by rank/stride, never `numel()==1` alone.
- Per-token/per-channel and per-tensor/per-tensor are the initial legal scale pairs. Asymmetric/mixed-sign is out of scope (`sudot4` is gfx11+, not gfx1030).
- Prefill seed: 64×64×64 i8, WG256, 16 KiB double-buffered LDS.
- Activation quant may be separate for prefill; decode must fuse quant/reduction or consume prequantized A8 before it is eligible.

## Op/dispatch surface (provisional)

```text
moe_w8a16_w13_rdna2(...)
moe_w8a16_w2_rdna2(...)
moe_w8a8_w13_rdna2(a8, a_scale, ...)
moe_w8a8_w2_rdna2(a8, a_scale, ..., topk_weights, combine_mode)
```

One Python method may own both routes only if the checkpoint provides compatible W8 weights/scales. Do not silently requantize a W8A16 checkpoint into a new W8A8 numeric contract.

## Correctness gates

- Empty expert, sentinel `-1`, nonuniform rows/expert, tails.
- `top_k={1,2,8}`, duplicate routed rows, TP=4 collective ordering.
- W8A16 parity vs dequantized PyTorch fp32-accum reference.
- W8A8 exact i32 dot reference before scale; scale-shape tests including `(1,1)`.
- Saturation/rounding policy locked to the checkpoint quant contract.
- No allocations/host sync in graph replay; guard-page/OOB tests.

## Benchmark matrix

Rows/expert `{1,2,4,8,16,32,64,128}`, representative `(K,N)`, realistic skew and uniform synthetic. Compare:

- existing W8A16 HIP,
- new/tuned W8A16 route,
- W8A8 including activation-quant time,
- current fallback.

Report kernel-only and routed MoE end-to-end; eager and graph; prefill and steady decode separately. Record ISA, VGPR/SGPR/LDS, spills, waves/SIMD32, bytes and launch count. Dispatch crossover is data, not a guessed constant.

## Project card

One Todo card: **Native HIP INT8 MoE — W8A16 fdot2 + W8A8 sdot4**. Links this page and [kernels/int8-moe.md](../kernels/int8-moe.md). Priority after the current `fa_rdna2` occupancy subject.

- [ ] Audit existing `moe_w8a16_rdna2.cu` correctness/dispatch.
- [ ] W8A16 decode shape matrix.
- [ ] W8A8 activation quant + grouped prefill.
- [ ] Conditional W8A8 decode experiment.
- [ ] Routed-row crossover table.
- [ ] TP=4 + graph-safe validation.

## Non-goals

AITER/CK/MFMA, bf16, `sudot4`, collectives inside GEMM, one universal threshold, or conflating W8A8-FP8 (`fdot2`) with integer W8A8 (`sdot4`).