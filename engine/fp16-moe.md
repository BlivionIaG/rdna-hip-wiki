# Native HIP FP16 MoE on gfx1030 — draft

Date: 2026-08-18. **Spec / Todo.** Engine contract; silicon tile belongs in [kernels/fp16-moe.md](../kernels/fp16-moe.md). Target: vLLM V1, gfx1030 wave32, fp16 activations + fp16 weights, fp32 accumulation. No MFMA/WMMA/AGPR/bf16. Inner op must be `V_DOT2_F32_F16` (`fdot2`).

This is a new unquantized FP16 MoE backend, not a rewrite of `moe_q_gemm_rdna2` and not AITER/CK.

## Verdict

One tile cannot serve both phases:

| Phase | Expert-token rows | Kernel family | Goal |
|---|---:|---|---|
| Decode | usually 1–8 per local expert | skinny grouped GEMV/GEMM | stream each expert weight once per token bucket; maximize resident waves |
| Prefill / verify | tens–hundreds | tiled grouped GEMM | reuse A in LDS/registers across N; amortize weight traffic |

Dispatch by **tokens per expert after routing**, not total batch. Initial crossover is tunable; benchmark `{1,2,4,8,16,32}` rows/expert.

## vLLM contract

Input after existing align/sort:

- `x`: `[M, K]` fp16.
- `topk_ids`, `topk_weights`: `[M, top_k]`.
- `sorted_token_ids`, `expert_ids`, `num_tokens_post_pad` from vLLM MoE alignment. Do not write a second router/sorter in v1.
- `w13`: `[E, 2N, K]` fp16; `w2`: `[E, K, N]` fp16 (logical shapes; one-time native layout transform is allowed).
- output: `[M, K]` fp16.

Pipeline:

1. Align/sort routed token copies by expert.
2. `w13`: grouped FP16 GEMM, fp32 accumulation.
3. Fuse bias (if present) + split gate/up + SiLU + multiply into an fp16 intermediate `[M*top_k, N]`.
4. `w2`: grouped FP16 GEMM, fp32 accumulation.
5. Fuse `topk_weight` into the `w2` epilogue. If TP/EP requires an external collective, write unreduced routed rows; combine only where vLLM semantics permit it.

Two GEMMs remain separate because SiLU is between them. “Single fused MoE kernel” must not imply an illegal grid-wide dependency.

## Kernel architecture

### Decode skinny path

- Specialize routed rows/expert: `BLOCK_M ∈ {1,2,4,8}`; avoid padding every expert to a large M tile.
- Work item = `(expert, N tile, routed-row bucket)` from the sorted metadata. No atomics for ownership.
- Stage/broadcast A; stream contiguous packed `half2` weights; accumulate fp32 with `fdot2`.
- Split-K only when the `(expert × N-tile × row-bucket)` grid cannot fill 72 CUs; reduction cost must be measured.
- Fuse gate/up activation in `w13`; fuse route scale and legal combine in `w2`.

### Prefill / verify grouped path

- Initial exploration tile: `M={16,32}`, `N={32,64}`, `K=32` per stage. Silicon page owns the final shape.
- Cooperative 16-byte global loads; double-buffer A/B only if VGPR/LDS occupancy survives.
- A tile in LDS, B streamed or tiled according to measured reuse. Pad LDS strides to avoid 32-bank conflicts.
- `fdot2` packed-half inner loop, fp32 accumulators, vector fp16 stores.

## Layout

One-time weight packing at `process_weights_after_loading`, never per forward:

- K contiguous and even; pack as `half2`/32-bit words for `fdot2`.
- Preserve an expert-major outer dimension and N-tile contiguity.
- Keep `w13` gate/up adjacent enough for the fused epilogue without duplicating weights.
- Record the layout/version on the method; reject incompatible strides instead of silently copying each step.

## Occupancy contract

- No CUDA interpretation of HIP `__launch_bounds__`.
- Report VGPR/SGPR/LDS, waves/SIMD32 and spills for every specialization.
- Decode target starts at `amdgpu_waves_per_eu(4,8)` only if the register budget permits; verify ISA and occupancy rather than forcing it blindly.
- No local-memory spills in the K loop. Reduce tile/accumulator count before accepting spills.

This is a separate card from the current `fa_rdna2` occupancy fix; `fa_rdna2` remains first.

## Dispatch / fallback

Enable only when:

- gfx10x + ROCm, fp16 x/w13/w2, supported activation, K even and native packed layout present.
- Expert metadata matches the vLLM grouped contract.

Fallback to the existing upstream/Triton/torch path on unsupported shapes. Do not affect W4A16/W8A16/MXFP4 dispatch.

Proposed op surface (names provisional):

```text
moe_fp16_w13_rdna2(x_sorted, w13_packed, expert_ids, offsets, out_gateup)
moe_fp16_w2_rdna2(gateup, w2_packed, expert_ids, offsets,
                  topk_weights, out, combine_mode)
```

## Correctness gates

1. Empty expert; empty batch; expert sentinel `-1`.
2. Nonuniform rows/expert and tail M/N/K.
3. `top_k={1,2,8}` and duplicate routing rows.
4. SiLU×up parity against PyTorch; fp32-accum tolerances.
5. TP>1: no premature combine before AllReduce/AllToAll.
6. CUDA graph: no allocation, packing, or data-dependent host sync in forward.
7. Numeric + guard-page test before performance claims.

## Benchmark matrix

- Rows/expert `{1,2,4,8,16,32,64,128}`; realistic skew plus uniform synthetic.
- Representative K/N from served dense and routed experts.
- Compare end-to-end routed MoE and kernel-only against current fallback, rocBLAS loop, and existing quantized MoE only as context.
- Report first-token prefill separately from steady decode; TP=4 on 4×V620; eager and graph.
- Counters: DRAM bytes, L2/Infinity-cache hit behavior, occupancy, spills, launch count. No invented tok/s.

## Milestones / project card

One project card: **Native HIP FP16 MoE (gfx1030)**, Status **Todo**, priority after the current occupancy subject.

- [ ] Silicon tile + ISA page (`kernels/fp16-moe.md`).
- [ ] Reference op and vLLM method/dispatch.
- [ ] Decode `M=1/2/4/8` kernel.
- [ ] Prefill grouped tile.
- [ ] Fused w13 SiLU×up and w2 route-scale epilogues.
- [ ] TP=4 correctness; graph-safe.
- [ ] Occupancy/ISA dump and benchmark matrix.

## Non-goals

AITER/CK ports, MFMA emulation, bf16, expert offload, RCCL collectives inside the GEMM, changing the router, and merging this into quantized MoE kernels.