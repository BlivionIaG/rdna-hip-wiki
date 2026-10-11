---
title: "Triton attention on gfx1030 — kernel baseline and ISA map"
sources:
  - https://github.com/vllm-project/vllm/commit/49fb2ee3481246ce88224220442c6ef30f88b209
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/triton_attn.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/rocm_attn.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/triton_unified_attention.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/prefix_prefill.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/mla/triton_mla.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/triton_decode_attention.py
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/platforms/rocm.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/prefix_prefill.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/triton_mla.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py
---
# Triton attention on gfx1030 — kernel baseline and ISA map

Snapshot: vLLM `49fb2ee3481246ce88224220442c6ef30f88b209` (2026-08-18). **Baseline evidence, not tuning.** FP16 only for gfx1030 comparisons.[^c-49fb2ee]

## Names that must not be conflated

| Name | Actual path |
|---|---|
| `TRITON_ATTN` | vLLM unified Triton paged attention [^f-49fb2ee-triton-attn-py][^f-49fb2ee-rocm-attn-py] |
| `ROCM_ATTN` | hybrid: Triton prefill, native ROCm decode only when strict gates pass, otherwise Triton decode [^f-49fb2ee-triton-attn-py][^f-49fb2ee-rocm-attn-py] |
| Triton AMD FlashAttention package | external FlashAttention/AITER Triton AMD path used by ViT `FLASH_ATTN`; current vLLM gate is gfx11/gfx12 [^f-49fb2ee-triton-attn-py][^f-49fb2ee-rocm-attn-py] |
| `TRITON_MLA` | separate grouped Triton MLA decode + reduction [^f-49fb2ee-triton-attn-py][^f-49fb2ee-rocm-attn-py] |

On current main, gfx1030 fails the native Radeon paged-attention predicate. `ROCM_ATTN` decode is expected to call Triton `kernel_paged_attention_2d`; label it **ROCM_ATTN / Triton fallback**, never native Radeon attention.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-rocm-attn-py]

## `TRITON_ATTN`

Sources: `vllm/v1/attention/backends/triton_attn.py` and `ops/triton_unified_attention.py` (`unified_attention`, `kernel_unified_attention`, `reduce_segments`).[^f-49fb2ee-triton-attn-py]

Stock contract: cache block multiple of 16; head size >=32 padded to next power of two; logical cache `(num_blocks,num_kv_heads,block_size,2*head_size)` with NHD/HND selected by strides.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]

Stock tile rules:

- `BLOCK_M=16` for GQA ratio <=16, otherwise next power of two of the ratio.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]
- `BLOCK_Q=BLOCK_M/queries_per_kv`.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]
- FP16 prefill `TILE_SIZE=32`; decode 16.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]
- Gemma3 head 128/256 with sliding window 1024 uses 32.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]
- Explicit 8 warps/2 stages is B200 head-256 prefill only, not AMD tuning.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]

2D grid is `(total_query_blocks,num_kv_heads)`. Small pure-decode batches may use 16 softmax segments in a 3D grid plus `reduce_segments`; initial threshold is `128//num_kv_heads`, rounded to a graph capture size. Prefill, larger batches, missing workspace or batch-invariant mode force 2D.[^f-49fb2ee-triton-attn-py][^f-49fb2ee-triton-unified-attention-py]

## `ROCM_ATTN` hybrid

Sources: `backends/rocm_attn.py`, `ops/chunked_prefill_paged_decode.py`, `ops/prefix_prefill.py`, `platforms/rocm.py`.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]

- Query length >1: Triton `context_attention_fwd`.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]
- Decode: native `ops.paged_attention_rocm` only if its predicate passes; else Triton `kernel_paged_attention_2d`.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]
- Cache update: native write only for compatible block/layout; otherwise `triton_reshape_and_cache_flash`.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]

Current Radeon native gate is gfx11/gfx12, FP16/BF16, head 128, block 16, GQA 3–16, no ALiBi, automatic KV dtype and no sinks. gfx1030 is excluded.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]

Stock prefill: power-of-two block uses `BLOCK_M=128`, `BLOCK_N=64`; non-power-of-two uses 32/32; context tile 32; 4 warps, 1 stage; cache/request unroll 4/1.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]

Fallback decode: logical tile `min(block_size,128)` for power-of-two blocks, else 32; head padded to next power of two; GQA width padded to at least 16. Physical block remains independent.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]

Graph capture fills sequence lengths with 1 and zeros query-start locations to avoid pathological capture/invalid accesses. Test capture and replay per shape.[^f-49fb2ee-rocm-attn-py][^f-49fb2ee-chunked-prefill-paged-decode-py][^f-49fb2ee-prefix-prefill-py]

## Triton MLA

Sources: `backends/mla/triton_mla.py`, `ops/triton_decode_attention.py`.[^f-49fb2ee-triton-mla-py][^f-49fb2ee-triton-decode-attention-py]

KV splits: `min(next_power_of_2(max(1,max_seq_len//512)),2*num_compute_units)`; batch-invariant mode forces one.[^f-49fb2ee-triton-mla-py][^f-49fb2ee-triton-decode-attention-py]

HIP grouped stage: `BLOCK_N=16`, `BLOCK_H=16`, 4 warps, 1 stage, `waves_per_eu=1`, `matrix_instr_nonkdim=16`, `kpack=2`. Reduction: 4 warps, 2 stages, `waves_per_eu=4`. Workspace `(batch,q_heads,splits,kv_lora_rank+1)` fp32. Graph support is uniform single-token decode only.[^f-49fb2ee-triton-mla-py][^f-49fb2ee-triton-decode-attention-py]

On gfx1030 these compiler hints do not prove MFMA or good occupancy. Record generated ISA and actual waves.[^f-49fb2ee-triton-mla-py][^f-49fb2ee-triton-decode-attention-py]

## Honest gfx1030 matrix

1. Forced FP16 `TRITON_ATTN`.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]
2. Forced FP16 `ROCM_ATTN`, labeled hybrid/Triton fallback after profiler proof.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]
3. `TRITON_MLA` only if the exact shape compiles and passes correctness.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]
4. Custom `fa_rdna2` only after cache layout, masks, sinks, sliding-window, graph mode and tolerances match.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]

Sweep query `{1,16,64,256}`, context `512..32768`, head `{64,96,128,256}`, block `{16,32,64}`, GQA and sliding-window. Separate prefill, decode, extend and MLA.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]

Record backend, profiler symbol, 2D/3D choice, split count, constexpr signature, cold compile, warm disk restart, steady median/p10/p90, graph capture/replay, correctness, VGPR/LDS/spills/occupancy and ISA.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]

## Tuning order

Freeze stock evidence first. Tune logical tiles, warps, stages, sequence splits and cache layout on a separate page. Verify lowering, LDS, waitcnt and spills. Compare compile-signature count and disk-cache behavior as well as runtime. HIP A/B uses identical semantics and timing boundaries.[^f-49fb2ee-rocm-py][^f-49fb2ee-triton-unified-attention-py]

## Primary sources

- https://github.com/vllm-project/vllm/commit/49fb2ee3481246ce88224220442c6ef30f88b209
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/prefix_prefill.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/triton_mla.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py
- https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_triton_unified_attention.py *(unverifiable: file not present at vLLM `49fb2ee` nor on current `main`)*

[^c-49fb2ee]: vllm-project/vllm commit [`49fb2ee`](https://github.com/vllm-project/vllm/commit/49fb2ee3481246ce88224220442c6ef30f88b209) (2026-08-17) — "\[ROCm\]\[CI\] add Aiter ops tests (#52208)".
[^f-49fb2ee-triton-attn-py]: vllm-project/vllm `vllm/v1/attention/backends/triton_attn.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/triton_attn.py).
[^f-49fb2ee-rocm-attn-py]: vllm-project/vllm `vllm/v1/attention/backends/rocm_attn.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/rocm_attn.py).
[^f-49fb2ee-triton-unified-attention-py]: vllm-project/vllm `vllm/v1/attention/ops/triton_unified_attention.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/triton_unified_attention.py).
[^f-49fb2ee-chunked-prefill-paged-decode-py]: vllm-project/vllm `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/chunked_prefill_paged_decode.py).
[^f-49fb2ee-prefix-prefill-py]: vllm-project/vllm `vllm/v1/attention/ops/prefix_prefill.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/prefix_prefill.py).
[^f-49fb2ee-triton-mla-py]: vllm-project/vllm `vllm/v1/attention/backends/mla/triton_mla.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/mla/triton_mla.py).
[^f-49fb2ee-triton-decode-attention-py]: vllm-project/vllm `vllm/v1/attention/ops/triton_decode_attention.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/ops/triton_decode_attention.py).
[^f-49fb2ee-rocm-py]: vllm-project/vllm `vllm/platforms/rocm.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/platforms/rocm.py).
