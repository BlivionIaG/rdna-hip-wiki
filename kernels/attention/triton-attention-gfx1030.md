# Triton attention on gfx1030 — kernel baseline and ISA map

Snapshot: vLLM `49fb2ee3481246ce88224220442c6ef30f88b209` (2026-08-18). **Baseline evidence, not tuning.** FP16 only for gfx1030 comparisons.

## Names that must not be conflated

| Name | Actual path |
|---|---|
| `TRITON_ATTN` | vLLM unified Triton paged attention |
| `ROCM_ATTN` | hybrid: Triton prefill, native ROCm decode only when strict gates pass, otherwise Triton decode |
| Triton AMD FlashAttention package | external FlashAttention/AITER Triton AMD path used by ViT `FLASH_ATTN`; current vLLM gate is gfx11/gfx12 |
| `TRITON_MLA` | separate grouped Triton MLA decode + reduction |

On current main, gfx1030 fails the native Radeon paged-attention predicate. `ROCM_ATTN` decode is expected to call Triton `kernel_paged_attention_2d`; label it **ROCM_ATTN / Triton fallback**, never native Radeon attention.

## `TRITON_ATTN`

Sources: `vllm/v1/attention/backends/triton_attn.py` and `ops/triton_unified_attention.py` (`unified_attention`, `kernel_unified_attention`, `reduce_segments`).

Stock contract: cache block multiple of 16; head size >=32 padded to next power of two; logical cache `(num_blocks,num_kv_heads,block_size,2*head_size)` with NHD/HND selected by strides.

Stock tile rules:

- `BLOCK_M=16` for GQA ratio <=16, otherwise next power of two of the ratio.
- `BLOCK_Q=BLOCK_M/queries_per_kv`.
- FP16 prefill `TILE_SIZE=32`; decode 16.
- Gemma3 head 128/256 with sliding window 1024 uses 32.
- Explicit 8 warps/2 stages is B200 head-256 prefill only, not AMD tuning.

2D grid is `(total_query_blocks,num_kv_heads)`. Small pure-decode batches may use 16 softmax segments in a 3D grid plus `reduce_segments`; initial threshold is `128//num_kv_heads`, rounded to a graph capture size. Prefill, larger batches, missing workspace or batch-invariant mode force 2D.

## `ROCM_ATTN` hybrid

Sources: `backends/rocm_attn.py`, `ops/chunked_prefill_paged_decode.py`, `ops/prefix_prefill.py`, `platforms/rocm.py`.

- Query length >1: Triton `context_attention_fwd`.
- Decode: native `ops.paged_attention_rocm` only if its predicate passes; else Triton `kernel_paged_attention_2d`.
- Cache update: native write only for compatible block/layout; otherwise `triton_reshape_and_cache_flash`.

Current Radeon native gate is gfx11/gfx12, FP16/BF16, head 128, block 16, GQA 3–16, no ALiBi, automatic KV dtype and no sinks. gfx1030 is excluded.

Stock prefill: power-of-two block uses `BLOCK_M=128`, `BLOCK_N=64`; non-power-of-two uses 32/32; context tile 32; 4 warps, 1 stage; cache/request unroll 4/1.

Fallback decode: logical tile `min(block_size,128)` for power-of-two blocks, else 32; head padded to next power of two; GQA width padded to at least 16. Physical block remains independent.

Graph capture fills sequence lengths with 1 and zeros query-start locations to avoid pathological capture/invalid accesses. Test capture and replay per shape.

## Triton MLA

Sources: `backends/mla/triton_mla.py`, `ops/triton_decode_attention.py`.

KV splits: `min(next_power_of_2(max(1,max_seq_len//512)),2*num_compute_units)`; batch-invariant mode forces one.

HIP grouped stage: `BLOCK_N=16`, `BLOCK_H=16`, 4 warps, 1 stage, `waves_per_eu=1`, `matrix_instr_nonkdim=16`, `kpack=2`. Reduction: 4 warps, 2 stages, `waves_per_eu=4`. Workspace `(batch,q_heads,splits,kv_lora_rank+1)` fp32. Graph support is uniform single-token decode only.

On gfx1030 these compiler hints do not prove MFMA or good occupancy. Record generated ISA and actual waves.

## Honest gfx1030 matrix

1. Forced FP16 `TRITON_ATTN`.
2. Forced FP16 `ROCM_ATTN`, labeled hybrid/Triton fallback after profiler proof.
3. `TRITON_MLA` only if the exact shape compiles and passes correctness.
4. Custom `fa_rdna2` only after cache layout, masks, sinks, sliding-window, graph mode and tolerances match.

Sweep query `{1,16,64,256}`, context `512..32768`, head `{64,96,128,256}`, block `{16,32,64}`, GQA and sliding-window. Separate prefill, decode, extend and MLA.

Record backend, profiler symbol, 2D/3D choice, split count, constexpr signature, cold compile, warm disk restart, steady median/p10/p90, graph capture/replay, correctness, VGPR/LDS/spills/occupancy and ISA.

## Tuning order

Freeze stock evidence first. Tune logical tiles, warps, stages, sequence splits and cache layout on a separate page. Verify lowering, LDS, waitcnt and spills. Compare compile-signature count and disk-cache behavior as well as runtime. HIP A/B uses identical semantics and timing boundaries.

## Primary sources

- https://github.com/vllm-project/vllm/commit/49fb2ee3481246ce88224220442c6ef30f88b209
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/prefix_prefill.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/triton_mla.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py
- https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_triton_unified_attention.py
