# Stock ROCm Triton / HIP dispatch on gfx1030

Date: 2026-08-18. **Stock only.** vLLM `main` @ `49fb2ee`. Do not overwrite with tuned configs. Tuning: [triton-tuning.md](triton-tuning.md). Order: [baseline-order.md](baseline-order.md). Extras HIP vs leftover Triton JIT (after RMSNorm AOT): [kernels/triton-jit-aot.md](../kernels/triton-jit-aot.md).

"Supported" here means source/dispatch, not measured tok/s.

## Attention backends

ROCm auto-priority (`vllm/platforms/rocm.py::_get_backend_priorities`) for ordinary MHA: `ROCM_ATTN`, optional AITER, then `TRITON_ATTN`, `TURBOQUANT`. Select explicitly with `--attention-backend`.

| Backend | What it is | gfx1030 stock |
|---|---|---|
| `TRITON_ATTN` | Unified Triton FA (`triton_unified_attention.py`) | **Clean baseline.** No arch gate. Use `--dtype half`. |
| `ROCM_ATTN` | Hybrid: Triton prefill + *maybe* HIP decode | **Triton/Triton.** Native `paged_attention_rocm` is CDNA or gfx11/12 only. Decode falls to `kernel_paged_attention_2d`. |
| AITER FA / MLA | CDNA3+ / selected RDNA4 | **Dead** on this box |
| `TRITON_MLA` | Triton decode-attn split | Source-plausible, **unvalidated** on V620 |
| `RDNA_ATTN` / `fa_rdna2` | Our HIP | Fork only. Not the stock baseline. |

Removed / stale docs: `VLLM_V1_USE_PREFILL_DECODE_ATTENTION` and `VLLM_ROCM_CUSTOM_PAGED_ATTN` are not in current `vllm/envs.py`. Do not cite the ROCm 7.2.1 guide as current flags.

## `TRITON_ATTN` stock knobs (do not retune here)

`TritonAttentionImpl.forward` → `unified_attention`:

- `BLOCK_M=16` if GQA ratio ≤16, else next power of two. `BLOCK_Q = BLOCK_M / GQA`.
- Decode `TILE_SIZE=16`. Prefill `TILE_SIZE=32`.
- 3D split-softmax decode only if scratch exists, `max_seqlen_q==1`, seqs ≤ `seq_threshold_3D` (`128 // num_kv_heads`, snapped to a graph capture size), batch-invariance off. `NUM_PAR_SOFTMAX_SEGMENTS=16`.
- `VLLM_TRITON_USE_TD` exists; tensor descriptors auto-enable on XPU only. Do not force TD as the gfx1030 baseline.

Files: `vllm/v1/attention/backends/triton_attn.py`, `ops/triton_unified_attention.py`.

## `ROCM_ATTN` stock flow

`RocmAttentionImpl.forward`:

1. Prefill (`max_query_len>1`) → Triton `context_attention_fwd` (`prefix_prefill.py`). Stock: `BLOCK_M=128`, `BLOCK_N=64` (power-of-two blocks); `32/32` for nonstandard. `num_unroll_cache=4`, `num_warps=4`, `num_stages=1`.
2. Decode eligibility → `use_rocm_custom_paged_attention`.
3. Eligible → HIP `ops.paged_attention_rocm` (`csrc/rocm/attention.cu`).
4. Else → Triton `kernel_paged_attention_2d`.

Native decode gates: CDNA (head 64/128, …) or `_ON_GFX1X` (gfx11/12, head 128, …). **gfx1030 matches neither.** So this backend is still worth A/B vs unified `TRITON_ATTN` (different cache layout and decode kernel), but it is **not** a HIP-decode baseline on V620.

## Triton MLA stock

`TritonMLABackend` / `triton_decode_attention.py`. Auto when AITER MLA is off. Decoder-only. ROCm grouped: `BLOCK_N=16`, `BLOCK_H=16`, `num_warps=4`, `num_stages=1`, `waves_per_eu=1`, `kpack=2`. Split heuristic: min work 512, cap `2 * CUs`. No gfx1030 validation found. FP8 MLA is not a baseline here.

## Skinny GEMM stock

`rocm_unquantized_gemm_impl` (`vllm/model_executor/layers/utils.py`):

1. gfx950 `wvSplitKrc`
2. AITER Triton `gemm_a16w16` (allowlist)
3. HIP `wvSplitK` — requires `on_gfx9() or on_gfx1x()`
4. `LLMM1` — same arch gate; `n==1`, `K<=8192`, no bias
5. gfx950 AITER tuned GEMM
6. `torch.nn.functional.linear`

Master flag `VLLM_ROCM_USE_SKINNY_GEMM` (default on) **has no effect on gfx1030** under this gate. Stock decode GEMM is BLAS. ikantkode's overlay opening `LLMM1` for gfx1030 is **not stock** ([qwen35.md](qwen35.md)).

Detail: [kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md).

## Sources

- vLLM `49fb2ee`: `platforms/rocm.py`, `backends/triton_attn.py`, `backends/rocm_attn.py`, `ops/triton_unified_attention.py`, `ops/chunked_prefill_paged_decode.py`, `ops/prefix_prefill.py`, `layers/utils.py`
- #38107 (bf16 on RDNA2)
