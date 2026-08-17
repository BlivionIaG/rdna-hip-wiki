# Triton tuning knobs (not stock)

Date: 2026-08-18. **Do not copy these onto stock pages.** Stock configs: [triton-rocm.md](triton-rocm.md), [triton-flash-attention.md](triton-flash-attention.md), [kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md).

Change one axis at a time. Reject NaN, OOB, spill blow-up, or tolerance fail. Compile/autotune time is a metric, not hidden.

## Unified `TRITON_ATTN`

- decode `TILE_SIZE`: 16 (stock), 32, 64
- prefill `TILE_SIZE`: 16, 32 (stock), 64
- `BLOCK_M`: 8, 16 (stock), 32
- parallel softmax segments: 8, 16 (stock), 32
- 2D vs 3D around `seq_threshold_3D`
- `num_warps`: 4 vs 8
- `num_stages`: 1 vs 2
- do **not** force `VLLM_TRITON_USE_TD` as the first experiment

## Split `ROCM_ATTN` prefill

- `BLOCK_M/N`: (64,32), (64,64), (128,32), (128,64 stock)
- `num_unroll_cache`: 1, 2, 4 (stock)
- `num_warps`: 4 (stock), 8 (ikantkode overlay on paged decode — measure, don't assume)
- keep `num_stages=1` first

## Triton MLA

- `BLOCK_N`: 8, 16 (stock), 32
- `BLOCK_H`: 8, 16 (stock)
- split quantum 256 / 512 (stock) / 1024
- `waves_per_eu`: 1 (stock), 2
- `kpack`: 1, 2 (stock)

## Skinny GEMM

Stock has no Triton skinny on gfx1030. Experiments:

- open `LLMM1` / `wvSplitK` (arch-gate only) and ISA
- a new Triton GEMV (`tl.dot` vs scalar `tl.sum` — ikantkode GEMV was `tl.sum`, not `fdot2`)
- HIP `fdot2` skinny from [fp16-moe.md](fp16-moe.md)

`VLLM_TRITON_FORCE_FIRST_CONFIG=1` kills autotune noise; it does not pick the fastest config.
