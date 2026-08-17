# Stock skinny GEMM on gfx1030 (and what does not fire)

Date: 2026-08-18. **Stock only.** vLLM `main` @ `49fb2ee`. Engine map: [engine/triton-rocm.md](../engine/triton-rocm.md). Tuning: [engine/triton-tuning.md](../engine/triton-tuning.md).

## Verdict

Stock unquantized decode GEMM on gfx1030 is **`torch.nn.functional.linear`** (rocBLAS / maybe hipBLASLt). HIP `wvSplitK` and `LLMM1` exist in `csrc/rocm/skinny_gemms.cu` but the Python gate is `on_gfx9() or on_gfx1x()`. **gfx1030 is neither.** `VLLM_ROCM_USE_SKINNY_GEMM=1` does nothing here.

Our fork occupancy ticket on `skinny_gemms.cu` is still real if we open that gate. ikantkode opening `LLMM1` for gfx1030 is overlay, not stock ([engine/qwen35.md](../engine/qwen35.md)).

## Dispatcher (`rocm_unquantized_gemm_impl`)

| Step | Kernel | gfx1030 |
|---|---|---|
| 1 | `wvSplitKrc` | gfx950 only |
| 2 | AITER Triton `gemm_a16w16` | CDNA3+ / selected RDNA4 |
| 3 | HIP `wvSplitK` | no (`on_gfx9` / `on_gfx1x`) |
| 4 | HIP `LLMM1` (`LLGemm1_kernel`) | no (same gate). `n==1`, `K<=8192`, no bias, N%4==0 |
| 5 | gfx950 AITER tuned GEMM | no |
| 6 | `F.linear` | **yes** |

`wvSplitK` extra: N>8, tokens 1..5, `K%8==0`, fp16/bf16, contiguous. `wvSplitKQ` FP8: MI3xx or gfx12 only. Rename PR #40827 (`vecMatMul`) is **not** this SHA — still `LLMM1`.

## Fair lab (after harness)

Shapes: real `q_proj` / `o_proj` / MLP, M `{1,2,4,5,8,16,64,…}`. Compare:

1. Stock `F.linear`.
2. `TORCH_BLAS_PREFER_HIPBLASLT=1` (label as prefer, not guaranteed).
3. Experimental: open `LLMM1` / `wvSplitK` on gfx1030 (fork / overlay) and ISA them.
4. Triton skinny GEMM (new) only after standalone numeric.
5. Custom HIP `fdot2` skinny / FP16 MoE decode.

Cold compile vs steady-state. Bias and non-contiguous as separate cells. Occupancy still first on any HIP we already ship.

## Sources

- `vllm/model_executor/layers/utils.py`
- `csrc/rocm/skinny_gemms.cu`, `torch_bindings.cpp` (`LLMM1`, `wvSplitK`)
- PRs #34709, #40827
