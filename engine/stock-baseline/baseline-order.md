# Baseline research order (stock first)

Date: 2026-08-18. Engine index. Stock evidence is **immutable**. Tuning lives on a different page. Occupancy on `fa_rdna2` / `skinny_gemms.cu` remains first.

Source snapshot for stock dispatch: vLLM `main` @ `49fb2ee` (2026-08-18). Fork tip at the time *(historical)* was `perf/rdna2_w4a16` @ `0c59068e`; the current dest is [`opengfx1030/vllm-rdna` `rdna_extras`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras) (v0.31.0 port, tip `1dc3bac4`). Do not mix stock and dest without saying which tree.

## Order

0. **Harness / dispatch proof.** Same model, dtype `half`, clocks, graph mode, TP, block size, request set. Log the **actual** backend and kernel name. No tok/s until dispatch is proven.
1. **Stock map.** [triton-rocm.md](triton-rocm.md). What gfx1030 actually launches today.
2. **Skinny GEMM lab.** [kernels/gemm-moe/triton-skinny-gemm.md](../../kernels/gemm-moe/triton-skinny-gemm.md). Stock is rocBLAS/`torch.nn.functional.linear`. HIP `wvSplitK` / `LLMM1` do **not** dispatch on gfx1030 in stock vLLM.[^vllm-pr34709]
3. **Triton FA lab.** [triton-flash-attention.md](triton-flash-attention.md). `TRITON_ATTN` is the clean baseline. `ROCM_ATTN` on gfx1030 is Triton prefill + Triton `kernel_paged_attention_2d`, **not** HIP decode.
4. **HIP A/B.** Same contract as the stock path. `fa_rdna2` / custom skinny only after parity.
5. **FlyDSL gates.** [flydsl.md](../alt-engines/flydsl.md). Compiler object → `fdot2`/`sdot4` ISA → skinny proof. No engine claim before that.

Tuning knobs: [triton-tuning.md](triton-tuning.md). Never write a tuned config back onto a stock page.

## Fair harness (locked)

- `--dtype half`. gfx1030 has no bf16 dot/ALU, so BF16 runs through f32 conversion[^llvm-amdgpu-td] (#38107[^vllm-issue38107]).
- Record vLLM SHA, ROCm, Triton, graph mode, TP, block size, warmup excluded from steady-state.
- Attention: auto / `TRITON_ATTN` / `ROCM_ATTN`. q_len `{1,4,32,128,512,2048}`, ctx `{128,2k,8k,32k}`, concurrent `{1,4,16,64}`, head `{64,96,128,256}`, GQA `{1,4,8,16}`.
- GEMM M `{1,2,4,5,8,16,64,128,256,2048}` on real projection shapes. Do not count `TORCH_BLAS_PREFER_HIPBLASLT=1` as a hipBLASLt variant on gfx1030: gfx1030 is not in PyTorch's hipBLASLt-supported arch list, so torch warns "Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas" and the run re-measures rocBLAS[^torch-hipblaslt]. hipBLASLt itself excludes gfx1030 in TheRock[^therock1062]. Only label it a hipBLASLt variant on gfx1100.
- Confirm dispatch in logs/profiler. On gfx1030, `ROCM_ATTN` must show the Triton decode fallback, not `paged_attention_rocm`.

## Card

parent epic already on Project 4. Point children at these pages. Occupancy still first.

## Sources

Added in the 2026-10-11 sourcing pass:

- vLLM `rocm_attn.py` (ROCM_ATTN backend) — <https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py>
- vLLM `triton_attn.py` (TRITON_ATTN backend) — <https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py>

[^vllm-pr34709]: vLLM PR #34709 — wvSplitK skinny GEMM for gfx1x (gfx1030 not included) — <https://github.com/vllm-project/vllm/pull/34709>
[^vllm-issue38107]: vLLM issue #38107 — gfx1030 bf16 is emulated — <https://github.com/vllm-project/vllm/issues/38107>
[^torch-hipblaslt]: PyTorch `main` @ `7265531410fb`: `Context::blasPreferredBackend()` falls back to hipBLAS on unsupported archs — <https://github.com/pytorch/pytorch/blob/7265531410fb/aten/src/ATen/Context.cpp>; supported list (`gfx90a`, `gfx942`, `gfx1100`, `gfx1101`, …, no `gfx1030`) in `getHipblasltSupportedArchs()` — <https://github.com/pytorch/pytorch/blob/7265531410fb/aten/src/ATen/cuda/detail/CUDAHooks.cpp>
[^therock1062]: ROCm/TheRock#1062, hipBLASLt gfx archs excluded (incomplete tensilelite support) — <https://github.com/ROCm/TheRock/issues/1062>
[^llvm-amdgpu-td]: LLVM `llvm/lib/Target/AMDGPU/AMDGPU.td` @ `694d1a69de0d`: `FeatureISAVersion10_3_0` (gfx1030) has dot1/2/5/6/7/10 only; `v_dot2_bf16_bf16` is `dot9-insts`, `v_dot2_f32_bf16` is `dot12-insts` (gfx11+) — <https://github.com/llvm/llvm-project/blob/694d1a69de0d/llvm/lib/Target/AMDGPU/AMDGPU.td>
