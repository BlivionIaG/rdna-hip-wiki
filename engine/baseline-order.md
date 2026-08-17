# Baseline research order (stock first)

Date: 2026-08-18. Engine index. Stock evidence is **immutable**. Tuning lives on a different page. Occupancy on `fa_rdna2` / `skinny_gemms.cu` remains first.

Source snapshot for stock dispatch: vLLM `main` @ `49fb2ee` (2026-08-18). Our fork tip is `perf/rdna2_w4a16` @ `0c59068e`. Do not mix the two without saying which tree.

## Order

0. **Harness / dispatch proof.** Same model, dtype `half`, clocks, graph mode, TP, block size, request set. Log the **actual** backend and kernel name. No tok/s until dispatch is proven.
1. **Stock map.** [triton-rocm.md](triton-rocm.md). What gfx1030 actually launches today.
2. **Skinny GEMM lab.** [kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md). Stock is rocBLAS/`torch.nn.functional.linear`. HIP `wvSplitK` / `LLMM1` do **not** dispatch on gfx1030 in stock vLLM.
3. **Triton FA lab.** [triton-flash-attention.md](triton-flash-attention.md). `TRITON_ATTN` is the clean baseline. `ROCM_ATTN` on gfx1030 is Triton prefill + Triton `kernel_paged_attention_2d`, **not** HIP decode.
4. **HIP A/B.** Same contract as the stock path. `fa_rdna2` / custom skinny only after parity.
5. **FlyDSL gates.** [flydsl.md](flydsl.md). Compiler object → `fdot2`/`sdot4` ISA → skinny proof. No engine claim before that.

Tuning knobs: [triton-tuning.md](triton-tuning.md). Never write a tuned config back onto a stock page.

## Fair harness (locked)

- `--dtype half`. BF16 on gfx1030 is software-emulated (#38107).
- Record vLLM SHA, ROCm, Triton, graph mode, TP, block size, warmup excluded from steady-state.
- Attention: auto / `TRITON_ATTN` / `ROCM_ATTN`. q_len `{1,4,32,128,512,2048}`, ctx `{128,2k,8k,32k}`, concurrent `{1,4,16,64}`, head `{64,96,128,256}`, GQA `{1,4,8,16}`.
- GEMM M `{1,2,4,5,8,16,64,128,256,2048}` on real projection shapes. Separate `TORCH_BLAS_PREFER_HIPBLASLT=1` as a labelled BLAS variant, not a guaranteed hipBLASLt win.
- Confirm dispatch in logs/profiler. On gfx1030, `ROCM_ATTN` must show the Triton decode fallback, not `paged_attention_rocm`.

## Card

@VLLM_FORK_Manager: parent epic already on Project 4. Point children at these pages. Occupancy still first.
