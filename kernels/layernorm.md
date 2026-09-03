# HIP AOT RMSNorm — extras `83de31cf`

## dest tip 2026-09-03 — `opengfx1030/vllm-rdna` `rdna_extras` @ `ea78104d`

`layernorm.cu` is now on the dest CMake EXL3-unconditional list (same tile as `83de31cf`: one CTA/row, BLOCK_DIM 128/256/512/1024, LDS `NUM_WARPS` floats, no `fdot2`, no `__launch_bounds__`). Re-registered `rms_norm` / `fused_add_rms_norm` in `torch_bindings.cpp`. Not a new ISA flip.

Tip **`83de31cf8c55`** (`8e35767f` + `f22a15b6` + `83de31cf`, 2026-09-01). Replaces Triton `layer_norm_fwd_kernel` / `rms_norm_kernel` JIT on ROCm so cudagraph capture does not see a per-shape compile during inference. Occupancy leftover still FA prefill `(N, 1)` / LDS 1 WG per 64 KB — this is not a new first subject. FA pin stays closed.

Commit says **tested pending rebuild**. Do not claim PIECEWISE / CG-PATH NaNs are gone.

## Tile (sourced)

| Knob | Value |
|---|---|
| File | `csrc/rocm/layernorm.cu` |
| CMake | appended with EXL3 (`exl3_dot2_*` / `exl3_hadamard.cu`), **not** inside `VLLM_ROCM_HAS_GFX1030` |
| Grid | one CTA per row (`dim3 grid(M)`) |
| WG | `BLOCK_DIM` template 128 / 256 / 512 / 1024 via `pick_block_dim(N)` |
| Occupancy attr | **none** (no `__launch_bounds__`, no `waves_per_eu`) |
| LDS | `__shared__ float s_partial[NUM_WARPS]` = 16 / 32 / 64 / 128 B |
| Inner | scalar `__half2float` + fp32 FMA. **Not** `fdot2` |
| Reduce | `__shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, …)` then warp-0 LDS fold |
| Dtype | fp16 in/out, fp32 accum, fp16 weight. fp16 only |
| API | `rms_norm(out, input, weight, eps)`; `fused_add_rms_norm` overwrites `input` in-place (CUDA-compatible) |

`pick_block_dim`: N≤512 → 128; N≤2048 → 256; N≤8192 → 512; else 1024.

HIP `__shfl_xor_sync` MaskT must be 64-bit (`f22a15b6`). `0xFFFFFFFFu` fails the static assert.

Public entries are global-scope (`83de31cf` closed `vllm::rocm_layernorm` before them). Internal kernels stay in the namespace.

## Leave

- Copying tok/s or claiming the cudagraph NaN is fixed (tested pending)
- `fdot2` / half2 vectorize this lock (scalar loads; later HIP pass)
- Gemma fused `(1+w)` fp32 (ikantkode Take is still spec; this is Llama-style `x * rstd * w` fp16)
- `__launch_bounds__` without a VGPR dump
- Editing `rdna2_extras`
- Occupancy card: still FA prefill LDS, not this kernel
