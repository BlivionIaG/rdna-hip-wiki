---
title: "HIP AOT RMSNorm — extras `83de31cf` + gated `71a54552`"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/27e5fa40da6338dd2344927c97b3dad63990bf1f
  - https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/csrc/rocm/layernorm.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/vllm/model_executor/layers/layernorm.py
  - https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/docs/rdna2/tuning.md
  - https://github.com/opengfx1030/vllm-rdna/commit/13fbe2f9160027e71ab6c915ecbcb95d42373a2e
  - https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/csrc/rocm/layernorm.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/vllm/model_executor/layers/layernorm.py
  - https://github.com/opengfx1030/vllm-rdna/commit/56f67111dd87847c5e92e14a7076fec2c9ab6f22
  - https://github.com/opengfx1030/vllm-rdna/commit/71a54552ae8f6f8ad7f315c5a2291cebdb747690
  - https://github.com/opengfx1030/vllm-rdna/blob/56f67111dd87847c5e92e14a7076fec2c9ab6f22/csrc/rocm/layernorm.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/71a54552ae8f6f8ad7f315c5a2291cebdb747690/csrc/rocm/layernorm.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/71a54552ae8f6f8ad7f315c5a2291cebdb747690/csrc/rocm/ops.h
  - https://github.com/opengfx1030/vllm-rdna/blob/71a54552ae8f6f8ad7f315c5a2291cebdb747690/csrc/rocm/torch_bindings.cpp
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/layernorm.cu
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/torch_bindings.cpp
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/exl3_hadamard.cu
  - https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb
  - https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a
  - https://github.com/BlivionIaG/vllm/commit/f22a15b64bf4b3525dba8a7fca0379c3513cf5ca
---
# HIP AOT RMSNorm — extras `83de31cf` + gated `71a54552`

**At dest `cec69c12` (checked 2026-11-02):** new op `gemma_rms_norm(out, input, weight, eps)` on the plain RMSNorm tile. `rms_norm_kernel` gains a `bool kGemma` template parameter and a `long in_stride` argument; with `kGemma` each weight is turned into `(1 + w)` rounded to fp16 inside the kernel. Both ops now share one host launcher. `gemma_rms_norm` accepts a 2D input with unit column stride and any row stride `>= N` (a row-strided view, no copy); `rms_norm` keeps the contiguous-input contract; `weight` and `out` must be contiguous for both, and `M == 0` returns without a launch. Grid, `pick_block_dim`, LDS and the lack of occupancy attributes are unchanged.[^ln-cec6-k][^ln-cec6-c] The Python `gemma_rms_norm` op uses it for fp16 input and weight when `VLLM_RDNA_GEMMA_NORM_HIP` (default **1**, 0 = off) is set and the leading dims merge into one row stride; other dtypes and `[T, heads, D]` slices keep the eager cast/add/cast/`_C.rms_norm` path. The commit says the reduction order differs from `_C.rms_norm`, so results can differ in the last fp16 bit.[^ln-cec6-py][^ln-cec6-tune] *Historical:* up to `5bda4c24` the Gemma scale was always built on the host side (`(w.float() + 1.0).to(x.dtype)`) before `_C.rms_norm`, and the `rms_norm_kernel` read input rows at `row * N`.[^ln-5bda]

**At dest `13fbe2f9` (checked 2026-11-02):** `gated_rms_norm` accepts a strided `z`. `z` may be 2D `[M, N]` contiguous (as before) or 3D `[T, H, N]` with contiguous heads (`stride(2) == 1`, `stride(1) == N`) and any token stride, with `T * H == M`. The kernel gains `z_tok_stride` and `heads` arguments and row `r` reads `z` at `(r / heads) * z_tok_stride + (r % heads) * N`, so `z` can be a view into the packed qkvz projection with no copy. `input`, `weight` and `out` must still be contiguous. Tile, WG choice and the lack of occupancy attributes are unchanged.[^ln-13fb-k] `RMSNormGated.forward_hip` passes the 3D `z` through when its heads are contiguous and otherwise still reshapes and copies it.[^ln-13fb-py] *Historical:* up to `b30757c2` the op required 2D `z` with `input`'s exact shape and contiguous layout.[^ln-b307]

**At dest `1dc3bac4` (checked 2026-10-11):** `layernorm.cu` (with `gated_rms_norm`) is still on the unconditional list next to EXL3, not inside the gfx1030 block.[^ln-tip]


## dest tip 2026-09-11 — `opengfx1030/vllm-rdna` `rdna_extras` @ `71a54552`

Tip moved **`56f67111` → `71a54552`** (+3). Silicon: new **`gated_rms_norm`** AOT HIP for Qwen3.x GDN `RMSNormGated` (norm-before-gate). Same tile family as plain `rms_norm` — not a new occupancy subject. FA leftover still first. ticket-26 still EXL3 `-cb 3inst`. W4A16 ConfigA lock unchanged.[^c-56f67111][^c-71a54552][^f-56f67111-layernorm-cu]

Companion infra (Python, same tip window): Gemma `(1+w)` scale folded to `x.dtype` so `vllm_c` `rms_norm` dtype-match fires; per-replay cudagraph `isnan().any()` scan gated behind `VLLM_CG_NAN_INPUT_CHECK=1` (default off; *historical*: neither the scan nor the env exists in `vllm/compilation/cuda_graph.py` at `1dc3bac4`[^ln-tip-cg]) — [silicon/hip-craft/graph-capture.md](../../silicon/hip-craft/graph-capture.md).[^c-71a54552]

## Gated RMSNorm tile (`71a54552`)

| Knob | Value |
|---|---|
| File | `csrc/rocm/layernorm.cu` [^f-71a54552-layernorm-cu] |
| Op | `_rocm_C::gated_rms_norm` (`ops.h` + `torch_bindings.cpp`; out is `Tensor!`) [^f-71a54552-ops-h][^f-71a54552-torch-bindings-cpp][^f-71a54552-layernorm-cu] |
| Semantics | `y = x * rstd * weight * act(z)`; `rstd = rsqrt(mean(x^2)+eps)` [^c-71a54552] |
| act | `0` = silu (`z * sigmoid(z)`); `1` = sigmoid [^c-71a54552] |
| Grid | one CTA per row (`dim3 grid(M)`) [^c-71a54552] |
| WG | `BLOCK_DIM` 128 / 256 / 512 / 1024 via same `pick_block_dim(N)` [^f-71a54552-layernorm-cu] |
| Occupancy attr | **none** (no `__launch_bounds__`, no `waves_per_eu`) [^c-71a54552] |
| LDS | same `block_reduce_sum`: `__shared__ float s_partial[NUM_WARPS]` [^f-71a54552-layernorm-cu] |
| Inner | scalar `__half2float` + fp32 FMA + `expf` for gate. **Not** `fdot2` [^c-71a54552] |
| Reduce | `__shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, …)` then warp-0 LDS fold [^c-71a54552] |
| Dtype | fp16 in/`z`/weight/out only [^c-71a54552] |
| Python | `RMSNormGated.forward_hip` when ROCm + op present + `group_size is None` + `norm_before_gate` + fp16 + 2D + act in `{silu,swish,sigmoid}`; else `forward_native` [^c-71a54552] |
| `enabled()` | override returns True on ROCm-with-kernel so Inductor `custom_ops=none` does not leave the eager ~9-kernel path [^c-71a54552] |

Leave: Triton FLA gated-norm under cudagraph on RDNA; group-norm gated; norm-after-gate; bf16/fp32 gated path; tok/s claims; `__launch_bounds__` without VGPR dump.[^c-71a54552]

## Plain RMSNorm tile (sourced, `83de31cf` / tip `ea78104d`)

`layernorm.cu` on the dest CMake EXL3-unconditional list (same tile: one CTA/row, BLOCK_DIM 128/256/512/1024, LDS `NUM_WARPS` floats, no `fdot2`, no `__launch_bounds__`). Re-registered `rms_norm` / `fused_add_rms_norm` in `torch_bindings.cpp`.[^f-83de31cf-layernorm-cu][^f-83de31cf-torch-bindings-cpp]

| Knob | Value |
|---|---|
| File | `csrc/rocm/layernorm.cu` [^f-83de31cf-layernorm-cu] |
| CMake | appended with EXL3 (`exl3_dot2_*` / `exl3_hadamard.cu`), **not** inside `VLLM_ROCM_HAS_GFX1030` [^f-83de31cf-exl3-hadamard-cu] |
| Grid | one CTA per row (`dim3 grid(M)`) [^c-83de31cf][^c-ea78104d][^f-83de31cf-layernorm-cu] |
| WG | `BLOCK_DIM` template 128 / 256 / 512 / 1024 via `pick_block_dim(N)` [^f-83de31cf-layernorm-cu] |
| Occupancy attr | **none** |
| LDS | `__shared__ float s_partial[NUM_WARPS]` = 16 / 32 / 64 / 128 B [^c-83de31cf][^c-ea78104d][^f-83de31cf-layernorm-cu] |
| Inner | scalar `__half2float` + fp32 FMA. **Not** `fdot2` [^c-83de31cf][^c-ea78104d][^f-83de31cf-layernorm-cu] |
| Reduce | `__shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, …)` then warp-0 LDS fold [^c-83de31cf][^c-ea78104d][^f-83de31cf-layernorm-cu] |
| Dtype | fp16 in/out, fp32 accum, fp16 weight. fp16 only [^c-83de31cf][^c-ea78104d][^f-83de31cf-layernorm-cu] |
| API | `rms_norm(out, input, weight, eps)`; `fused_add_rms_norm` overwrites `input` in-place (CUDA-compatible) [^c-83de31cf][^c-ea78104d][^f-83de31cf-layernorm-cu] |

`pick_block_dim`: N≤512 → 128; N≤2048 → 256; N≤8192 → 512; else 1024.[^f-83de31cf-layernorm-cu]

HIP `__shfl_xor_sync` MaskT must be 64-bit (`f22a15b6`). `0xFFFFFFFFu` fails the static assert.[^c-f22a15b6]

Public entries are global-scope (`83de31cf` closed `vllm::rocm_layernorm` before them). Internal kernels stay in the namespace.[^c-83de31cf]

## Leave (family)

- Copying tok/s or inventing numbers
- Claiming cudagraph NaN is universally fixed (arenas + AOT help; debug scan is opt-in)[^c-83de31cf][^f-83de31cf-layernorm-cu]
- `fdot2` / half2 vectorize this lock (scalar loads; later HIP pass)[^c-83de31cf][^f-83de31cf-layernorm-cu]
- Leaving Gemma `(1+w)` scale in fp32 (silent fallback off `vllm_c`)[^c-83de31cf][^f-83de31cf-layernorm-cu]
- `__launch_bounds__` without a VGPR dump[^c-83de31cf][^f-83de31cf-layernorm-cu]
- Editing `rdna2_extras`[^c-83de31cf][^f-83de31cf-layernorm-cu]
- Occupancy card: still FA prefill LDS, not this kernel[^c-83de31cf][^f-83de31cf-layernorm-cu]

## Sources

[^c-56f67111]: opengfx1030/vllm-rdna commit [`56f67111`](https://github.com/opengfx1030/vllm-rdna/commit/56f67111dd87847c5e92e14a7076fec2c9ab6f22) (2026-09-11) — "cleanup(w4a16): drop measured-slower experiment configs, keep debug knobs".
[^c-71a54552]: opengfx1030/vllm-rdna commit [`71a54552`](https://github.com/opengfx1030/vllm-rdna/commit/71a54552ae8f6f8ad7f315c5a2291cebdb747690) (2026-09-11) — "perf: HIP gated_rms_norm kernel for Qwen3.x GDN RMSNormGated".
[^f-56f67111-layernorm-cu]: opengfx1030/vllm-rdna `csrc/rocm/layernorm.cu` @ [`56f67111`](https://github.com/opengfx1030/vllm-rdna/blob/56f67111dd87847c5e92e14a7076fec2c9ab6f22/csrc/rocm/layernorm.cu).
[^f-71a54552-layernorm-cu]: opengfx1030/vllm-rdna `csrc/rocm/layernorm.cu` @ [`71a54552`](https://github.com/opengfx1030/vllm-rdna/blob/71a54552ae8f6f8ad7f315c5a2291cebdb747690/csrc/rocm/layernorm.cu).
[^f-71a54552-ops-h]: opengfx1030/vllm-rdna `csrc/rocm/ops.h` @ [`71a54552`](https://github.com/opengfx1030/vllm-rdna/blob/71a54552ae8f6f8ad7f315c5a2291cebdb747690/csrc/rocm/ops.h).
[^f-71a54552-torch-bindings-cpp]: opengfx1030/vllm-rdna `csrc/rocm/torch_bindings.cpp` @ [`71a54552`](https://github.com/opengfx1030/vllm-rdna/blob/71a54552ae8f6f8ad7f315c5a2291cebdb747690/csrc/rocm/torch_bindings.cpp).
[^f-83de31cf-layernorm-cu]: BlivionIaG/vllm `csrc/rocm/layernorm.cu` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/layernorm.cu).
[^f-83de31cf-torch-bindings-cpp]: BlivionIaG/vllm `csrc/rocm/torch_bindings.cpp` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/torch_bindings.cpp).
[^f-83de31cf-exl3-hadamard-cu]: BlivionIaG/vllm `csrc/rocm/exl3_hadamard.cu` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/exl3_hadamard.cu).
[^c-83de31cf]: BlivionIaG/vllm commit [`83de31cf`](https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb) (2026-09-01) — "fix(rocm/layernorm): close namespace braces before global-scope public entries".
[^c-ea78104d]: opengfx1030/vllm-rdna commit [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a) (2026-09-03) — "fix(rocm): platform gates + MoE W8A16 parity (completes fb1687bde)".
[^c-f22a15b6]: BlivionIaG/vllm commit [`f22a15b6`](https://github.com/BlivionIaG/vllm/commit/f22a15b64bf4b3525dba8a7fca0379c3513cf5ca) (2026-09-01) — "fix(rocm/layernorm): 64-bit shfl mask + move public entries to global scope".
[^ln-tip-cg]: No `isnan` / `VLLM_CG_NAN_INPUT_CHECK` in `vllm/compilation/cuda_graph.py` at dest 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/compilation/cuda_graph.py
[^ln-tip]: `csrc/rocm/layernorm.cu` (`gated_rms_norm_kernel` L136) on the unconditional ROCm list next to EXL3 (`CMakeLists.txt` L1570-L1575, outside the `VLLM_ROCM_HAS_GFX1030` block that starts at L1517), dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/layernorm.cu#L136 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1517-L1575
[^ln-13fb-k]: `gated_rms_norm_kernel` signature and z addressing L139-L151, launcher L205-L215, host checks L287-L312, `csrc/rocm/layernorm.cu` at 13fbe2f9 (commit [`13fbe2f9`](https://github.com/opengfx1030/vllm-rdna/commit/13fbe2f9160027e71ab6c915ecbcb95d42373a2e), 2026-11-02). https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/csrc/rocm/layernorm.cu#L139-L151 ; https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/csrc/rocm/layernorm.cu#L287-L312
[^ln-13fb-py]: `vllm/model_executor/layers/layernorm.py` L478-L492, dest at 13fbe2f9. https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/vllm/model_executor/layers/layernorm.py#L478-L492
[^ln-b307]: `gated_rms_norm` host checks (`input and z must be 2D`, `input and z must match shapes`, contiguous) at b30757c2. https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/csrc/rocm/layernorm.cu
[^ln-cec6-c]: opengfx1030/vllm-rdna commit [`27e5fa40`](https://github.com/opengfx1030/vllm-rdna/commit/27e5fa40da6338dd2344927c97b3dad63990bf1f) (2026-11-02) "rdna: shared-expert gate_up tile and one-kernel Gemma norm (batch 4)"; dest tip `cec69c12`.
[^ln-cec6-k]: `rms_norm_kernel<BLOCK_DIM, kGemma>` L68-L100, launcher L190-L198, shared host checks and dispatch L235-L313, `csrc/rocm/layernorm.cu` at cec69c12. https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/csrc/rocm/layernorm.cu#L68-L100 ; https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/csrc/rocm/layernorm.cu#L235-L313
[^ln-cec6-py]: `_gemma_norm_hip`, `_norm_rows` and the HIP branch of `gemma_rms_norm`, `vllm/model_executor/layers/layernorm.py` L509-L560 at cec69c12. https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/vllm/model_executor/layers/layernorm.py#L509-L560
[^ln-cec6-tune]: `VLLM_RDNA_GEMMA_NORM_HIP` row, `docs/rdna2/tuning.md` L403 at cec69c12. https://github.com/opengfx1030/vllm-rdna/blob/cec69c12c80d1fdcd3fd9d7938809abf056d9b52/docs/rdna2/tuning.md#L403
[^ln-5bda]: `csrc/rocm/layernorm.cu` and `vllm/model_executor/layers/layernorm.py` at 5bda4c24. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/layernorm.cu ; https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/vllm/model_executor/layers/layernorm.py
