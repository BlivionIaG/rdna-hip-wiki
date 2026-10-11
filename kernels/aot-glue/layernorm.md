---
title: "HIP AOT RMSNorm — extras `83de31cf` + gated `71a54552`"
sources:
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

## dest tip 2026-09-11 — `opengfx1030/vllm-rdna` `rdna_extras` @ `71a54552`

Tip moved **`56f67111` → `71a54552`** (+3). Silicon: new **`gated_rms_norm`** AOT HIP for Qwen3.x GDN `RMSNormGated` (norm-before-gate). Same tile family as plain `rms_norm` — not a new occupancy subject. FA leftover still first. ticket-26 still EXL3 `-cb 3inst`. W4A16 ConfigA lock unchanged.[^c-56f67111][^c-71a54552][^f-56f67111-layernorm-cu]

Companion infra (Python, same tip window): Gemma `(1+w)` scale folded to `x.dtype` so `vllm_c` `rms_norm` dtype-match fires; per-replay cudagraph `isnan().any()` scan gated behind `VLLM_CG_NAN_INPUT_CHECK=1` (default off) — [silicon/graph-capture.md](../../silicon/graph-capture.md).[^c-71a54552]

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
