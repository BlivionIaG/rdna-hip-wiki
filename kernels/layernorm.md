# HIP AOT RMSNorm — extras `83de31cf` + gated `71a54552`

## dest tip 2026-09-11 — `opengfx1030/vllm-rdna` `rdna_extras` @ `71a54552`

Tip moved **`56f67111` → `71a54552`** (+3). Silicon: new **`gated_rms_norm`** AOT HIP for Qwen3.x GDN `RMSNormGated` (norm-before-gate). Same tile family as plain `rms_norm` — not a new occupancy subject. FA leftover still first. ticket-26 still EXL3 `-cb 3inst`. W4A16 ConfigA lock unchanged.

Companion infra (Python, same tip window): Gemma `(1+w)` scale folded to `x.dtype` so `vllm_c` `rms_norm` dtype-match fires; per-replay cudagraph `isnan().any()` scan gated behind `VLLM_CG_NAN_INPUT_CHECK=1` (default off) — [../silicon/graph-capture.md](../silicon/graph-capture.md).

## Gated RMSNorm tile (`71a54552`)

| Knob | Value |
|---|---|
| File | `csrc/rocm/layernorm.cu` |
| Op | `_rocm_C::gated_rms_norm` (`ops.h` + `torch_bindings.cpp`; out is `Tensor!`) |
| Semantics | `y = x * rstd * weight * act(z)`; `rstd = rsqrt(mean(x^2)+eps)` |
| act | `0` = silu (`z * sigmoid(z)`); `1` = sigmoid |
| Grid | one CTA per row (`dim3 grid(M)`) |
| WG | `BLOCK_DIM` 128 / 256 / 512 / 1024 via same `pick_block_dim(N)` |
| Occupancy attr | **none** (no `__launch_bounds__`, no `waves_per_eu`) |
| LDS | same `block_reduce_sum`: `__shared__ float s_partial[NUM_WARPS]` |
| Inner | scalar `__half2float` + fp32 FMA + `expf` for gate. **Not** `fdot2` |
| Reduce | `__shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, …)` then warp-0 LDS fold |
| Dtype | fp16 in/`z`/weight/out only |
| Python | `RMSNormGated.forward_hip` when ROCm + op present + `group_size is None` + `norm_before_gate` + fp16 + 2D + act in `{silu,swish,sigmoid}`; else `forward_native` |
| `enabled()` | override returns True on ROCm-with-kernel so Inductor `custom_ops=none` does not leave the eager ~9-kernel path |

Leave: Triton FLA gated-norm under cudagraph on RDNA; group-norm gated; norm-after-gate; bf16/fp32 gated path; tok/s claims; `__launch_bounds__` without VGPR dump.

## Plain RMSNorm tile (sourced, `83de31cf` / tip `ea78104d`)

`layernorm.cu` on the dest CMake EXL3-unconditional list (same tile: one CTA/row, BLOCK_DIM 128/256/512/1024, LDS `NUM_WARPS` floats, no `fdot2`, no `__launch_bounds__`). Re-registered `rms_norm` / `fused_add_rms_norm` in `torch_bindings.cpp`.

| Knob | Value |
|---|---|
| File | `csrc/rocm/layernorm.cu` |
| CMake | appended with EXL3 (`exl3_dot2_*` / `exl3_hadamard.cu`), **not** inside `VLLM_ROCM_HAS_GFX1030` |
| Grid | one CTA per row (`dim3 grid(M)`) |
| WG | `BLOCK_DIM` template 128 / 256 / 512 / 1024 via `pick_block_dim(N)` |
| Occupancy attr | **none** |
| LDS | `__shared__ float s_partial[NUM_WARPS]` = 16 / 32 / 64 / 128 B |
| Inner | scalar `__half2float` + fp32 FMA. **Not** `fdot2` |
| Reduce | `__shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, …)` then warp-0 LDS fold |
| Dtype | fp16 in/out, fp32 accum, fp16 weight. fp16 only |
| API | `rms_norm(out, input, weight, eps)`; `fused_add_rms_norm` overwrites `input` in-place (CUDA-compatible) |

`pick_block_dim`: N≤512 → 128; N≤2048 → 256; N≤8192 → 512; else 1024.

HIP `__shfl_xor_sync` MaskT must be 64-bit (`f22a15b6`). `0xFFFFFFFFu` fails the static assert.

Public entries are global-scope (`83de31cf` closed `vllm::rocm_layernorm` before them). Internal kernels stay in the namespace.

## Leave (family)

- Copying tok/s or inventing numbers
- Claiming cudagraph NaN is universally fixed (arenas + AOT help; debug scan is opt-in)
- `fdot2` / half2 vectorize this lock (scalar loads; later HIP pass)
- Leaving Gemma `(1+w)` scale in fp32 (silent fallback off `vllm_c`)
- `__launch_bounds__` without a VGPR dump
- Editing `rdna2_extras`
- Occupancy card: still FA prefill LDS, not this kernel
