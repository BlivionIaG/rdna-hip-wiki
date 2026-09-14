# Qwen3.8-Flash-Next / Qwen4Exp HIP — tip `5c3c0c6f`

Dest: `opengfx1030/vllm-rdna` `rdna_extras` @ **`5c3c0c6f`** (2026-09-14). Flash-Next / Qwen4Exp gfx1030 HIP opt-in paths via PR #6/#8 stack. No tok/s. Occupancy still first.

CMake gfx1030 `EXT_SRC` now includes: `rdna_fused_glue.cu`, `hc_rdna2.cu`, `qsa_rdna2.cu`, `ple_short_conv_rdna2.cu`, `mrope_rdna2.cu` (plus prior list).

## Take (silicon)

### T46 `rdna_fused_glue.cu` — fused decode glue (M ≤ 8)

- Explicit `__builtin_amdgcn_fdot2` on fp16 and int8→half2 weight rows (wave32 lanes stride K).
- `__launch_bounds__(WAVES * 32)`; tiny `__shared__ float s_gate[kMaxM]` on one kernel.
- Opt-in: `VLLM_RDNA_FUSED_HC=1` + `on_gfx10x()`. Default off.
- No WMMA. Real DOT path for Flash-Next decode glue.

### T47 `hc_rdna2.cu` — HyperConnection prefill glue

- Five elementwise/affine kernels (grouped Gemma RMSNorm, silu scaler, gate mix, combine, combine+norm).
- vec8 fp16 (uint4) loads; scalar/half2 FMA only — **no fdot2**.
- Opt-in: `VLLM_RDNA_HC_PREFILL_HIP=1` + `on_gfx10x()`. Default off (Triton).

### T48 `qsa_rdna2.cu` — QSA decode glue

- `qsa_store_cache_rows` / `qsa_compress_groups` scalar; MQA host wrapper reuses existing `paged_mqa_logits_decode_rdna2` when layout matches.
- Opt-in: `VLLM_RDNA_QSA_HIP=1` + `on_gfx10x()`. Default off.

### T49 `ple_short_conv_rdna2.cu` — dilated PLE short-conv

- Depthwise dilated conv1d decode+prefill; per-channel threads; `history_buf[64]` in regs; scalar FMA + silu.
- Opt-in: `VLLM_RDNA_PLE_CONV_HIP=1` + `on_gfx10x()`. Default off (F.conv1d).

### `mrope_forward_rdna2` build fixes

- Already room-locked Take (Qwen M-RoPE bind). Tip adds CUDAGuard include + host wrapper at global scope so `torch_bindings` resolves. Still one CTA/token, scalar, no fdot2.

## Leave

- Do not treat fused_glue / HC / QSA / PLE defaults as on — `VLLM_RDNA_FUSED_HC`, `VLLM_RDNA_HC_PREFILL_HIP`, `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_PLE_CONV_HIP` are all env-gated opt-in (default off) until soak.
- No transplant of HC/PLE onto GLM KDA / norm-after-gate (same split as gated_rms_norm / causal_conv).
- No WMMA/FP8 objects; gfx1030 stays fdot2 / EXL3.

## Occupancy

FA prefill leftover still first (`__launch_bounds__(*, 1)`). ticket-26 EXL3 `-cb 3inst` unchanged.
