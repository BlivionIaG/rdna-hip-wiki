# `rdna_extras` tip `ea78104d` — HIP / silicon delta (2026-09-03)

Dest: [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) branch **`rdna_extras`** @ **`ea78104d5fc8`**. Prior recorded tip: `8f2583d2`. Ahead by 23 commits.

## Take (silicon)

1. **FA INT8 fused decode** — `fa_rdna2.cu` templates `IS_INT8`; LDS fp16 + `fdot2`; decode keeps `waves_per_eu(4, 8)`. Writer `reshape_and_cache_int8_rdna2`. Details: [kernels/kv-int8.md](../../kernels/kv-int8.md), [silicon/fa-occupancy.md](../../silicon/fa-occupancy.md).
2. **FA INT8 prefill splitk** — new kernels; still `(N, 1)` occupancy trap.
3. **Sparse MLA HIP prefill** — `sparse_mla_prefill_kernel` `__launch_bounds__(32)`. [kernels/mla-sparse.md](../../kernels/mla-sparse.md).
4. **W8A8-FP8 dense** — per-block-K `a_scale`; same LDS + `fdot2`. [kernels/w8a8-fp8.md](../../kernels/w8a8-fp8.md).
5. **AOT RMSNorm on dest CMake** — `layernorm.cu` listed. [kernels/layernorm.md](../../kernels/layernorm.md).
6. **RDNA_ATTN / platform gates** — `VLLM_USE_RDNA2_FA=1`; head 128/256; `block_size >= 1`.

## Leave

- `VLLM_FORCE_CUSTOM_ALL_REDUCE` / amdsmi IndexError tolerance — AR fabric (UNC-27 class), not a dest kernel.
- EXL3 Python mul1 fold-to-dense, Hadamard DBG env, GDN ssm zeroing / debug logs.
- MXFP4 MoE Python routing (`rdna2_mxfp4_moe.py`) without a new `.cu` this window (bindings for existing `moe_mxfp4_gemm_rdna2` / `mxfp4_gemm_rdna2`).
- Editing the human branch. Copying tok/s. Claiming FA occupancy closed.

## Occupancy

Still first: FA prefill `__launch_bounds__(N, 1)` / LDS 1 WG per 64 KB. Decode pin already `(N)` + `(4, 8)`.
