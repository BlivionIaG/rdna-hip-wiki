---
title: "W8A8-FP8 dense — silicon / HIP contract"
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/BlivionIaG/vllm/commit/750ca545a70c21f55dbc3680cd93c67e60a2f743
  - https://github.com/BlivionIaG/vllm/blob/750ca545a70c21f55dbc3680cd93c67e60a2f743/csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/qdq_fp8_rdna2.cuh
  - https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a
---
# W8A8-FP8 dense — silicon / HIP contract

## extras lock 2026-09-03 — `rdna_extras` @ `ea78104d`

`gemm_w8a8_fp8_dense_rdna2.cu`: activation scale now accepts per-(token, K-block) `a_scale` `[M, K/gk]` (DeepSeek V4 Flash dynamic FP8, block `[128,128]`) in addition to per-row / per-tensor. Template gains `PER_CHANNEL_SCALE` for fused-QKV weight scales. LDS tile unchanged: `__shared__ half block_a[M_TILE][BLOCK_KN_SIZE + LDS_PAD]` (`LDS_PAD=8`), inner still bit-trick E4M3→fp16 then `v_dot2_f32_f16`. Binding `gemm_w8a8_fp8_dense(..., a_scale_K_groups)` registered. **Not** a new DOT class. GPU correctness still as upstream commit says — do not invent numbers.[^f-ea78104d-gemm-w8a8-fp8-dense-rdna2-cu][^isa]

---

Live: `gemm_w8a8_fp8_dense_rdna2.cu` @ `750ca545`. **Incomplete** — commit says GPU correctness pending.[^c-750ca545][^f-750ca545-gemm-w8a8-fp8-dense-rdna2-cu]

Both A and B are E4M3 bytes. **Still `fdot2`.** Bit-trick both sides to fp16 (`fp8_e4m3_to_fp16_bits` in `qdq_fp8_rdna2.cuh`), then the W8A16-FP8 dequant helper. No LUT. No `sdot4`.[^f-ea78104d-qdq-fp8-rdna2-cuh]

This is **not** the spec W8A8 INT8 path. Extras vs contract: [w8a8.md](w8a8.md). Study: [w8a8-mxfp4.md](w8a8-mxfp4.md) (`sdot4`, i32 through K).[^c-ea78104d]

## Why not sdot4

sdot4 wants **integer** i8×i8. These bytes are E4M3. Integer DOT on FP8 bit patterns is wrong. Unpack → fp16 → `fdot2` is the only legal gfx1030 path until a real INT8 W8A8 kernel exists.[^isa][^f-750ca545-gemm-w8a8-fp8-dense-rdna2-cu]

## Done-when

- ISA dump: two E4M3 loads, bit-trick cvt, `v_dot2_f32_f16`. No `v_dot4c_i32_i8`.[^isa]
- GPU correctness on the shapes the launcher actually hits.[^isa][^f-750ca545-gemm-w8a8-fp8-dense-rdna2-cu]

## Sources

[^f-ea78104d-gemm-w8a8-fp8-dense-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu).
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^c-750ca545]: BlivionIaG/vllm commit [`750ca545`](https://github.com/BlivionIaG/vllm/commit/750ca545a70c21f55dbc3680cd93c67e60a2f743) (2026-08-13) — "feat(rocm): add native RDNA2 W8A8-FP8 dense GEMM for DeepSeek V4 Flash".
[^f-750ca545-gemm-w8a8-fp8-dense-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu` @ [`750ca545`](https://github.com/BlivionIaG/vllm/blob/750ca545a70c21f55dbc3680cd93c67e60a2f743/csrc/rocm/gemm_w8a8_fp8_dense_rdna2.cu).
[^f-ea78104d-qdq-fp8-rdna2-cuh]: opengfx1030/vllm-rdna `csrc/rocm/qdq_fp8_rdna2.cuh` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/qdq_fp8_rdna2.cuh).
[^c-ea78104d]: opengfx1030/vllm-rdna commit [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a) (2026-09-03) — "fix(rocm): platform gates + MoE W8A16 parity (completes fb1687bde)".
