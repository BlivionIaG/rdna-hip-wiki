---
title: "HIP sparse MLA — silicon / HIP contract"
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/sparse_mla_rdna2.cu
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/torch_bindings.cpp
  - https://github.com/BlivionIaG/vllm/commit/9a344444645f34b3fb7306fde78d68df2c716078
  - https://github.com/BlivionIaG/vllm/blob/9a344444645f34b3fb7306fde78d68df2c716078/csrc/rocm/sparse_mla_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/fa_rdna2.cu
---
# HIP sparse MLA — silicon / HIP contract

## extras lock 2026-09-03 — `rdna_extras` @ `ea78104d`

HIP **prefill** landed in `csrc/rocm/sparse_mla_rdna2.cu` (was Triton-only on earlier tips).[^f-ea78104d-sparse-mla-rdna2-cu]

| Piece | Value |
|---|---|
| Kernel | `sparse_mla_prefill_kernel` — `__launch_bounds__(32)` (1 wave32), grid `(T, H/HEADS_PER_CTA)` [^f-ea78104d-sparse-mla-rdna2-cu] |
| KV | Plain `half` / `__hip_bfloat16` rows `[skv, COMB_DIM]` — **no** FP8 slots / E8M0 (fp8_ds_mla is decode-cache only) [^ocp-mx] |
| LDS | Double-buffered `__shared__ scalar_t s_kv[2][COMB_DIM]` [^c-ea78104d] |
| QK / PV | Still **scalar fp32 FMA** (same as decode). Not `fdot2` yet [^c-ea78104d] |
| Bindings | `sparse_mla_prefill_rdna2` + decode already registered in `torch_bindings.cpp` [^f-ea78104d-torch-bindings-cpp][^f-ea78104d-sparse-mla-rdna2-cu] |

Decode contract unchanged. Prefill is no longer “still Triton” on this dest tip. Occupancy leftover remains FA prefill, not MLA’s 32-thread WG.[^c-ea78104d]

---

Live: `sparse_mla_rdna2.cu` @ `9a344444`. Engine: [engine/serving-stack/fork-delta.md](../../engine/serving-stack/fork-delta.md). **Incomplete.** Env `VLLM_USE_RDNA2_MLA=1`. Prefill still Triton.[^c-9a344444][^f-9a344444-sparse-mla-rdna2-cu]

## ISA (confirmed)

| Piece | What it is |
|---|---|
| q / out | templated `half` on gfx1030. No fp16→bf16 cast. [^c-9a344444][^f-9a344444-sparse-mla-rdna2-cu] |
| K_nope | E4M3 + E8M0 / 64-ch. `__hip_cvt_fp8_to_halfraw` then `* exp2f(sc-127)` [^ocp-mx] |
| K_rope | **bf16 in the slot** (cache contract). `__bfloat162float` [^c-9a344444][^f-9a344444-sparse-mla-rdna2-cu] |
| QK / PV | **scalar fp32 FMA**, not `fdot2` [^c-9a344444][^f-9a344444-sparse-mla-rdna2-cu] |
| Occupancy | `__launch_bounds__(32)` only — no `(1,1)` trap [^c-9a344444][^f-9a344444-sparse-mla-rdna2-cu] |
| Grid | `(B, H/4)`, H multiple of 4 (K128=16, K256=64) [^c-9a344444][^f-9a344444-sparse-mla-rdna2-cu] |

Unpack already lands in `half`; 14+2 dims/thread are even. Later upgrade: stop promoting, `fdot2` the pairs. Not a new ticket. Occupancy still first for `fa_rdna2`.[^f-ea78104d-fa-rdna2-cu]

## Sources

[^f-ea78104d-sparse-mla-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/sparse_mla_rdna2.cu` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/sparse_mla_rdna2.cu).
[^ocp-mx]: Open Compute Project, *OCP Microscaling Formats (MX) v1.0 Specification* — MXFP4 = E2M1 elements with one shared E8M0 scale per 32-element block. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
[^c-ea78104d]: opengfx1030/vllm-rdna commit [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a) (2026-09-03) — "fix(rocm): platform gates + MoE W8A16 parity (completes fb1687bde)".
[^f-ea78104d-torch-bindings-cpp]: opengfx1030/vllm-rdna `csrc/rocm/torch_bindings.cpp` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/torch_bindings.cpp).
[^c-9a344444]: BlivionIaG/vllm commit [`9a344444`](https://github.com/BlivionIaG/vllm/commit/9a344444645f34b3fb7306fde78d68df2c716078) (2026-08-17) — "feat(rocm): dtype-templated + H-generic HIP sparse MLA decode".
[^f-9a344444-sparse-mla-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/sparse_mla_rdna2.cu` @ [`9a344444`](https://github.com/BlivionIaG/vllm/blob/9a344444645f34b3fb7306fde78d68df2c716078/csrc/rocm/sparse_mla_rdna2.cu).
[^f-ea78104d-fa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/fa_rdna2.cu` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/fa_rdna2.cu).
