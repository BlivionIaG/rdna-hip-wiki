---
title: "NVFP4 on gfx1030 — silicon / HIP contract"
sources:
  - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/BlivionIaG/vllm/blob/add17dd7b35942748fae68f7a7cec939c46d0b82/csrc/rocm/mxfp4_dot2_common.cuh
  - https://github.com/BlivionIaG/vllm/blob/add17dd7b35942748fae68f7a7cec939c46d0b82/csrc/rocm/qdq_fp8_rdna2.cuh
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/clang/include/clang/Basic/BuiltinsAMDGPU.td#L448-L459
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292
  - https://github.com/ggml-org/llama.cpp/issues/24438
  - https://github.com/BlivionIaG/vllm/blob/add17dd7b35942748fae68f7a7cec939c46d0b82/csrc/rocm/mxfp4_dot2_dense.cu
  - https://github.com/BlivionIaG/vllm/commit/add17dd7b35942748fae68f7a7cec939c46d0b82
  - https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
---
# NVFP4 on gfx1030 — silicon / HIP contract

Engine dispatch and checkpoint layout: [engine/quant-formats/nvfp4.md](../../engine/quant-formats/nvfp4.md). This page is the ISA and tile. Check a future kernel against the list at the bottom.[^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh]

**There is no NVFP4 unit on gfx1030.** Inner op is unpack E2M1 → fp16, apply E4M3 (× optional FP32), then `fdot2`. Same DOT as live mxfp4. The delta is the **scale**, not a new instruction.[^nvfp4-blog]

## Native ops we actually have

| Need | Instruction | HIP [^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh] |
|---|---|---|
| 2-wide fp16 inner product | `V_DOT2_F32_F16` / `V_DOT2C` | `__builtin_amdgcn_fdot2(a, b, acc, false)` [^isa] |
| E2M1 nibble → fp16 bits | VALU integer | reuse `mxfp4_e2m1_to_fp16_bits` in `mxfp4_dot2_common.cuh` [^f-add17dd7-mxfp4-dot2-common-cuh] |
| E4M3 → fp16 | bit-cast / existing FP8 LUT | reuse `qdq_fp8_rdna2.cuh` (W8A16-FP8 already does this) [^f-add17dd7-qdq-fp8-rdna2-cuh] |
| Scale apply | `v_mul_f16` / `v_mul_f32` | **multiply**. Not an exponent add. [^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh] |
| Optional tensor scale | one `fmul` in the epilogue or folded into the E4M3 | FP32 scalar / `[N]` [^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh] |

## Do not use

| Op | Why |
|---|---|
| `V_DOT8_I32_I4` / `sdot8` | Signed i4 ≠ E2M1 codebook `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` [^isa] |
| `mxfp4_apply_e8m0_bits` | E8M0 is `2^(s8-127)` (add to exponent). NVFP4 scale is **E4M3**, a real multiply. [^f-add17dd7-mxfp4-dot2-common-cuh][^ocp-mx][^nvfp4-blog] |
| `v_dot2_f32_bf16` / bf16 | Absent on gfx1030. Host `kFloat16` only. [^isa][^clang-dot][^llvm-gfx1030] |
| `sdot4` | Wrong type. This is not W8A8. [^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh] |
| CUTLASS / Marlin / FlashInfer / Blackwell swizzle | Dead on this silicon. Linear scales only. [^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh] |

## On-the-fly dequant (VGPR)

Per thread, one dword of B = 8 E2M1 nibbles (same pack as mxfp4: `uint32 [K/8, N]`).[^nvfp4-blog][^isa][^f-add17dd7-mxfp4-dot2-common-cuh]

```
// 8 K of B
half2 dq[4] = e2m1_unpack8(b_dword);          // reuse mxfp4 LUT
half  s0    = e4m3_to_f16(scale[k/16]);       // first 16
half  s1    = e4m3_to_f16(scale[k/16 + 1]);   // next 16  (two scales / 32-K window)
dq[0] *= s0; dq[1] *= s0;                     // K+0..7  if this dword is in the first 16
// … or s1 if the dword sits in the second 16
acc = fdot2(dq[0], a0, acc);
acc = fdot2(dq[1], a1, acc);
acc = fdot2(dq[2], a2, acc);
acc = fdot2(dq[3], a3, acc);
```

Two E4M3 scales per 32-K window (`K % 16 == 0`). Fold the optional FP32 tensor scale into `s0/s1` on the host **or** one epilogue `fmul` — do not re-read it in the inner loop.[^nvfp4-blog]

Dequant **in VGPR**. Stage **A** (fp16) in LDS. Stream packed B. Do **not** park a 16-entry nibble LUT in LDS/K$ (serializes; llama.cpp #24438).[^lc-24438]

## Tiles

Copy mxfp4 / W4A16, not Blackwell.[^isa][^lc-24438]

| Regime | Seed | LDS [^isa][^lc-24438] |
|---|---|---|
| Decode `M=1,2,4,8` | 256 threads, `BLOCK_KN=256`, 4 N/thread, K-split `gridDim.z`, A in LDS | `K·M·2` fp16. Cap `K·M ≤ 32768` elems (64 KB) [^isa][^lc-24438] |
| Prefill `M≥32` | `64×64×32` fp16-after-unpack, WG=256 | A 4 KB + B packed 4 KB ≈ 8 KB [^isa][^lc-24438] |
| Occupancy | `amdgpu_waves_per_eu(4, 8)`. No `__launch_bounds__(*, 1)` | 64–256 VGPR [^isa][^lc-24438] |

Bank walk is dword-based. Sequential `int` along K, wave32, 64 banks: conflict-free.[^isa]

W4A4 checkpoint: dequant A to fp16 (ignore act key) or refuse. Same `fdot2` kernel. No software FP4×FP4 DOT.[^isa][^lc-24438]

## Clone list (from live tree @ `add17dd7`)

| Copy | Change |
|---|---|
| `mxfp4_dot2_common.cuh` | keep E2M1 unpack + 4×`fdot2`; **replace** E8M0 apply with E4M3→fp16 + `fmul` [^f-add17dd7-mxfp4-dot2-common-cuh][^ocp-mx] |
| `mxfp4_dot2_dense.cu` / `_moe.cu` | same control flow; scale tensor is `[K/16, N]` E4M3 [^f-add17dd7-mxfp4-dot2-dense-cu] |
| `qdq_fp8_rdna2.cuh` | E4M3→fp16 helper already exists for W8A16-FP8 [^f-add17dd7-qdq-fp8-rdna2-cuh] |

Name: `nvfp4_dot2_common.cuh`, `nvfp4_gemm_rdna2`, `moe_nvfp4_gemm_rdna2`.[^c-add17dd7]

## Done-when (ISA dump)

- [ ] Inner loop is `v_dot2_f32_f16` / `v_dot2c_f32_f16`[^isa]
- [ ] No `v_dot8_*`, no `v_dot4_*` on the weight path[^isa]
- [ ] E4M3 scale is a mul, not `mxfp4_apply_e8m0_bits`[^f-add17dd7-mxfp4-dot2-common-cuh][^nvfp4-blog]
- [ ] Host `kFloat16`
- [ ] `waves_per_eu(4, 8)` on decode-class
- [ ] GPU smoke vs CPU `e2m1 * e4m3 * f32` reference
- [ ] No tok/s claimed from this page


Petit (CDNA NVFP4/MXFP4 reference): [silicon/petit-kernel.md](../../silicon/petit-kernel.md) — Take offline shuffle + denorm caveat; Leave MatrixCore.[^nvfp4-blog]

## Sources

- NVIDIA NVFP4: E2M1 + E4M3 / 16 + optional FP32 — [blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
- Live mxfp4: `csrc/rocm/mxfp4_dot2_common.cuh` @ `add17dd7` (4× `fdot2` / 8 K)
- Live FP8 dequant: `csrc/rocm/qdq_fp8_rdna2.cuh`
- RDNA 2 ISA 70648: `V_DOT2_F32_F16`; no FP4 MMA; no `v_dot2_f32_bf16`
- [engine/quant-formats/nvfp4.md](../../engine/quant-formats/nvfp4.md), [kernels/quant/w8a8-mxfp4.md](w8a8-mxfp4.md), [silicon/hip-craft.md](../../silicon/hip-craft.md)

[^nvfp4-blog]: NVIDIA, *Introducing NVFP4 for efficient and accurate low-precision inference*. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^f-add17dd7-mxfp4-dot2-common-cuh]: BlivionIaG/vllm `csrc/rocm/mxfp4_dot2_common.cuh` @ [`add17dd7`](https://github.com/BlivionIaG/vllm/blob/add17dd7b35942748fae68f7a7cec939c46d0b82/csrc/rocm/mxfp4_dot2_common.cuh).
[^f-add17dd7-qdq-fp8-rdna2-cuh]: BlivionIaG/vllm `csrc/rocm/qdq_fp8_rdna2.cuh` @ [`add17dd7`](https://github.com/BlivionIaG/vllm/blob/add17dd7b35942748fae68f7a7cec939c46d0b82/csrc/rocm/qdq_fp8_rdna2.cuh).
[^ocp-mx]: Open Compute Project, *OCP Microscaling Formats (MX) v1.0 Specification* — MXFP4 = E2M1 elements with one shared E8M0 scale per 32-element block. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
[^clang-dot]: Clang `BuiltinsAMDGPU.td` @ llvm-project `eb61ef62df` L448–L459: `fdot2` = `dot10-insts`, `fdot2_f32_bf16` = `dot12-insts`, `sdot4`/`sdot8` = `dot1-insts`, `udot4`/`udot8` = `dot7-insts`, `sudot4`/`sudot8` = `dot8-insts`. <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/clang/include/clang/Basic/BuiltinsAMDGPU.td#L448-L459>
[^llvm-gfx1030]: LLVM `AMDGPU.td` @ `eb61ef62df` L2278–L2292, `FeatureISAVersion10_3_0` (gfx1030): `FeatureDot1/2/5/6/7/10Insts`, no `Dot8`/`Dot12`, `FeatureMaxWavesPerEU16`. <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292>
[^lc-24438]: llama.cpp issue #24438 — ROCm/HIP MoE token-gen at ~40% bandwidth on gfx1151 (per-lane nibble LUT serialization). <https://github.com/ggml-org/llama.cpp/issues/24438>
[^f-add17dd7-mxfp4-dot2-dense-cu]: BlivionIaG/vllm `csrc/rocm/mxfp4_dot2_dense.cu` @ [`add17dd7`](https://github.com/BlivionIaG/vllm/blob/add17dd7b35942748fae68f7a7cec939c46d0b82/csrc/rocm/mxfp4_dot2_dense.cu).
[^c-add17dd7]: BlivionIaG/vllm commit [`add17dd7`](https://github.com/BlivionIaG/vllm/commit/add17dd7b35942748fae68f7a7cec939c46d0b82) (2026-08-17) — "feat(rocm): fp16 sparse MLA Triton kernels + indexer int64 fix for gfx1030".
