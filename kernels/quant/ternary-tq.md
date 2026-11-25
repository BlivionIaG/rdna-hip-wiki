---
title: "Ternary TQ1_0 / TQ2_0 on HIP: decode, sdot4 and occupancy on gfx1030 (stew675 block 17)"
sources:
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/ffdf416993a51f4abb883b67b70fb1c457dcaddd
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/fcd77def387026d173e8b4182733955e241b3eb7
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/178efe846235854880b552aa8d878a8b0a492ab8
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/common.cuh
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/vendors/hip.h
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/ggml-cuda.cu
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-common.h
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/commit/5290ac44934128d45e8b42014db05e0bfa5bde05
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1150-L1161
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_device.cpp#L712-L715
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://godbolt.org/
---
# Ternary TQ1_0 / TQ2_0 on HIP: decode, sdot4 and occupancy on gfx1030

Source: stew675/llama-cpp-rdna-boosts r356, **block 17**, "TQ1_0 / TQ2_0 (ternary) weights on the GPU". It is contributor PR #358 by @overdoingism, merged in `fcd77def` and promoted in `ffdf4169`. `178efe84` then fixed the gate script.[^r356][^merge358][^gate-fix] The block was measured on gfx1201, with gfx1100 and gfx1151 neutrality arms. **No gfx1030 arm exists.**[^r356][^p-header] Back to the [quant index](README.md). Closest DOT page: [sdot4-explore.md](sdot4-explore.md).

This page is about **occupancy first**, then the decode and DOT path. It gives no throughput numbers.

## Verdicts

| Verdict | Claim |
|---|---|
| **Take (fact)** | On gfx1030 every TQ matmul with more than one column, prefill included, goes to the **small-batch dp4a kernel**. One column goes to the **vec kernel**. The int8-WMMA kernel only runs when `GGML_CUDA_CC_IS_RDNA4(cc)` is true, and its body compiles only under `AMD_WMMA_AVAILABLE && RDNA4`. TQ types do not use MMQ or MMVQ: `ggml_cuda_should_use_mmvq` returns false for them, and `ggml_cuda_mul_mat` branches to `tq.cu` first.[^p-launch][^p-mma-guard][^p-mmvq][^p-dispatch] |
| **Take (ISA, CE run)** | `ggml_cuda_dp4a` on RDNA2 is `__builtin_amdgcn_sdot4(a, b, c, false)`. For gfx1030 it lowers to **`v_dot4c_i32_i8`**, the VOP2 form with the accumulator tied to the destination. The VOP3 `v_dot4_i32_i8` form does not appear. There are 8 per (block, column, lane), chained on one accumulator.[^lc-dp4a][^lc-rdna2][^ce][^isa] |
| **Take (ISA, CE run)** | Decoding needs **no LUT, no LDS table and no `v_perm`**. TQ2_0 is `v_lshrrev_b32` plus `v_and_b32 0x03030303`, giving four 2-bit trits per dword straight into the sdot4 operand. TQ1_0 extracts base-3 digits by multiplying by 3 inside two 16-bit fields per dword, which shows up as `v_mul_u32_u24` plus `v_and_or_b32`. The TQ1_0 small kernel body has 44 static `v_mul_u32_u24`. The TQ2_0 body has none.[^p-decode][^ce] |
| **Take (occupancy, CE run)** | Small-batch kernel: **65 VGPRs (TQ2_0) / 72 (TQ1_0)**, allocated as 80, which gives **12 waves/SIMD**. TQ2_0 is **one VGPR** past the 64-VGPR line that gives 16 waves/SIMD. It uses no LDS and no scratch. Vec kernel: 41–57 VGPRs, which gives 16 waves/SIMD from VGPRs alone.[^ce][^llvm-baseinfo] |
| **Take (occupancy, arithmetic)** | The vec kernel's real ceiling is its **grid**, not VGPRs or LDS. It launches 512 threads (16 wave32 waves) per block and at most `2 × nsm` blocks. On gfx1030 `nsm` = `multiProcessorCount` counts **WGPs** (36 on a 72-CU V620). That is at most 72 blocks, about 2 per WGP, so **≤ 8 waves/SIMD out of 16** if the blocks are spread evenly. Even spreading is an assumption (unsourced). The cap is on purpose: fewer blocks means fewer copies of the src1 quantize.[^p-vec-defs][^p-vec-grid][^lc-nsm][^clr-cu-halve][^clr-props] |
| **Take (numerics, CE run)** | `#pragma clang fp contract(off)` holds in gfx1030 codegen. The block accumulate is `v_cvt_f32_i32`, then `v_mul_f32`, then `v_add_f32`. The product of the f16 scale and the f32 Q8 scale becomes `v_fma_mix_f32 … neg(0)`: an FMA with a −0 addend, which is one correctly rounded product, not a contraction. FMAs appear only in the `__fdiv_rn` expansion inside the src1 quantize.[^p-fmul][^ce] |
| **Leave** | Porting this to dest/extras. It is a llama.cpp GGUF path for BitNet-style ternary models. vLLM `rdna_extras` has no ternary format, and this block has no `MUL_MAT_ID`, so TQ MoE experts stay on the CPU.[^p-header][^p-supports] |
| **Later (measure)** | Whether a `__launch_bounds__(256)` / `amdgpu_waves_per_eu` hint gets the TQ2_0 small kernel from 65 to ≤ 64 VGPRs (12 → 16 waves/SIMD) without spilling. **Not tried.** It also only matters if the kernel is occupancy-bound and not VMEM-bound, which nobody has measured (unsourced). |

## Formats (what is decoded)

| Type | Block (256 values) | Stored trit | Per-thread decode in the kernel |
|---|---|---|---|
| TQ2_0 | `qs[64]` 2 bits each + `d` (fp16) = 66 B | 0..2, value = (trit − 1)·d | `(w >> 2l) & 0x03030303`, l = 0..3, on two dwords (`qs[4L..]`, `qs[32+4L..]`) |
| TQ1_0 | `qs[48]` base-3, 5 trits/byte + `qh[4]` 4 trits/byte + `d` = 54 B | 0..2 | digit n of byte b = `((uint8)(b·3ⁿ)·3) >> 8`. It is done on even and odd bytes in two 16-bit fields at once: b·3ⁿ < 2¹⁵, so no carry crosses fields |

Sizes: `ggml-common.h` static_asserts.[^lc-common-tq] Decode formulas: block-17 `tq.cu` / `common.cuh`.[^p-trit][^p-decode]

The trits stay **unsigned 0..2** in the sdot4 operand. The −1 offset is applied once per lane as `s = -bsum`, where bsum is the Q8_K sum of that lane's 32 values. So signed `sdot4` is exact here: 0..2 fits in a signed i8. RDNA3/4 `sudot4(true, …, true, …)` computes the same signed×signed product.[^p-lanedot][^lc-dp4a]

```
// per CPU lane L (8 threads per weight row), per 256-value block, per src1 column
s  = -bsum[L]
s  = v_dot4c_i32_i8(t[k], q8[32L+4k .. +3], s)   // k = 0..7, one accumulator chain
acc = acc + (float)s * (y.d * x.d)               // mul then add, never fma (bit-identity with ggml-cpu AVX2)
// end: hsum over the 8 lanes in hsum_float_8 order (shfl_xor 4, 2, 1)
```

The lane layout copies the CPU backend's eight 32-bit AVX2 lanes: lane L holds the values whose index mod 32 is 4L..4L+3. Src1 is quantized to a lane-major `block_q8_tq` (256 i8 + 8 i32 bsum + f32 d + pad = **304 B**).[^p-header][^p-q8tq]

## Occupancy on gfx1030 (wave32, WGP mode)

Budget, from existing pages: 1024 VGPRs per SIMD32 (wave32), allocation granule 16, at most 16 waves/SIMD. LDS is 128 KB per WGP with at most 64 KB per workgroup.[^llvm-baseinfo][^isa] See [architecture.md](../../silicon/arch-and-memory/architecture.md) §2.4 and [lds-occupancy.md](../../silicon/lds-and-tiles/lds-occupancy.md).

| Kernel (gfx1030, CE) | Threads/WG | VGPR (alloc) | LDS | Scratch / VGPR spill | Waves/SIMD (VGPR) | Binding limit |
|---|---|---|---|---|---|---|
| `quantize_q8_tq` | 32 × 4 | 23 (32) | 0 | 0 / 0 | 16 | — |
| `mul_mat_tq_small<TQ2_0>` | 256 | **65 (80)** | 0 | 0 / 0 | **12** | VGPR (1 over the 64 cliff) |
| `mul_mat_tq_small<TQ1_0>` | 256 | **72 (80)** | 0 | 0 / 0 | **12** | VGPR |
| `mul_mat_tq_vec<TQ2_0>` / `<TQ1_0>` | 512 | 48 / 44 | dyn. `nb × 304` B ≤ 48 KB | 0 / 0 | 16 | **grid ≤ 2·nsm → ≤ 8/SIMD** |
| `mul_mat_tq_vec<…, glu>` (gate+up+SwiGLU) | 512 | 41 / 57 | same | 0 / 0 | 16 | same |

VGPR, LDS-static, scratch and the "Occupancy" comment come from the CE run.[^ce] The 65 → 80 → 12 arithmetic is `floor(1024 / alignTo(VGPR, 16))`, capped at 16. The compiler's own `; Occupancy:` line agrees (12 / 12 / 16 …).[^llvm-baseinfo][^ce]

**Vec-kernel LDS.** Each block quantizes the whole src1 row into dynamic LDS: `nb × 304` B, with nb = K/256. The kernel only runs while that is ≤ `TQ_VEC_SMEM` = 48 KB, so nb ≤ 161 and K ≤ 41216. Bigger K falls back to quantize + small kernel.[^p-vec-defs][^p-vec-ok] At the 48 KB cap, the 128 KB WGP pool holds 2 blocks, the same as the grid cap. Below 32 KB (K ≤ 27392) it holds 4. So **LDS never binds before the grid cap** on gfx1030 (arithmetic from the cited numbers).[^llvm-baseinfo][^isa]

**Vec-kernel grid.** `grid.x = min(ceil(rows / 64), TQ_VEC_BPSM × nsm)`, with `TQ_VEC_BPSM` = 2 and 64 rows per 512-thread block (8 lanes per row). Blocks loop over row groups `gridDim.x` apart.[^p-vec-defs][^p-vec-grid] llama.cpp sets `nsm = prop.multiProcessorCount`. In default WGP mode, CLR halves `maxComputeUnits_`, so a 72-CU V620 reports 36.[^lc-nsm][^clr-cu-halve][^clr-props] That gives ≤ 72 blocks × 16 waves = 1152 resident waves against 36 WGPs × 64 slots = 2304. **At most 50% wave-slot fill by design.** The full grid needs ≥ 4608 rows, counted after QKV / gate+up fusion. Smaller matrices launch fewer blocks. The same `2 × nsm` rule applies on gfx1201, so this is not a gfx1030-only effect. The trade is fewer redundant src1 quantizes per launch.[^p-vec-defs]

**Small-batch kernel** (2+ columns, every batch on gfx1030). Each thread keeps 8 column accumulators (`TQ_SMALL_NCOLS`) and reads Q8 data with `global_load_dwordx4` straight from global memory, with no LDS staging. So VGPRs, not LDS, are its occupancy term.[^p-small][^ce]

## Instruction mix (CE run, per kernel body)

| Kernel | `v_dot4c_i32_i8` | `v_mul_u32_u24` | `v_and_or_b32` | `v_perm_b32` | `ds_read_b128` | `global_load_dwordx4` |
|---|---|---|---|---|---|---|
| small TQ2_0 | 64 | 0 | 0 | 0 | 0 | 16 |
| small TQ1_0 (unroll 2) | 128 | 44 | 16 | 0 | 0 | 32 |
| vec TQ2_0 | 24 | 0 | 2 | 2 | 6 | 2 |
| vec TQ1_0 | 16 | 44 | 18 | 2 | 4 | 2 |

The 2 `v_perm_b32` in the vec kernels come from the inlined src1 quantize, which packs bytes. The same 2 appear in standalone `quantize_q8_tq`. None are in the weight decode.[^ce] The counts are static instruction counts, not dynamic ones.

## Validation status

- Gate 42 (`scripts/gate-tq.sh`): PASS 18/18 on gfx1201. Every test-backend-ops NMSE is exactly 0 (61/61 MUL_MAT, 16/16 GET_ROWS).[^r356]
- `178efe84`: the CPU-backend text arm is a *placement*, not an oracle (`GGML_CUDA_TQ=0` moves neighbouring ops too). The NMSE op oracle read 0/305 on **gfx1201 and gfx1151**. The gate's census counters are now summed over every CUDA context. This is instrument-only, with no kernel change.[^gate-fix]
- **gfx1030: not run.** Bit-identity on V620 is untested. The CE run above shows the codegen preconditions hold (no contraction, sdot4 path), but that is not a silicon result.

## CE run (how the numbers above were produced)

Run on 2026-11-25 (Europe/Paris) through the godbolt.org API. Compiler `hipclang-trunk-rocm-70201` (clang 24.0.0git, llvm-project `5290ac44934128d45e8b42014db05e0bfa5bde05`, ROCm 7.2.1 device libs). Flags: `-x hip --offload-arch=gfx1030 --cuda-device-only -O3` (CE adds `-g1 -S`).[^ce]

The harness is `tq.cu` lines 1–386 of the block-17 patch, verbatim with its `#include`s removed, plus a stub header. The stubs copy `block_tq{1,2}_0`, `get_int_b2`, `ggml_cuda_dp4a` (RDNA2 branch), `ggml_cuda_shfl_xor` and `WARP_SIZE 32` from llama.cpp at the block's base `91979450`. They make two simplifications: `ggml_half` = `_Float16` with a cast for `__half2float`, and SwiGLU as `up · x/(1+e^{−x})`. Explicit instantiations are added for the six kernels. This is a **standalone harness, not the real build**. A ROCm release compiler, or the full `common.cuh` include chain, can move VGPR counts by a few registers. Treat the 65-VGPR "one over the cliff" result as **fragile**.

## Sources

[^r356]: stew675/llama-cpp-rdna-boosts commit [`ffdf4169`](https://github.com/stew675/llama-cpp-rdna-boosts/commit/ffdf416993a51f4abb883b67b70fb1c457dcaddd) (2026-11-24) — "r356: block 17 -- TQ1_0 / TQ2_0 (ternary) weights on the GPU (contributor PR #358)": default-on, `GGML_CUDA_TQ` / `_FUSE` / `_CENSUS`, unfused accumulation fix, gate 42 PASS 18/18 on gfx1201.
[^merge358]: stew675/llama-cpp-rdna-boosts commit [`fcd77def`](https://github.com/stew675/llama-cpp-rdna-boosts/commit/fcd77def387026d173e8b4182733955e241b3eb7) — "Merge contributor PR #358 (@overdoingism): the TQ1_0 / TQ2_0 (ternary) GPU campaign, promoted to block 17 at r356".
[^gate-fix]: stew675/llama-cpp-rdna-boosts commit [`178efe84`](https://github.com/stew675/llama-cpp-rdna-boosts/commit/178efe846235854880b552aa8d878a8b0a492ab8) — "gate-tq: the CPU-backend text arm is a PLACEMENT, not an oracle (gfx1151 measured it), and the census sums every line" (`scripts/gate-tq.sh` only).
[^p-header]: Block-17 patch header @ `ffdf4169`, L8–L45 (MUL_MAT not MUL_MAT_ID, decode vec kernel, 2..16 small kernel, RDNA4 WMMA, measured on gfx1201, gfx1100/gfx1151 arms). https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L8-L45
[^p-launch]: Block-17 patch, `ggml_cuda_mul_mat_tq_launch`: `mma = ne11 > TQ_SMALL_MAX && GGML_CUDA_CC_IS_RDNA4(cc) && …`, else `mul_mat_tq_small` with 256 threads, 32 rows/block. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L1066-L1104
[^p-mma-guard]: Block-17 patch, `mul_mat_tq_mma` body under `#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)`. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L912-L916
[^p-mmvq]: Block-17 patch, `mmvq.cu` `ggml_cuda_should_use_mmvq` returns false for TQ1_0 / TQ2_0. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L484-L489
[^p-dispatch]: Block-17 patch, `ggml_cuda_mul_mat` early branch to `ggml_cuda_mul_mat_tq`. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L388-L394
[^p-supports]: Block-17 patch, `supports_op`: TQ only for `GGML_OP_MUL_MAT` ("MUL_MAT_ID has no TQ path, its experts stay with the CPU backend"). https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L460-L463
[^p-fmul]: Block-17 patch, `TQ_FMUL` / `TQ_FADD` under clang = plain operators in `fp contract(off)` scopes; rationale (CPU emits vmulps + vaddps). https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L509-L523
[^p-vec-defs]: Block-17 patch, `TQ_LANES 8`, `TQ_SMALL_NCOLS 8`, `TQ_SMALL_MAX 16`, `TQ_VEC_THREADS 512`, `TQ_VEC_BPSM 2` ("thread blocks per SM at most (they take turns on the row groups)"), `TQ_VEC_SMEM 48*1024`. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L525-L534
[^p-q8tq]: Block-17 patch, `block_q8_tq` (lane-major q[256], bsum[8], d, pad[3]; static_assert 304). https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L539-L547
[^p-decode]: Block-17 patch, `tq1_0_digits4` and `tq_decode` for TQ2_0 / TQ1_0. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L661-L699
[^p-lanedot]: Block-17 patch, `tq_lane_dot`: `s = -bsum` then 8× `ggml_cuda_dp4a`. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L701-L713
[^p-small]: Block-17 patch, `mul_mat_tq_small` (8 lanes per row, `acc[TQ_SMALL_NCOLS]`, Q8 read from global). https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L723-L780
[^p-vec-ok]: Block-17 patch, `ggml_cuda_mul_mat_tq_vec_ok`: one column and `(ne0/QK_K)·sizeof(block_q8_tq) ≤ TQ_VEC_SMEM`. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L1106-L1111
[^p-vec-grid]: Block-17 patch, vec launch: `grid(min(ceil(rows/rows_f), TQ_VEC_BPSM*nsm), 1, …)`, `TQ_VEC_THREADS`, dynamic smem `nb*sizeof(block_q8_tq)`. https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L1113-L1129
[^p-trit]: Block-17 patch, `common.cuh` `ggml_cuda_tq1_0_trit` (qs / qh layout, digit formula). https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L68-L88
[^lc-dp4a]: llama.cpp `common.cuh` @ base `91979450`, `ggml_cuda_dp4a`: `CDNA || RDNA2 || gfx906` → `__builtin_amdgcn_sdot4(a, b, c, false)`; RDNA3/4 → `__builtin_amdgcn_sudot4(true, a, true, b, c, false)`. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/common.cuh#L772-L797
[^lc-rdna2]: llama.cpp `vendors/hip.h` @ `91979450`: `RDNA2` defined for `__gfx1030__` … `__gfx1037__`. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/vendors/hip.h#L238-L240
[^lc-nsm]: llama.cpp `ggml-cuda.cu` @ `91979450` L410: `info.devices[id].nsm = prop.multiProcessorCount`. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/ggml-cuda.cu#L410
[^lc-common-tq]: llama.cpp `ggml-common.h` @ `91979450`, `block_tq1_0` / `block_tq2_0` and size static_asserts. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-common.h#L275-L288
[^clr-cu-halve]: CLR `rocdevice.cpp` `maxComputeUnits_` /2 in WGP mode. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1150-L1161 (as cited on the 2026-11-20 split-KV page)
[^clr-props]: HIP `hip_device.cpp` `multiProcessorCount`. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_device.cpp#L712-L715
[^llvm-baseinfo]: LLVM AMDGPUBaseInfo.cpp (IsaInfo): `getMaxWavesPerEU` 16 for gfx10.3, `getVGPRAllocGranule` 16 (wave32), `getTotalNumVGPRs` 1024, `getLocalMemorySize` 128 KB WGP / 64 KB addressable. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp (checked 2026-10-11, as on architecture.md)
[^isa]: AMD "RDNA 2" ISA Reference Guide (doc 70648): VOP2 opcode 13 `V_DOT4C_I32_I8`, VOP3P `V_DOT4_I32_I8`; LDS 128 KB per WGP, up to 64 KB per workgroup. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[^ce]: Compiler Explorer run, 2026-11-25 Europe/Paris, compiler `hipclang-trunk-rocm-70201` = clang 24.0.0git @ llvm-project [`5290ac44`](https://github.com/llvm/llvm-project/commit/5290ac44934128d45e8b42014db05e0bfa5bde05) with ROCm 7.2.1 libs; `-x hip --offload-arch=gfx1030 --cuda-device-only -O3`; harness = block-17 `tq.cu` L1–386 (https://github.com/stew675/llama-cpp-rdna-boosts/blob/ffdf416993a51f4abb883b67b70fb1c457dcaddd/patches/0017-rdna-boosts-block-17-TQ1_0-TQ2_0-ternary-weights-on-.patch#L498-L883) + stubs described above. Read from the emitted `.s`: `.vgpr_count`, `.sgpr_spill_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`, `.amdhsa_wavefront_size32 1`, `.amdhsa_workgroup_processor_mode 1`, `; Occupancy:`, and per-kernel static opcode counts. No shortlink was created; rerun from the description.
