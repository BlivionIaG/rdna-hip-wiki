# gfx1030 VALU sheet (HIP)

Engine “we fire” column filled 2026-08-17. Occupancy still first. Companions: [hip-craft.md](hip-craft.md), [architecture.md](architecture.md). Engine index: [coverage.md](../engine/coverage.md).

**How to read.** Enc = microcode class. Size = instruction dwords in I$ (VOP2 = 1, VOP3/VOP3P = 2). Issue = VALU issues per clock **per SIMD32** — **unknown** in the ISA for DOT; the GPUOpen **ops/clk/CU** column is the sourced peak (2 SIMD32/CU). Dual-issue (VOPD) is **gfx11+**, not gfx1030.

Rate math that matches GPUOpen RX 6950 XT: 1 DOT/clk/SIMD × 2 SIMD/CU × 32 lanes × (muls+adds in the DOT). DOT2 = 4 FLOP → **256** FP16. DOT4 = 8 integer ops → **512** IU8. DOT8 = 16 integer ops → **1024** IU4.

## gfx1030 — ops we can issue

| HIP / builtin | ISA | Enc | Size | Issue | Peak ops/clk/CU | Acc | We fire |
|---|---|---|---|---|---|---|---|
| `__builtin_amdgcn_fdot2` | `V_DOT2C_F32_F16` (prefer) / `V_DOT2_F32_F16` | VOP2 / VOP3 | 1 / 2 | 1/SIMD (inferred) | **256** FP16 | f32 | **Live:** W4A16, W8A16, W8A16-FP8, W8A8-FP8, mxfp4, `fa_rdna2` QK. **Live also:** EXL3 `3inst`→`fdot2` (`a2c8d5cf`; GEMMs still no `waves_per_eu`). **Queued:** NVFP4, INT2/W2A16, mixed INT2/INT4 MoE, INT8 KV (after cvt). |
| `__hfma2` / `v_fma_f32` | `V_FMA_F32` / `V_PK_FMA_F16` | VOP3 / VOP3P | 2 | 1/SIMD (typical VALU) | 128 FMA = 256 FLOP if packed | f32 / f16 | **HIP MLA decode, Lightning indexer** (later `fdot2`). ikantkode GEMV is Triton `tl.sum` — do not port. |
| `__builtin_amdgcn_sdot4` | `V_DOT4C_I32_I8` / `V_DOT4_I32_I8` | VOP2 / VOP3 | 1 / 2 | 1/SIMD (inferred) | **512** IU8 | i32 | **Spec / not extras @ `83de31cf`:** W8A8 INT8, Sage QK, W4A8 (after W→i8 unpack). Live “W8A8” is FP8→`fdot2`. [kernels/w8a8.md](../kernels/w8a8.md) |
| `__builtin_amdgcn_udot4` | `V_DOT4_U32_U8` | VOP3 | 2 | same class | 512 IU8 | u32 | unused (signed weights) |
| `__builtin_amdgcn_sdot8` | `V_DOT8_I32_I4` | VOP3 | 2 | 1/SIMD (inferred) | **1024** IU4 | i32 | **Explore:** W4A4 integer only. Not W4A8. Not E2M1. |
| `__builtin_amdgcn_udot8` | `V_DOT8_U32_U4` | VOP3 | 2 | same class | 1024 IU4 | u32 | unused |
| `__int2half_rn` / `v_cvt_f16_i16` | cvt | VOP1 / VOP3 | 1–2 | **unknown** | — | — | W8A16 unpack |
| `fp8_e4m3_to_fp16_bits` / `__hip_cvt_fp8_to_halfraw` | **software** | — | — | — | — | — | W8A16-FP8, W8A8-FP8, MLA K_nope. No FP8 unit. |
| `__bfloat162float` | bit-shift + cvt | — | — | — | — | — | MLA K_rope (slot is bf16). No bf16 DOT. |

Prefer the **C** form (`DOT2C` / `DOT4C`): VOP2, 1 dword, `vdst = src2`. hipcc will **not** peephole `__hfma2` into DOT2 — issue the builtin ([hip-craft.md](hip-craft.md) §0.7).

## gfx1030 — absent (do not emit)

| Name | Why | Source |
|---|---|---|
| `V_DOT2_F32_BF16` / `fdot2_f32_bf16` | gfx11 `dot12`/`dot13` | LLVM; GPUOpen BF16 = N/A on RX 6950 XT |
| `V_WMMA_*` / `__builtin_amdgcn_wmma_*` | RDNA3 only | GPUOpen WMMA blog |
| MFMA / `__builtin_amdgcn_mfma_*` | CDNA | skinny_gemms.cu compiled out (`use_mfma=false`) |
| `sudot4` / `V_DOT4_I32_IU8` | gfx11+ mixed-sign | llama.cpp RDNA3 path |
| FP8 / FP4 / MX tensor unit | no silicon | `supports_fp8()` / `supports_mx()` false |
| VOPD dual-issue (`v_dual_dot2acc_*`) | gfx11 | LLVM #186179 |

## gfx1100 extras (do not use on V620)

| Extra | Enc | Peak vs gfx1030 | HIP | Our stance |
|---|---|---|---|---|
| WMMA 16×16×16 f16/bf16/iu8/iu4 | wave matrix, ~32 clk schedule | FP16 **256→512**, IU8/IU4 **unchanged** (512 / 1024) | `__builtin_amdgcn_wmma_*_w32` | Ignore on gfx1030. gfx1100 prefill only if we ever ship a 1100 tile. |
| `V_DOT2_F32_BF16` | VOP3P (8 B) unless VOPD-paired | 512 BF16 | `__builtin_amdgcn_fdot2_f32_bf16` | Dead on gfx1030. User forces fp16. |
| VOPD `v_dual_dot2acc_f32_f16` | dual VOP2 | two DOT2/clk when paired | automatic if builtin is VOP2 | gfx11 win; not here. |
| `sudot4` | VOP3 | mixed-sign i8 | `__builtin_amdgcn_sudot4` | Not on gfx1030. |

WMMA does **not** beat DOT4/DOT8 on INT8/INT4 (same 512/1024). It only doubles **FP16/BF16**. That is why W8A8/W4A4 stay `sdot4`/`sdot8` even on a 1100 card.

## Issue the right one

| Kernel | gfx1030 op |
|---|---|
| W4A16 / W8A16 / FP8-storage / mxfp4 / NVFP4 / INT2 / EXL3 | `fdot2` |
| `fa_rdna2` QK / INT8 KV (after i8→fp16) | `fdot2` |
| HIP MLA / indexer (today) | scalar FMA — later `fdot2` |
| W8A8 INT8 / Sage QK / W4A8 | `sdot4` |
| W4A4 integer | `sdot8` |
| MXFP4/NVFP4 “W4A4” | dequant A → `fdot2`, **not** `sdot8` |

## Sources

- RDNA 2 ISA 70648 feature list: DOT2 / DOT4 / DOT8
- GPUOpen WMMA on RDNA3: https://gpuopen.com/learn/wmma_on_rdna3/ — RX 6950 XT vs 7900 XTX table
- LLVM `VOP2Instructions.td`: `V_DOT2C_F32_F16_e32`, `V_DOT4C_I32_I8_e32` when no src mods
- Clang AMDGPU builtins; llama.cpp `ggml_cuda_dp4a`
- LLVM #186179: gfx11 VOPD / bf16 DOT
