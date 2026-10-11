---
title: "SGLang overlay — silicon (import extras, no second DOT tree)"
sources:
  - https://gpuopen.com/learn/wmma_on_rdna3/
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---

# SGLang overlay — silicon (import extras, no second DOT tree)

Date: 2026-08-20. Engine: [../engine/sglang-fork.md](../../engine/serving-stack/sglang-fork.md). Occupancy still first.

An SGLang fork does **not** get a new ISA. V620 is still gfx1030: `fdot2` / `sdot4` / `sdot8`, no WMMA, no MFMA, no hardware bf16. Serving (radix, overlap) is CPU/engine. Compute objects come from **`rdna2_extras` after occupancy**.[^isa70648]

## Import, do not rewrite

| Keep (one HIP tree) | Refuse on V620 |
|---|---|
| `fa_rdna2` occupancy-fixed (`waves_per_eu(4,8)`, never `(1,1)`) | AITER / CK XDL / rocWMMA / Instinct FA |
| Skinny GEMV + W4A16 dequant → `fdot2` | `on_rdna()` as a gfx1030 gate (that is gfx11/12) |
| `on_gfx10x()` whitelist | FlashKDA CUTLASS / SM90 / bf16 |
| Same `__launch_bounds__` lesson as extras | A second copy of the `.cu` with different occupancy |

gfx1100 (W7800 attn tier) may use **local WMMA**. gfx900 (V340L) is hippih, own host — `mad_mix` / `pk_fma`, not these objects.[^m-gpuopen-com-learn-wmma-on-rdna3]

Triton in SGLang is the same compile-time tax we already rejected for decode GEMV ([../kernels/ikantkode-gfx1030.md](../../kernels/ikantkode-gfx1030.md)). Overlay may keep Triton as a **fallback**, not as the inner loop we rewrite HIP to replace.

## Order

1. extras occupancy + `load_row` + CMake gap.
2. Then copy **those** objects into the SGLang overlay.
3. Do not start a parallel `fdot2` tree “so SGLang can move.”

No tok/s from their A100/H20 posts. No occupancy retip.

## Sources

- [../engine/sglang-fork.md](../../engine/serving-stack/sglang-fork.md), [fa-occupancy.md](../attention/fa-occupancy.md), [fp16-rdna2.md](../arch-and-memory/fp16-rdna2.md), [valu.md](../arch-and-memory/valu.md), [codegen-stack.md](../hip-craft/codegen-stack.md)
- Note 2026-08-20: own SGLang path

[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)

[^m-gpuopen-com-learn-wmma-on-rdna3]: How to accelerate AI applications on RDNA 3 using WMMA - AMD GPUOpen. https://gpuopen.com/learn/wmma_on_rdna3/ (checked 2026-10-11)
