---
title: "VMEM addressing modes (global SADDR, offsets) vs VGPR occupancy (gfx1030)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp
  - https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp
  - https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNFeatures.td
  - https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNSubtarget.h
---

# VMEM addressing modes (global SADDR, offsets) vs VGPR occupancy (gfx1030)

Lock: **Addressing mode is not a PIX / `llvm-calc-occupancy` MaxWaves row.** It touches occupancy only through the VGPRs that hold addresses. A plain `global_load` takes a full 64-bit address in a VGPR pair. The SADDR form takes a uniform 64-bit base in SGPRs plus a 32-bit VGPR offset, so each live address costs one VGPR instead of two. The immediate offset field decides how many unrolled loads can share one address register before the compiler needs a fresh add.

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-11-04** Europe/Paris.

Companions: [vgpr-occupancy.md](vgpr-occupancy.md) (VGPR waves/EU and granule), [constant-bus-occupancy.md](constant-bus-occupancy.md) (VGPR copies from operand rules), [hard-clause-occupancy.md](hard-clause-occupancy.md) (one base + offsets load bursts), [hazard-occupancy.md](hazard-occupancy.md) (gfx101x / BC-250 bug list), [cache-policy.md](../arch-and-memory/cache-policy.md) (buffer intrinsics).

## Take / Leave

| | |
|---|---|
| **Take** | Global loads and stores on gfx1030 have two legal forms: a 32- or 64-bit VGPR address with SADDR = NULL, or a **64-bit SGPR base plus a 32-bit VGPR offset**. Both may add the instruction immediate offset.[^isa-legal] Per live pointer the SADDR form holds one VGPR, not two. |
| **Take (craft)** | LLVM picks SADDR only when the base is uniform (not divergent) and the per-lane part is an i32 that is **zero-extended**, or sign-extended with a provably zero sign bit. The signed-offset and scaled-offset variants are GFX1250-only.[^isel-saddr][^isel-ext][^st-gvs] So keep kernel base pointers uniform and per-lane indices as unsigned 32-bit (or provably non-negative) byte offsets. A signed `int` index the compiler cannot prove non-negative can fall back to the 64-bit VGPR address. That last step is our reading of the matcher and is **unverified on the box** (no compiler, see below). |
| **Take (craft)** | Immediate offset windows on gfx1030: **global / scratch `OFFSET` is 12-bit signed** (-2048..+2047 bytes); **FLAT (generic) offset must be non-negative** (0..2047 in LLVM's legality check); **MUBUF `OFFSET` is 12-bit unsigned** (0..4095 bytes).[^isa-flat-off][^isa-mubuf-off][^llvm-bits][^llvm-legal][^llvm-mubuf] Unrolled loads whose constant offsets fit one window share one address register; past it LLVM splits the constant into an immediate part plus a remainder held in a register.[^llvm-split][^isel-saddr] Lay out per-lane K/V or weight bursts so each burst spans at most 4 KiB from a buffer base or ±2 KiB from a global base. |
| **Take (BC-250 split)** | The gfx101x feature list carries `FlatSegmentOffsetBug` ("inst_offset is ignored when flat instructions access global memory") and `NegativeUnalignedScratchOffsetBug`; gfx1030 (`FeatureISAVersion10_3_0`, built on `FeatureISAVersion10_Common`) does not. gfx1013 (BC-250) is built on the gfx101x list.[^td-bugs][^td-101][^td-103] Another reason gfx1030 objects are not BC-250 objects, next to the hazard split. |
| **Take Later (fork call, probe only)** | Dump extras hot-loop ISA (W4 dequant / MoE GEMM, FA K/V loads) and count `global_load` with `off` (64-bit VGPR address) versus an `s[..]` SADDR operand. Where a kernel near the 128-VGPR line keeps several 64-bit addresses live, switching the index to `uint32_t` byte offsets from a uniform base may free VGPRs. No compiler on the box today (clang-22 gone), so this is unverified. Routed as a fork note, not a ticket; no extras edit. |
| **Leave** | Not a waves/EU term by itself; it only matters when address VGPRs push a kernel across a VGPR granule. Not a fix for scalar-cache or L0 behaviour; see [kcache-occupancy.md](kcache-occupancy.md) and [l0-gl1-occupancy.md](l0-gl1-occupancy.md). |

## 1. Forms in one table

| Instruction family | Address source | Per-lane VGPRs for address | Immediate offset on gfx1030 |
|---|---|---|---|
| `flat_*` | VGPR 64-bit, SADDR must be NULL[^isa-legal] | 2 | non-negative only[^isa-flat-off][^llvm-legal] |
| `global_*`, SADDR = NULL | VGPR 64-bit address[^isa-legal] | 2 | 12-bit signed[^isa-flat-off] |
| `global_*`, SADDR set | SGPR 64-bit base + VGPR 32-bit offset[^isa-legal] | 1 | 12-bit signed[^isa-flat-off] |
| `buffer_*` (MUBUF) | V# in 4 SGPRs + optional VGPR offset/index (`OFFEN`/`IDXEN`) + SGPR `soffset`[^isa-mubuf-off] | 0 to 2 | 12-bit unsigned[^isa-mubuf-off][^llvm-mubuf] |

When the whole address is uniform, LLVM still uses SADDR and materialises a single zero VGPR offset, because one `v_mov` is cheaper than copying a 64-bit SGPR pair into VGPRs.[^isel-zero]

## 2. What this page does not claim

- No measured VGPR delta for any extras kernel; the probe above is the way to get one.
- No claim that MUBUF is faster or slower than global on gfx1030; only the encoding facts above.

## Sources

[^isa-legal]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, §9.3.1 "Legal Addressing Combinations" (FLAT: VGPR full address, SADDR NULL; Global: VGPR address, or SGPR 64-bit address + VGPR 32-bit offset; every mode can add the instruction immediate offset). https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (local text `/workspace/rdna2-src/rdna2-isa.txt`, checked 2026-11-04)
[^isa-flat-off]: Same guide, §9.1 FLAT microcode fields: "OFFSET 12 … Scratch, Global: 12-bit signed byte offset. Flat: must be positive." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[^isa-mubuf-off]: Same guide, §8.1 buffer instruction fields: `SRSRC` (V# in four consecutive SGPRs), "OFFSET 12 Unsigned byte offset", `OFFEN` / `IDXEN` VGPR offset / index. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[^isel-saddr]: LLVM `AMDGPUDAGToDAGISel::SelectGlobalSAddr` @ `fd64c07f9ae1d32477849eb930379cf1222cb0d8`: "Match (64-bit SGPR base) + (zext vgpr offset) + sext(imm offset)"; requires `!LHS->isDivergent()` for the base; large constants split into a `v_mov` remainder plus immediate. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp#L2075-L2202
[^isel-ext]: Same file, `matchExtFromI32orI32`: accepts i32, the matching extend, any_extend, or the opposite extend only when `SignBitIsZero`. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp#L2058-L2073
[^isel-zero]: Same file: "It's cheaper to materialize a single 32-bit zero for vaddr than the two moves required to copy a 64-bit SGPR to VGPR." https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp#L2189-L2202
[^st-gvs]: LLVM `GCNSubtarget.h` @ `fd64c07f9ae1d32477849eb930379cf1222cb0d8`: `hasScaleOffset()` and `hasSignedGVSOffset()` return `HasGFX1250Insts`. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNSubtarget.h#L539-L542
[^llvm-bits]: LLVM `getNumFlatOffsetBits` @ `fd64c07f9ae1d32477849eb930379cf1222cb0d8` returns 12 with `FeatureFlatOffsetBits12`; `def FeatureGFX10` lists `FeatureFlatOffsetBits12`. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L3336-L3342 and https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNFeatures.td#L1811-L1844
[^llvm-legal]: LLVM `SIInstrInfo::isLegalFLATOffset` and `allowNegativeFlatOffset` @ `fd64c07f9ae1d32477849eb930379cf1222cb0d8`: "Pre-GFX12, flat instruction offsets can only be non-negative, global and scratch instruction offsets can also be negative"; offset legal iff `isIntN(N, Offset)` and (negative allowed or `Offset >= 0`); gfx10.3 row of the scratch table allows negative unaligned offsets. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L10764-L10858
[^llvm-split]: Same file, `SIInstrInfo::splitFlatOffset`. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L10818-L10849
[^llvm-mubuf]: Same file, `getMaxMUBUFImmOffset`: 12 offset bits before GFX12, max `(1 << 12) - 1`. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L10672-L10681
[^td-bugs]: LLVM `GCNFeatures.td` @ `fd64c07f9ae1d32477849eb930379cf1222cb0d8`: `FlatSegmentOffsetBug` "GFX10 bug where inst_offset is ignored when flat instructions access global memory"; `NegativeUnalignedScratchOffsetBug`. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNFeatures.td#L374-L387
[^td-101]: Same file, `FeatureISAVersion10_1_Common` lists both bugs under "gfx101x bugs"; `FeatureISAVersion10_1_3` builds on it. https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNFeatures.td#L2230-L2285
[^td-103]: Same file, `FeatureISAVersion10_Common` and `FeatureISAVersion10_3_0` (neither bug listed). https://github.com/llvm/llvm-project/blob/fd64c07f9ae1d32477849eb930379cf1222cb0d8/llvm/lib/Target/AMDGPU/GCNFeatures.td#L2219-L2302
