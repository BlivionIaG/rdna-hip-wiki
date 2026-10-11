---
title: "SGPR count vs occupancy (gfx1030)"
sources:
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/pull/154666
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://gpuopen.com/learn/occupancy-explained/
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---

# SGPR count vs occupancy (gfx1030)

Lock: **SGPRs are not an occupancy limiter on gfx1030.** LLVM `IsaInfo::isSGPROccupancyLimited` is **false** for ISA major ≥ 10. `getOccupancyWithNumSGPRs` returns `getMaxWavesPerEU` (16) without consulting the count. AMDHSA `COMPUTE_PGM_RSRC1.GRANULATED_WAVEFRONT_SGPR_COUNT` (bits 9:6) is **reserved and must be 0** on GFX10–GFX12 — the descriptor always budgets a fixed SGPR allocation (AMDGPUUsage: “128 SGPRs always allocated”). Completes the occupancy set: PIX limiters are VGPR / LDS / Thread Group Size / Barriers; SGPR is the explicit non-member.[^llvm-amdgpuusage][^llvm-baseinfo]

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [vgpr-occupancy.md](vgpr-occupancy.md), [lds-occupancy.md](../lds-and-tiles/lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [wg-size-occupancy.md](wg-size-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [occupancy-dump.md](occupancy-dump.md), [architecture.md](../arch-and-memory/architecture.md) §2.4 / §3.3, [hip-craft.md](../hip-craft/hip-craft.md) §1.3 / §6, [occupancy-composite.md](occupancy-composite.md).

## Take / Leave

| | |
|---|---|
| **Take** | Spend SGPRs on descriptors, strides, uniform pointers, kernarg mirrors, and broadcast constants. They are free for waves/EU on GFX10+. |
| **Take** | When reading a `.kd` / `llvm-objdump` of a gfx1030 kernel, expect `granulated_wavefront_sgpr_count = 0`. Non-zero is a descriptor bug (LLVM fixed the encoder in PR [#154666](https://github.com/llvm/llvm-project/pull/154666), merged 2025-08-27). [^github-pull-154666][^llvm-amdgpuusage] |
| **Take** | Still reject `.sgpr_spill_count > 0` in the occupancy dump — spill is **latency poison**, not a waves/SIMD cliff. Same reject rule as VGPR spill on [occupancy-dump.md](occupancy-dump.md). |
| **Leave** | Do not shrink SGPR use to raise occupancy. `isSGPROccupancyLimited` short-circuits; `llvm-calc-occupancy --sgprs=` cannot drop waves/EU on gfx1030. [^llvm-calc-occupancy][^llvm-baseinfo] |
| **Leave** | Do not treat pre-GFX10 SGPR budget math (`getSGPRBudgetPerWave` / trap reserve / alloc granule) as live for V620. Those paths are gated `Major < 10`. |
| **Leave** | Do not open a ticket / retip extras for this consolidation. |

## 1. LLVM formula (source of truth)

From `llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp` (`IsaInfo`, main tip 2026-09-10):[^llvm-baseinfo]

```text
bool isSGPROccupancyLimited(STI):
  return getIsaVersion(CPU).Major < 10   // gfx1030 → false

getOccupancyWithNumSGPRs(STI, SGPRs):
  if (!isSGPROccupancyLimited(STI))
    return getMaxWavesPerEU(Kind)        // 16 on gfx1030
  // else: closed-form inverse of getMaxNumSGPRs (pre-GFX10 only)

getMaxNumSGPRs(STI, WavesPerEU, Addressable)  // Major ≥ 10:
  return Addressable ? getAddressableNumSGPRs(Kind) : 108
  // no TotalNumSGPRs / WavesPerEU budget; no trap-reserve cliff
```

Pre-GFX10 still uses `getSGPRBudgetPerWave(Total / W − TrapReserve, Granule)` and its inverse (#201342 made them exact inverses). That math is **dead for gfx1030 occupancy** — keep it only as archaeology when reading Vega / gfx9 notes.

| Knob | gfx1030 |
|---|---|
| Occupancy-limited by SGPR? | **No** (`Major ≥ 10`) |
| Waves/EU from SGPR count | Always **16** (hardware max) |
| Addressable arch SGPRs (ISA) | **S0–S105** (106) + VCC in S106/S107 + trap temps |
| Descriptor SGPR field | **Must be 0** (128 always allocated) |
| Encoding granule (blocks) | 8 (still used for asm metadata; not an occupancy round) |

## 2. Descriptor lock (AMDGPUUsage + PR #154666)

`COMPUTE_PGM_RSRC1` bits **9:6** = `GRANULATED_WAVEFRONT_SGPR_COUNT`:[^llvm-amdgpuusage]

| Generation | Encoding |
|---|---|
| GFX6–GFX8 | `max(0, ceil(sgprs_used / 8) − 1)` |
| GFX9 | `2 * max(0, ceil(sgprs_used / 16) − 1)` |
| **GFX10–GFX12** | **Reserved, must be 0.** (128 SGPRs always allocated.) |

LLVM historically OR’d `SGPRBlocks` into those bits even on gfx10+, which `llvm-objdump` then rejected as “reserved bits … must be zero on gfx10+”. PR [#154666](https://github.com/llvm/llvm-project/pull/154666) (`SIProgramInfo::getComputePGMRSrc1`) writes **only** `VGPRBlocks` for `Generation ≥ GFX10`. Internal `SGPRBlocks` may still print non-zero in asm; the **on-disk `.kd` field must be 0**.[^github-pull-154666]

Practical check (same dump path as [occupancy-dump.md](occupancy-dump.md)):

```bash
llvm-objdump -d --section=.rodata path/to/kernel.co | rg 'granulated_wavefront_sgpr_count'
# expect: granulated_wavefront_sgpr_count = 0
```

## 3. GPUOpen / PIX (why SGPR is absent)

GPUOpen Occupancy explained (updated 2024-06-26):[^gpuopen-occupancy]

> On RDNA GPUs however, each wavefront is assigned a **fixed** number of SGPRs and there are always enough of those to fill the 16 slots.

PIX `WaveOccupancyLimiters` lists four: **VGPR, LDS, Thread Group Size, Barriers**. SGPR is not among them on RDNA. Do not invent a fifth limiter from `.sgpr_count` in NT_AMDGPU_METADATA.[^llvm-amdgpuusage]

## 4. What SGPR pressure *does* cost

| Cost | Binding? | Action |
|---|---|---|
| Extra SALU / move traffic | latency / issue | prefer descriptors already in SGPRs; avoid thrashing |
| `.sgpr_spill_count > 0` | latency (scratch) | reject config; same as VGPR spill |
| VCC / EXEC / M0 / specials | encoding / hazards | ISA rules; not waves/EU |
| Uniform→VGPR promotion when SGPR file feels “full” to the allocator | may raise **VGPR** | watch VGPR ladder, not SGPR occupancy |

## 5. Occupancy set (complete)

| Page | Role in `min(slots, …)` |
|---|---|
| [vgpr-occupancy.md](vgpr-occupancy.md) | waves/EU from VGPR file |
| [lds-occupancy.md](../lds-and-tiles/lds-occupancy.md) | WGs from LDS pool |
| [barrier-occupancy.md](barrier-occupancy.md) | multi-wave WG barrier slots |
| [wg-size-occupancy.md](wg-size-occupancy.md) | PIX Thread Group Size (atomic WG lifetime) |
| [scratch-occupancy.md](scratch-occupancy.md) | not in theoretical waves/EU; ROCr may cut waves_per_cu |
| **This page** | **explicit non-limiter** (GFX10+) |

## Sources

1. LLVM `IsaInfo::isSGPROccupancyLimited` / `getOccupancyWithNumSGPRs` / `getMaxNumSGPRs` — https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
2. LLVM AMDGPUUsage — `COMPUTE_PGM_RSRC1.GRANULATED_WAVEFRONT_SGPR_COUNT` GFX10–GFX12 “Reserved, must be 0. (128 SGPRs always allocated.)” — https://llvm.org/docs/AMDGPUUsage.html
3. LLVM PR [#154666](https://github.com/llvm/llvm-project/pull/154666) — force granulated SGPR count = 0 for gfx10+ (merged 2025-08-27)
4. GPUOpen Occupancy explained (updated 2024-06-26) — fixed SGPRs; 16 slots; four PIX limiters — https://gpuopen.com/learn/occupancy-explained/
5. RDNA 2 ISA 70648 §3.6.2 — S0–S105 + VCC

Footnotes:

[^llvm-amdgpuusage]: LLVM User Guide for AMDGPU Backend (AMDGPUUsage). https://llvm.org/docs/AMDGPUUsage.html (checked 2026-10-11)
[^llvm-baseinfo]: LLVM AMDGPUBaseInfo.cpp (IsaInfo), main snapshot eb61ef62df4f. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp (checked 2026-10-11)
[^github-pull-154666]: (AMDGPU) Set GRANULATED_WAVEFRONT_SGPR_COUNT of compute_pgm_rsrc1 to 0 for gfx10+ by Shoreshen · Pull Request #154666 · llvm/llvm-project. https://github.com/llvm/llvm-project/pull/154666 (checked 2026-10-11)
[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-10-11)
[^gpuopen-occupancy]: GPUOpen, "Occupancy explained". https://gpuopen.com/learn/occupancy-explained/ (checked 2026-10-11)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)

