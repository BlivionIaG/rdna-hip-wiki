# Wave size vs occupancy (gfx1030)

Lock: **wavefront size is not a PIX limiter row — it rescales every term that *is* one.** On gfx1030 HIP default (**wave32**), LLVM budgets a **1024**-VGPR file per SIMD32 with alloc granule **16**. Flip to **wave64** (`-mwavefrontsize64` / `+wavefrontsize64`) and the *same* SRAM is counted as **512** VGPRs with granule **8**; every VALU/VMEM beat is two passes on the SIMD32 (ISA §2.1); `N = ceil(FlatWG / WaveSize)` halves for the same threadgroup. Max waves/SIMD stays **16** either way. Completes the occupancy set as the **wave-size axis** — siblings still own VGPR / LDS / WG / barrier bytes and slots.

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [vgpr-occupancy.md](vgpr-occupancy.md), [wg-size-occupancy.md](wg-size-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [lds-occupancy.md](lds-occupancy.md), [lds-bank-occupancy.md](lds-bank-occupancy.md), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md), [occupancy-composite.md](occupancy-composite.md), [occupancy-dump.md](occupancy-dump.md), [hip-craft.md](hip-craft.md) §1 / §1.3, [architecture.md](architecture.md) §2.3–2.4.

## Take / Leave

| | |
|---|---|
| **Take** | Ship and size HIP kernels as **wave32**. Always pass `-mattr=+wavefrontsize32` to `llvm-calc-occupancy` on gfx1030 so the tool does not silently assume wave64 ([hip-craft.md](hip-craft.md) §1.3; CommandGuide example is often `gfx90a`). |
| **Take** | Compare modes with the same *logical* per-lane VGPR count: wave64 burns **2×** physical VGPR per wave → roughly **half** the waves/EU at equal pressure. Ladder: 256 VGPR → **4** waves wave32 vs **2** wave64; 64 VGPR wave32 / 32 VGPR wave64 → **16** either way ([architecture.md](architecture.md) §2.4; LLVM `getTotalNumVGPRs`). |
| **Take** | Recalculate WG packing when you change wave size: `N = ceil(FlatWG / WaveSize)` feeds barriers and Thread Group Size ([barrier-occupancy.md](barrier-occupancy.md), [wg-size-occupancy.md](wg-size-occupancy.md)). A 256-thr WG is **8** waves wave32 vs **4** wave64. |
| **Take (craft)** | GPUOpen: a bigger wave size implies **coarser** resource allocation granularity; do not casually pin wave size for occupancy alone — effects are complex and not performance-portable ([GPUOpen Occupancy explained](https://gpuopen.com/learn/occupancy-explained/)). |
| **Leave** | Do not ship `-mwavefrontsize64` as a HIP default on gfx10+. LLVM will compile it; HIP docs / runtime treat wave64 as unsupported on gfx10+ (`warpSize` 64; “`mwavefrontsize64` … not supported by HIP runtime”) ([hip-craft.md](hip-craft.md) §1). Treat wave64 as an LLVM experiment only. |
| **Leave** | Do not paste GPUOpen RDNA3 deck numbers (1536 VGPR → 96 / 48 at full slots) onto V620 — gfx1030 file is **1024** → **64** VGPR/wave (wave32) or **32** (wave64) at 16 slots ([occupancy-composite.md](occupancy-composite.md)). |
| **Leave** | Do not flip wave size to “get more barriers” or “fix LDS banks.” Barriers follow `N`; bank conflict-free indexed is ≈ **1 cycle wave32 / 2 cycles wave64**, worst still **64** (ISA §10.4.3). |
| **Leave** | Do not open a ticket / retip extras for wave-size occupancy. extras stays wave32. |

## 1. What wave size changes (and what it does not)

| Axis | wave32 (HIP default) | wave64 (`+wavefrontsize64`) | PIX row? |
|---|---|---|---|
| Physical VGPR file (`getTotalNumVGPRs`) | **1024** | **512** (same SRAM; counts double) | scales **VGPR** |
| Alloc granule | **16** | **8** | scales **VGPR** |
| Encoding granule | **8** | **4** | metadata only |
| Max waves / SIMD | **16** | **16** | unchanged slots |
| VALU / VMEM issue | 1 pass / SIMD32 | **2** passes (lo then hi; `EXEC` half may skip) | latency, not PIX |
| Waves in a WG | `ceil(WG/32)` | `ceil(WG/64)` | scales **WG size** + **Barriers** |
| LDS bytes / pool | unchanged | unchanged | **LDS** bytes unchanged |
| Bank conflict-free indexed | ≈ **1** cycle | ≈ **2** cycles | latency ([lds-bank-occupancy.md](lds-bank-occupancy.md)) |
| SGPR | fixed 128 / non-limiter | same | absent |

Wave size is therefore an **input** to the composite fold, not a fifth limiter. Flip it and re-run `llvm-calc-occupancy` with the matching `-mattr=`.

## 2. gfx1030 constants (not RDNA3 deck)

| Knob | wave32 | wave64 | Source |
|---|---|---|---|
| Clang / LLVM feature | `-mno-wavefrontsize64` (default) | `-mwavefrontsize64` | [AMDGPUUsage Target Features](https://llvm.org/docs/AMDGPUUsage.html) |
| HIP runtime default | **32** (`rocminfo` Wavefront Size) | not advertised on gfx10+ | HIP C++ language extensions; [hip-craft.md](hip-craft.md) |
| VGPR file / granule | 1024 / 16 | 512 / 8 | LLVM `getTotalNumVGPRs` / `getVGPRAllocGranule`; ISA §3.6.4 |
| Max waves/EU | 16 | 16 | LLVM `hasGFX10_3Insts` → 16; GPUOpen Occupancy explained |
| Addressable arch VGPRs | 256 | 256 | LLVM `getAddressableNumArchVGPRs` |

RDNA deck (1024-file) examples still apply; RDNA2 only cuts the *slot* cap vs RDNA1 (20 → 16):

- 4× wave32 × 256 VGPR
- 2× wave64 × 256 VGPR
- 16× wave32 × 64 VGPR
- 16× wave64 × 32 VGPR (full slots at the wave64 “half count”)

## 3. Worked folds (same WG, both modes)

Assume WGP mode, no LDS pressure, FlatWG = **256**.

| Mode | `N` waves/WG | VGPR/lane | Aligned | Waves/SIMD (VGPR term) | Barrier WGs (WGP) |
|---|---|---|---|---|---|
| wave32 | 8 | 64 | 64 | **16** | `min(64/8, 32) = 8` |
| wave64 | 4 | 64 | 64 → uses **128** physical equiv | **8** (`512/64`) | `min(64/4, 32) = 16` |
| wave32 | 8 | 128 | 128 | **8** | 8 |
| wave64 | 4 | 128 | 128 → **256** physical equiv | **4** (`512/128`) | 16 |
| wave32 | 8 | 256 | 256 | **4** | 8 |
| wave64 | 4 | 256 | 256 | **2** (`512/256`) | 16 |

Reading the table: at equal *named* VGPR counts, wave64 loses the VGPR term first. Barrier headroom can look *better* in wave64 because `N` is smaller — that does **not** make wave64 a barrier fix; you already left CU-mode for that reason ([barrier-occupancy.md](barrier-occupancy.md)).

CLI (theory only — HIP launch of wave64 objects on gfx1030 is still Leave):

```text
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --wg-size=256 --vgprs=128
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize64 --wg-size=256 --vgprs=128
```

## 4. Metadata / dump check

From [occupancy-dump.md](occupancy-dump.md): `.wavefront_size` must be **32** for HIP gfx1030 as shipped. **64** doubles VGPR cost per wave in the compiler accounting and is not the HIP default. If a dump shows 64, the object was built with `+wavefrontsize64` — treat as experimental, do not promote to extras.

## 5. Practice on V620 / extras

1. Keep `-offload-arch=gfx1030` without `-mwavefrontsize64`. Default wave32 + WGP.
2. Size with `llvm-calc-occupancy … -mattr=+wavefrontsize32` and gate with `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
3. Prefer lowering VGPR / LDS / WG size over changing wave size when measured occupancy is short ([occupancy-composite.md](occupancy-composite.md)).
4. Measure wave64 only as an LLVM side experiment (wide reduction / tool curiosity). Do not retip extras or open UNC for it.

## Sources

1. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/ (wave32 vs wave64 resource cost; do not pin wave size lightly)
2. GPUOpen GPC24 deck — https://gpuopen.com/download/GPC24_Occupancy_explained.pdf (1536-file examples; rescale to 1024 for gfx1030)
3. RDNA 2 ISA 70648 §2.1 (native wave32; wave64 = two passes), §3.6.4 (VGPR groups 16 / 8), §10.4.3 (LDS cycles)
4. LLVM AMDGPUUsage Target Features — `wavefrontsize64`, `cumode` — https://llvm.org/docs/AMDGPUUsage.html
5. LLVM `getTotalNumVGPRs` / `getVGPRAllocGranule` / occupancy tests (`03663e4` GFX10.3 wave32 vs wave64)
6. `llvm-calc-occupancy` — prints Wavefront size from STI; pass `-mattr=+wavefrontsize32|+wavefrontsize64`
7. HIP C++ language extensions / hardware implementation — wave32 primary on RDNA; wave64 option unsupported on gfx10+ runtime
8. Sibling wiki: [vgpr-occupancy.md](vgpr-occupancy.md), [architecture.md](architecture.md) §2.3–2.4, [hip-craft.md](hip-craft.md) §1
