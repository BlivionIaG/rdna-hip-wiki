# Occupancy composite (gfx1030)

Lock: **theoretical waves/EU is the min of the PIX limiter set**, folded the same way LLVM does. For HIP default (**wave32 + WGP**) that is:

```text
waves/EU = min(
  getOccupancyWithNumVGPRs(V),                          // VGPR page
  getOccupancyWithNumSGPRs(S),                          // always MaxWaves=16 on GFX10+
  getOccupancyWithWorkGroupSizes(LDS, {WG_lo, WG_hi})   // LDS + WG size + barriers
)
```

`llvm-calc-occupancy` is the thin CLI over that `GCNSubtarget` math. Scratch is **out** of the min (latency / rare ROCr cut only). Completes the occupancy set as the **fold** page — siblings own each term.

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [vgpr-occupancy.md](vgpr-occupancy.md), [lds-occupancy.md](lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [wg-size-occupancy.md](wg-size-occupancy.md), [wave-size-occupancy.md](wave-size-occupancy.md), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [icache-occupancy.md](icache-occupancy.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [l2-occupancy.md](l2-occupancy.md), [occupancy-dump.md](occupancy-dump.md), [hip-craft.md](hip-craft.md) §1.3 / §6, [fa-occupancy.md](fa-occupancy.md), [architecture.md](architecture.md) § occupancy.

## Take / Leave

| | |
|---|---|
| **Take** | Size with `llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --wg-size=N --vgprs=V --lds=L`. Read **Limited by:** then open the matching sibling page. Omit `--sgprs` on gfx1030 (non-limiter). |
| **Take** | Wave size is an **input** to the fold, not a PIX row — keep `+wavefrontsize32` unless measuring an LLVM-only wave64 experiment ([wave-size-occupancy.md](wave-size-occupancy.md)). |
| **Take** | Launch mode (WGP/`-mno-cumode` vs CU/`-mcumode`) is also an **input** — keep WGP unless measuring LDS-bandwidth CU ([wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md)). |
| **Take** | Gate launches with `hipOccupancyMaxActiveBlocksPerMultiprocessor` after the theory pass — runtime 0 means do not launch, even when the calc says otherwise ([vgpr-occupancy.md](vgpr-occupancy.md) §4). |
| **Take** | PIX / GPUOpen limiter names map 1:1 to wiki pages: **VGPR** → vgpr; **LDS** → lds; **Thread Group Size** → wg-size; **Barriers** → barrier. SGPR is absent on RDNA (fixed 128). |
| **Take** | Measured occupancy can sit under theory for **lack of work** (grid << 40 WGP × 4 SIMD × 16) or **launch-rate** drain — those are SPI/ACE issues, not a fifth PIX resource. |
| **Leave** | Do not use GPUOpen RDNA3 examples (1536 VGPR / RX 7900) as gfx1030 constants — V620 SIMD file is **1024** VGPR, MaxWaves **16**. |
| **Leave** | Do not put `.private_segment_fixed_size` / scratch into `llvm-calc-occupancy` or the theoretical min ([scratch-occupancy.md](scratch-occupancy.md)). |
| **Leave** | Do not put I$ / code size into the theoretical min — fetch stalls are effective occupancy only ([icache-occupancy.md](icache-occupancy.md)). |
| **Leave** | Do not put L0/GL1 capacity into the theoretical min — vector-cache thrash is effective occupancy only ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)). |
| **Leave** | Do not put L2 capacity into the theoretical min — GPU-wide mid-cache thrash is effective occupancy only ([l2-occupancy.md](l2-occupancy.md)). |
| **Leave** | Do not maximize occupancy as a goal. ALU-bound kernels want utilization, not more waves; memory-bound kernels can thrash IC/L2 if you over-fill ([GPUOpen Occupancy explained](https://gpuopen.com/learn/occupancy-explained/)). |
| **Leave** | Do not open a ticket / retip extras for this fold. |

## 1. PIX four → LLVM fold

| PIX / GPUOpen limiter | Wiki page | LLVM hook |
|---|---|---|
| VGPR | [vgpr-occupancy.md](vgpr-occupancy.md) | `getOccupancyWithNumVGPRs` / alloc granule 16 / file 1024 |
| LDS | [lds-occupancy.md](lds-occupancy.md) | `MaxWGsLDS = floor(LocalMemorySize / alignTo(LDS))` inside `getOccupancyWithWorkGroupSizes` |
| Thread Group Size | [wg-size-occupancy.md](wg-size-occupancy.md) | `WavesPerWG = ceil(WG / 32)`; atomic WG lifetime on SPI |
| Barriers | [barrier-occupancy.md](barrier-occupancy.md) | `getMaxWorkGroupsPerCU` → `min(MaxWaves/N, 32)` WGP / `16` CU; `N==1` free |
| *(absent)* SGPR | [sgpr-occupancy.md](sgpr-occupancy.md) | `isSGPROccupancyLimited` false → always 16 |
| *(out of min)* Scratch | [scratch-occupancy.md](scratch-occupancy.md) | not in `computeOccupancy` / no `--scratch` |
| *(out of min)* I$ / code size | [icache-occupancy.md](icache-occupancy.md) | not a PIX MaxWaves row; SQC miss → wave idle |
| *(out of min)* L0 / GL1 (TCP) | [l0-gl1-occupancy.md](l0-gl1-occupancy.md) | not a PIX MaxWaves row; thrash → memory-wait / weaker latency hiding |
| *(out of min)* L2 | [l2-occupancy.md](l2-occupancy.md) | not a PIX MaxWaves row; thrash → memory-wait / weaker latency hiding (mid vs IC) |

`getMaxWorkGroupsPerCU` (AMDGPUBaseInfo.cpp) already packs **wave slots + barriers**:

```text
MaxWaves = 16 * getNumWorkGroupSIMDs(WGP→4)   // 64 slots/WGP
N        = ceil(FlatWG / WaveSize)
if N == 1: return MaxWaves                    // no barrier
MaxBarriers = (GFX10+ && !CuMode) ? 32 : 16
return min(MaxWaves / N, MaxBarriers)
```

`getOccupancyWithWorkGroupSizes` then `min`s that with `MaxWGsLDS`, converts WGs×waves → waves/EU by spreading across `getNumWorkGroupSIMDs()`, and returns a **(min, max)** range when `amdgpu_flat_work_group_size` is a range (LLVM PR #123748).

Final CLI combine (`llvm-calc-occupancy.cpp`, same as `GCNSubtarget::computeOccupancy`):

```text
Occ = min(WG+LDS term, VGPR term, SGPR term)
Limited by: list every term that equals Occ
```

## 2. gfx1030 constants (not RDNA3 deck)

| Knob | gfx1030 WGP wave32 | Common GPUOpen example (RDNA3) |
|---|---|---|
| Max waves / SIMD (EU) | **16** | 16 |
| VGPR file / SIMD | **1024** | 1536 (RX 7900 class) |
| VGPR alloc granule | **16** | architecture-dependent |
| SIMDs / workgroup unit | **4** (WGP) | 4 |
| Wave slots / WGP | **64** | 64 |
| Barriers / WGP | **32** | 16 per SIMD *pair* (= CU half) |
| LDS pool | **128 KB** WGP / **64 KB** WG cap | same RDNA shape |
| V620 WGP count | **40** (72 CU / 2) | ASIC-specific |

Max theoretical wave slots on one V620 ≈ `40 × 4 × 16 = 2560`. A dispatch with fewer wavefronts **cannot** hit 100% measured occupancy even at full theory.

## 3. Worked folds (hand calc = CLI)

Assume HIP default: wave32, WGP, no `-mcumode`.

| Shape | WG | V | LDS | VGPR waves/EU | WGs from LDS | WGs from slots/bar | Binding WGs | waves/EU |
|---|---|---|---|---|---|---|---|---|
| Skinny decode-ish | 256 | 64 | 4 KiB | `floor(1024/64)=16` | `floor(128/4)=32` | `min(64/8,32)=8` | **8** | **8** (WG/barrier) |
| Mid tile | 256 | 128 | 32 KiB | 8 | 4 | 8 | **4** | **8** (LDS = VGPR) |
| Fat prefill-ish | 256 | 208 | 48 KiB | `floor(1024/208)=4` | 2 | 8 | **2** | **4** (LDS = VGPR) |
| Single-wave | 32 | 128 | 1 KiB | 8 | 128 | **64** (no barrier) | LDS/slots | **8** (VGPR) |

Recipe:

```bash
# theory (same math as backend remarks)
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 \
  --wg-size=256 --vgprs=128 --lds=32k
# → Occupancy: 8 waves/EU; Limited by: workgroup size / LDS, VGPRs

# dump live .co then re-check (reject spills)
# see occupancy-dump.md
```

`--limits` prints the Max-VGPR / Max-SGPR columns per occupancy rung — use it when walking a VGPR cliff; SGPR column stays at the hardware max on gfx1030.

## 4. Theoretical vs measured (GPUOpen)

GPUOpen Occupancy explained (updated 2024-06-26):

- **Theoretical** = resources reserved at compile / launch (the min above).
- **Measured** ≤ theory when (a) the grid cannot fill the ASIC, (b) end-of-dispatch drain, or (c) **launch-rate** (waves finish faster than SPI can refill; dependency barriers force empty-GPU ramp).
- PIX `WaveOccupancyLimiters` percentages are **binary which-resource** signals, **not** “how occupancy-bound.” A non-zero limiter means that resource blocked a launch attempt that clock; duration of waves warps the percentage.

Craft implication for extras kernels: if RGP / PIX shows theory high but measured low on a fat FA/W4 grid, first check **grid fill and barriers between dispatches**, not another VGPR shave. If theory equals measured and both are low, open the limiter sibling.

## 5. HIP API naming trap

| Name in API / docs | Means on gfx1030 |
|---|---|
| EU / Execution Unit in `amdgpu_waves_per_eu`, `__launch_bounds__` 2nd arg | **one SIMD32** |
| HIP “Compute Unit consists of 4 Execution Units” | **WGP** picture (4 SIMD), not ISA CU (2 SIMD) |
| `rocminfo` “SIMDs per CU: 4” | same WGP confusion |
| `hipOccupancyMaxActiveBlocksPerMultiprocessor` “multiprocessor” | treat as **WGP** under HIP default; verify against `llvm-calc-occupancy` × WGP count |

Pin `__launch_bounds__(REAL_BLOCK, MIN_WAVES)` or `amdgpu_waves_per_eu` + `amdgpu_flat_work_group_size` to the **actual** launch — never `__launch_bounds__(1024)` on a 128/256-thr kernel ([hip-craft.md](hip-craft.md) §1.3).

## 6. Operational checklist

1. Dump NT_AMDGPU_METADATA ([occupancy-dump.md](occupancy-dump.md)): reject `.vgpr_spill_count` / `.sgpr_spill_count` / unexpected private.
2. Run `llvm-calc-occupancy` with real WG / VGPR / LDS.
3. Read **Limited by:** → open sibling; change **one** lever (tile LDS, WG size, or VGPR ceiling via `waves_per_eu`).
4. Confirm `hipOccupancyMaxActiveBlocksPerMultiprocessor` > 0 for the launch config.
5. Profile measured vs theory only after theory matches the intended limiter.

## Sources

1. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
2. `llvm-calc-occupancy` — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
3. LLVM `AMDGPUSubtarget::getOccupancyWithWorkGroupSizes` / `IsaInfo::getMaxWorkGroupsPerCU` — `AMDGPUSubtarget.cpp`, `AMDGPUBaseInfo.cpp` (main tip 2026-09-11)
4. LLVM `llvm-calc-occupancy.cpp` combine loop (`Limited by:` list)
5. RDNA 2 ISA 70648 §2.3.1 / §10.3 — WGP vs CU, LDS cap
6. Sibling locks in this folder (VGPR / LDS / barrier / WG-size / SGPR / scratch / I$ / L0-GL1 / L2 / dump)
