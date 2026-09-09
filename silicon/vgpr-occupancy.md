# VGPR count vs occupancy (gfx1030)

Lock: **VGPR pressure is the per-SIMD waves/EU term.** On gfx1030 HIP default (**wave32**), LLVM budgets a **1024**-VGPR file per SIMD32 with alloc granule **16**. Waves/EU = `min(16, floor(1024 / alignTo(vgpr_count, 16)))`. Addressable ceiling is **V0–V255**. SGPRs never bind on GFX10+. Barriers / LDS / scratch stay on their own pages — this page is the VGPR term in `min(slots, VGPR, LDS, WG/barrier)`.

Does **not** change extras HIP or UNC cards. No tok/s. Do not restate the FA pin.

Companions: [occupancy-dump.md](occupancy-dump.md), [lds-occupancy.md](lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [hip-craft.md](hip-craft.md) §1.3 / §6, [architecture.md](architecture.md) §2.4 / §3.3, [fa-occupancy.md](fa-occupancy.md), [wg-size-occupancy.md](wg-size-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Round `.vgpr_count` **up** to 16 before dividing into 1024. A 203-count spike allocates **208** and drops a rung. |
| **Take** | EU in `__launch_bounds__(…, MIN_WARPS)` and `amdgpu_waves_per_eu` = **one SIMD32**, not the ISA CU (2 SIMD) and not rocminfo’s “4 SIMDs per CU” WGP picture. |
| **Take** | Cap design VGPR so a whole WG still fits the per-SIMD budget after SPI spreads waves across the WGP’s 4 SIMDs (see §4). Gate launches with `hipOccupancyMaxActiveBlocksPerMultiprocessor`, not compiler remarks alone. |
| **Take (craft)** | Prefer `__attribute__((amdgpu_waves_per_eu(min[,max])))` + `amdgpu_flat_work_group_size` when you need a VGPR ceiling; `__launch_bounds__(MAX_THREADS, MIN_WARPS)` also works — set `MAX_THREADS` to the **real** block size (never 1024 on a 128/256-thr kernel). |
| **Leave** | Do not chase SGPR occupancy on gfx1030 — `isSGPROccupancyLimited` is false for Major ≥ 10. |
| **Leave** | Do not open UNC / retip extras because of this consolidation. Skinny decode already aims ≤ 64–128 VGPR; fat prefill stays LDS-bound first. |

## 1. LLVM formula (source of truth)

From `IsaInfo::getNumWavesPerEUWithNumVGPRs` / `getVGPRAllocGranule` / `getTotalNumVGPRs` / `getMaxWavesPerEU` (`AMDGPUBaseInfo.cpp`, main tip 2026-09-08):

```text
MaxWaves     = getMaxWavesPerEU(gfx1030)           // 16 (hasGFX10_3Insts)
TotalVGPRs   = getTotalNumVGPRs(wave32)            // 1024 (no Feature1536VGPRs)
Granule      = getVGPRAllocGranule(wave32)         // 16 (hasGFX10_3Insts)
Addressable  = getAddressableNumArchVGPRs()        // 256 (no Feature1024AddressableVGPRs)

if NumVGPRs < Granule:
  waves/EU = MaxWaves
else:
  Rounded  = alignTo(NumVGPRs, Granule)
  waves/EU = min(max(TotalVGPRs / Rounded, 1), MaxWaves)

// Budget the other way (codegen / waves_per_eu):
MaxNumVGPRs(W) = min(alignDown(TotalVGPRs / W, Granule), Addressable)
```

| Constant | gfx1030 wave32 | wave64 note |
|---|---|---|
| Physical file (`getTotalNumVGPRs`) | **1024** | **512** (same SRAM; wave64 counts double) |
| Alloc granule (`getVGPRAllocGranule`) | **16** | **8** |
| Encoding granule (`getVGPREncodingGranule`) | **8** | **4** |
| Addressable arch VGPRs | **256** | **256** |
| Max waves/SIMD (`getMaxWavesPerEU`) | **16** | **16** |

**Alloc vs encoding granule:** occupancy / SPI allocation uses **16** on wave32 GFX10.3+. The metadata encoding block size is still **8**. Design and `llvm-calc-occupancy --vgprs=` use the **alloc** round; do not size tiles off the encoding granule.

GPUOpen Occupancy explained: RDNA2/3 have **16** slots per SIMD; SGPRs are fixed enough that they never limit those slots; VGPR is one of the four PIX `WaveOccupancyLimiters` (with LDS, Thread Group Size, Barriers).

## 2. Quick ladder (wave32, alloc round 16)

| `.vgpr_count` (raw) | Aligned | Waves/SIMD | Max VGPR still at this rung (`alignDown(1024/W, 16)`) |
|---|---|---|---|
| 1–15 | — | **16** | 64 |
| 16–64 | 16…64 | **16** | 64 |
| 65–128 | 80…128 | **8** | 128 |
| 129–170 | 144…160 | **6** | 160 |
| 171–256 | 176…256 | **4** | 256 |
| 257–341 | 272…336 | **3** | 336 |
| 342–512 | 352…512 | **2** | 512 |
| 513–1024 | 528…1024 | **1** | 1024 (addressable still 256 unless you spill) |

Practical rungs for extras shapes:

| Target waves/SIMD | Max VGPR/lane | Typical use |
|---|---|---|
| 16 | ≤ 64 | skinny decode / M=1..4 W4 |
| 8 | ≤ 128 | mid decode / 128-thr tiles |
| 4 | ≤ 256 | fat prefill only if LDS still allows the WG pack |

`llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --wg-size=N --vgprs=V` uses this same `GCNSubtarget` math. `--limits` prints the Max-VGPR column per occupancy level.

## 3. `__launch_bounds__` / `amdgpu_waves_per_eu`

| Knob | Meaning on gfx1030 |
|---|---|
| `__launch_bounds__(MAX_THREADS[, MIN_WARPS])` | HIP validates launch ≤ `MAX_THREADS`. Optional `MIN_WARPS` (default **1**) is waves per **EU = SIMD32**; compiler derives a VGPR cap ≈ `alignDown(1024 / MIN_WARPS, 16)`. |
| `amdgpu_waves_per_eu(min[, max])` | Backend limits VGPR/LDS/scratch so occupancy stays in `[min, max]` waves/EU. Prefer over deprecated `amdgpu_num_vgpr`. |
| `amdgpu_flat_work_group_size(min, max)` | Pins legal WG size so the compiler does not budget a 1024-thread block you never launch. |

Do **not** ship `__launch_bounds__(1024)` on a 128- or 256-thread kernel — the compiler then spends VGPR like the block is 1024. Match the first arg to the launch (e.g. `__launch_bounds__(128)` or `(256, 4)`).

HIP docs’ “Compute Unit consists of 4 Execution Units” is the **GCN/WGP** picture (4×SIMD32). ISA CU = 2×SIMD32. Size `MIN_WARPS` against **SIMD32**.

## 4. Workgroup spanning vs per-SIMD VGPR

SPI places a whole workgroup on **one WGP** (WGP mode). Waves of that WG are spread across the WGP’s **4** SIMD32s. VGPR allocation is still **per SIMD**:

```text
N            = ceil(WG_threads / 32)          // waves in the WG
waves_on_SIMD ≈ ceil(N / 4)                   // WGP, even spread
need_per_SIMD = waves_on_SIMD * alignTo(VGPR, 16)
// must be ≤ 1024 or the WG does not fit that SIMD’s file
```

Example (llama.cpp #24672 class, already noted in [hip-craft.md](hip-craft.md) §6.5):

| Config | Threads | VGPR raw→aligned | Compiler “waves/SIMD” in isolation | WG math |
|---|---|---|---|---|
| Death tile | 256 | 203→**208** | 1024/208 = **4** | N=8 → ~2 waves/SIMD → 2×208=416 ≤ 1024 (theory OK) |
| Runtime | same | same | — | `hipOccupancy*` / launch can still be **0** (CU-mode accounting, hidden scratch, or API CU vs WGP mismatch) |

**Operational rule:** if the runtime occupancy query is 0, the kernel does not launch — regardless of `llvm-calc-occupancy`. Reject spills (`.vgpr_spill_count > 0`); scratch is latency poison and only rarely cuts `waves_per_cu` ([scratch-occupancy.md](scratch-occupancy.md)).

For a **128-thread** WG (N=4 → ~1 wave/SIMD): the per-SIMD budget is simply `alignTo(VGPR, 16) ≤ 1024`, so the isolation ladder matches SPI. Prefer 128-thr when you need high VGPR without multi-wave-per-SIMD packing risk.

## 5. Operational rules

1. Dump `.vgpr_count` from NT_AMDGPU_METADATA ([occupancy-dump.md](occupancy-dump.md)); round up to 16; apply §2.
2. Fold with siblings: final WGs/WGP = `min(VGPR-implied, MaxWGsLDS, barrier WGs, slot/WG)`. Fat FA tiles usually hit **LDS** first; skinny W4/EXL3 decode hits **VGPR** or grid fill first.
3. Pin occupancy at compile time (`waves_per_eu` / `__launch_bounds__`) **and** assert `hipOccupancyMaxActiveBlocksPerMultiprocessor > 0` before capture/graph install.
4. RGA live-VGPR column finds the instruction that bought the next 16-VGPR granule — one hot spike sets the whole kernel.
5. Occupancy is latency-hiding capacity. Raising waves/SIMD by starving the tile of live values is a measured trade (GPUOpen: peak occupancy ≠ peak perf when caches thrash).

## 6. Sources

1. LLVM `IsaInfo::getVGPRAllocGranule` / `getVGPREncodingGranule` / `getTotalNumVGPRs` / `getAddressableNumArchVGPRs` / `getNumWavesPerEUWithNumVGPRs` / `getMaxNumVGPRs` / `isSGPROccupancyLimited` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
2. LLVM `llvm-calc-occupancy` — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
3. RDNA 2 ISA 70648 §3.6.4 (VGPR groups of 16 dwords wave32 / 8 wave64) — https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
4. GPUOpen Occupancy explained (updated 2024-06-26) — https://gpuopen.com/learn/occupancy-explained/ — 16 slots/SIMD RDNA2; SGPR unlimited; VGPR limiter; RGA live-VGPR
5. Clang `amdgpu_waves_per_eu` — https://clang.llvm.org/docs/AttributeReference.html#amdgpu-waves-per-eu
6. HIP `__launch_bounds__` — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
7. llama.cpp #24672 / #23310 — 203 VGPR occupancy-0 class
8. Wiki priors: [architecture.md](architecture.md), [hip-craft.md](hip-craft.md), [occupancy-dump.md](occupancy-dump.md), [lds-occupancy.md](lds-occupancy.md)

Idle pass 2026-09-08 (Paris). Researcher lane only.
