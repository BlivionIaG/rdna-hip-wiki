# LDS bytes vs occupancy (gfx1030)

Lock: **LDS is the WG-packing limiter for our fat tiles, and the math is mode-split.** Physical LDS is **128 KB** on a WGP (`FeatureHalfAddressablePhysicalLocalMemory`), but one workgroup may address only **64 KB**. Occupancy uses `getLocalMemorySize()` (128 KB WGP / 64 KB CU), not the per-WG addressable cap, after aligning the kernel’s LDS to the **compiler** granule. Barriers / VGPR / scratch stay on their own pages — this page is the LDS term in `min(slots, VGPR, LDS, WG/barrier)`.

Does **not** change extras HIP or UNC cards. No tok/s. Do not restate the FA pin.

Companions: [lds-tiles.md](lds-tiles.md), [barrier-occupancy.md](barrier-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [occupancy-dump.md](occupancy-dump.md), [hip-craft.md](hip-craft.md) §6, [architecture.md](architecture.md) §2.4 / §4.2, [wg-size-occupancy.md](wg-size-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | WGP mode: `MaxWGsLDS = floor(131072 / alignTo(LDSBytes, Gran))`. CU mode (`-mcumode`): denominator pool is **65536**. Per-WG hard cap stays **65536** either way (ISA + `getAddressableLocalMemorySize`). |
| **Take** | For occupancy dumps, use **launch** LDS bytes when `.group_segment_fixed_size == 0` (dynamic `extern __shared__`) — [occupancy-dump.md](occupancy-dump.md) §5. |
| **Take** | Size tiles so `alignTo(LDS, 1024) ≤ 32768` when you want **≥ 4 WGs/WGP**, or `≤ 65536` for the 2-WG ceiling. Round **up** before the divide. |
| **Take (craft)** | Know the **LLVM vs ISA granule mismatch** (§3). For borderline sizes in `(n·1024, n·1024+512]`, LLVM can report one more concurrent WG than SPI will pack. Prefer the **ISA 1 KB** round when you care about measured `hipOccupancy*`. |
| **Leave** | Do not flip `-mcumode` to “get more LDS WGs” — CU mode **halves** the shared pool (128→64 KB) and SIMDs. Only measure for LDS-*bandwidth* ([hip-craft.md](hip-craft.md) §1.2). |
| **Leave** | Do not open UNC / retip extras because of the granule mismatch. At 45–64 KB tiles both granules still give **2 WGs/WGP**. |

## 1. LLVM formula (source of truth)

From `AMDGPUSubtarget::getOccupancyWithWorkGroupSizes` (`AMDGPUSubtarget.cpp`, main tip 2026-09-07) and `IsaInfo::getLocalMemorySize` (`AMDGPUBaseInfo.cpp`):

```text
// Pool for concurrent WGs on the block the WG must share:
Physical = getMaxHWAddressableLocalMemorySize()   // gfx10: 65536
         * (FeatureHalfAddressablePhysicalLocalMemory ? 2 : 1)
         // → 131072 physical on gfx10/11/12
LocalMemorySize = Physical / (isFullSIMDMode ? 1 : 2)
                // WGP (HIP default): 131072
                // CU (-mcumode):       65536

Granularity = getLdsDwGranularity(STI) * 4   // bytes; see §3
LDSBytes    = alignTo(kernel_lds_bytes, Granularity)
MaxWGsLDS   = LocalMemorySize / max(LDSBytes, 1)   // integer divide
if MaxWGsLDS == 0: treat occupancy as 1   // LDS > pool

WGsPerCU = min(getMaxWorkGroupsPerCU(WGSize), MaxWGsLDS)  // barriers fold here
waves/EU = clamp( waves_from(WGsPerCU, WGSize) / NumWorkGroupSIMDs )
```

| Mode | `isFullSIMDMode` | SIMDs / WG | `LocalMemorySize` | Per-WG addressable |
|---|---|---|---|---|
| WGP (HIP default, `-mno-cumode`) | true | 4 | **131072** | **65536** |
| CU (`-mcumode`) | false | 2 | **65536** | **65536** |

`getAddressableLocalMemorySize` = `min(65536, LocalMemorySize)` — a single WG still cannot allocate more than 64 KB in WGP mode even though the WGP holds 128 KB for **two** WGs (ISA §2.3.1; LLVM comment on `getAddressableLocalMemorySize`).

GPUOpen Occupancy explained: LDS is one of the four PIX `WaveOccupancyLimiters` (with VGPR, Thread Group Size, Barriers). SGPRs never limit on RDNA.

## 2. Quick ladder (WGP, ISA 1 KB round)

Use this table for tile design. Bytes are **after** `alignTo(_, 1024)`.

| LDS / WG (aligned) | `MaxWGsLDS` (WGP) | `MaxWGsLDS` (CU) | Waves/WGP at WG=256 (`N=8`) if LDS binds |
|---|---|---|---|
| ≤ 1024 | 128 | 64 | barrier / slots bind first |
| 8192 | 16 | 8 | `min(8,16)=8` → 64 waves (slot cap) |
| 16384 | 8 | 4 | 8 → 64 waves |
| 32768 | **4** | 2 | 4 → **32 waves = 8/SIMD** |
| 36864–65536 | **2** | 1 | 2 → **16 waves = 4/SIMD** |
| > 65536 | illegal (WG addressable cap) | illegal | — |

So: every KB you keep above 32 KB on a 256-thread tile is buying **half** the LDS-limited WG count (4→2). That is the occupancy move for attention/softmax scratch — shrink the tile, do not chase barriers ([barrier-occupancy.md](barrier-occupancy.md)).

Skinny decode / GDN paths with tiny or zero LDS are **not** on this ladder; they fall back to VGPR / slots / grid fill.

## 3. Granule: ISA 1 KB vs LLVM 512 B

| Source | Granule | Note |
|---|---|---|
| RDNA 2 ISA 70648 §3.6.6 | **256 dwords = 1024 B**, 256-dword aligned | Feature note: “VGPR & LDS allocation-unit size **doubled**” vs RDNA 1 |
| LLVM `getLdsDwGranularity` for `FeatureAddressableLocalMemorySize65536` (gfx7…gfx12, **includes gfx1030**) | **128 dwords = 512 B** | `GCNSubtarget` sets `LDSAllocationGranularity = dw * 4` |
| Wiki priors / `lds-tiles.md` | Prefer **1024 B** for SPI / design | Matches ISA |

`getLdsDwGranularity` keys only on addressable-size features — it does **not** special-case `hasGFX10_3Insts`. So the **compiler’s** `alignTo` for occupancy / `amdgpu-waves-per-eu` budgeting still uses **512 B** on gfx1030, while hardware allocation is **1 KB**.

When the raw size sits in `(n·1024, n·1024 + 512]`, the two rounds diverge. Example:

```text
LDSBytes = 43009
alignTo(_, 512)  = 43520  →  MaxWGsLDS = 131072/43520 = 3
alignTo(_, 1024) = 44032  →  MaxWGsLDS = 131072/44032 = 2
```

Operational rule: **design and `hipOccupancy*` gates use ISA 1 KB.** Treat LLVM/`llvm-calc-occupancy` as optimistic by at most one WG on those edge sizes. Power-of-two tiles (8/16/32/64 KB) agree on both granules.

## 4. Dynamic shared and `amdgpu-lds-size`

| Path | What occupancy must use |
|---|---|
| Static `__shared__` / AS(3) globals | `.group_segment_fixed_size` from NT_AMDGPU_METADATA |
| `extern __shared__` + launch `smem` | **host** byte count (metadata often **0**) |
| LLVM attr `"amdgpu-lds-size"="min[,max]"` | `min` feeds `getWavesPerEU` as the LDS floor for codegen |

Double-buffered / staged tiles: count **all** live LDS at the fattest pipeline point (A+B, or K+V+softmax scratch), then align. Radiance-style `TILE×head×el×stages+256 ≤ 65536` clamp ([lds-tiles.md](lds-tiles.md)) is the capture-safe form of the same budget.

## 5. Operational rules

1. Compute `MaxWGsLDS` with **WGP 128 KB** pool unless the binary was built `-mcumode`.
2. Fold barriers: `WGs = min(MaxWGsLDS, getMaxWorkGroupsPerCU(WGSize))` — never attribute a 45–60 KB tile’s 1–2 WG pack to barriers.
3. Reject `LDSBytes > 65536` at compile/launch; SPI will not place the WG.
4. `llvm-calc-occupancy -mcpu=gfx1030 --wg-size=… --vgprs=… --lds=…` — pass **aligned launch LDS**, not raw metadata 0.
5. Occupancy is latency-hiding capacity. An LDS-bound FA tile at 2 WGs/WGP can still be the right shape; raising occupancy by starving the tile of K/V footprint is a measured trade, not a free win (GPUOpen: peak occupancy ≠ peak perf when caches thrash).

## 6. Sources

1. LLVM `IsaInfo::getLocalMemorySize` / `getAddressableLocalMemorySize` / `getPhysicalLocalMemorySize` / `getLdsDwGranularity` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
2. LLVM `AMDGPUSubtarget::getOccupancyWithWorkGroupSizes` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
3. LLVM commit `10cef70` — “Clean up LDS-related occupancy calculations” (WGP pool vs 64 KB addressable)
4. RDNA 2 ISA 70648 §2.3.1 / §3.6.6 / §10.3 — https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
5. GPUOpen Occupancy explained (updated 2024-06-26) — https://gpuopen.com/learn/occupancy-explained/ — LDS limiter; SGPR unlimited on RDNA
6. Wiki priors: [architecture.md](architecture.md), [lds-tiles.md](lds-tiles.md), [occupancy-dump.md](occupancy-dump.md), [barrier-occupancy.md](barrier-occupancy.md)

Idle pass 2026-09-07 (Paris). Researcher lane only.
