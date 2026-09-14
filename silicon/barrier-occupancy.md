# Barrier slots vs occupancy (gfx1030)

Lock: **hardware barriers are a real WG packing limit, but they are not the extras leftover.** On gfx1030 HIP default (**wave32 + WGP**), LLVM budgets **32** barriers per WGP; CU mode budgets **16**. A **single-wave** workgroup does **not** consume a barrier. FA / GDN / EXL3 tiles stay LDS- or VGPR-bound long before barriers bind. Do **not** retip `__launch_bounds__` or open a ticket for barriers.

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [wg-size-occupancy.md](wg-size-occupancy.md), [hip-craft.md](hip-craft.md) §6, [architecture.md](architecture.md) §2.4 / §3, [lds-tiles.md](lds-tiles.md), [fa-occupancy.md](fa-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [lds-occupancy.md](lds-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md), [occupancy-composite.md](occupancy-composite.md).

## 1. LLVM formula (source of truth)

From `llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp` `IsaInfo::getMaxWorkGroupsPerCU` (main tip, 2026-09-03):

```text
MaxWaves = getMaxWavesPerEU(Kind) * getNumWorkGroupSIMDs(isFullSIMDMode)
         = 16 * 4  // WGP / full-SIMD mode
         = 16 * 2  // CU mode (-mcumode)
N = ceil(FlatWorkGroupSize / WavefrontSize)   // wave32 → N = WG/32

if N == 1:
  return MaxWaves          // single-wave WGs do not consume barriers
else:
  MaxBarriers = 16
  if GFX10+ and not FeatureCuMode:
    MaxBarriers = 32       // WGP mode
  return min(MaxWaves / N, MaxBarriers)
```

| Mode | SIMDs in the “CU” LLVM means | Wave slots | Barriers | Max multi-wave WGs |
|---|---|---|---|---|
| WGP (HIP default, `-mno-cumode`) | 4 | 64 | **32** | `min(64/N, 32)` |
| CU (`-mcumode`) | 2 | 32 | **16** | `min(32/N, 16)` |

GPUOpen Occupancy explained matches the CU half: “up to **16** barriers in flight **per pair of SIMDs**.” Pair of SIMDs = one ISA CU; WGP = two pairs → 32. Same numbers as LLVM.

`isSGPROccupancyLimited` stays **false** on GFX10+ — SGPRs are never in this min().

## 2. PIX / GPUOpen limiter names (do not confuse)

AMD PIX `WaveOccupancyLimiters` (GPUOpen Occupancy explained) splits four static/dynamic counters:

| Limiter | Means | gfx1030 HIP note |
|---|---|---|
| VGPR | not enough VGPRs for another wave / WG | real for fat decode/GEMM |
| LDS | not enough LDS for another WG | **FA prefill leftover** (1 WG / 64 KB) |
| Thread Group Size | a multi-wave WG holds slots/resources until **all** its waves retire; a freed slot cannot start a new WG mid-barrier | common with 2–8 wave WGs |
| Barriers | slots + VGPR/LDS free, but **no barrier left** to launch another multi-wave WG | GPUOpen: “very specific … super rare” |

**Thread Group Size ≠ Barriers.** Both need `N ≥ 2`. Barrier-limited only after a *different* WG’s wave frees a slot while every barrier is still held. Do not treat a PIX “Barriers > 0%” blip as a reason to shrink FA tiles.

Measured occupancy can also sit under theoretical for **lack of work** (grid too small for 40 WGP × 4 SIMD × 16) or **launch-rate** drain — those are SPI/ACE issues, not barrier HW. Occupancy is latency-hiding capacity; ALU-bound kernels do not want more waves.

## 3. Extras shapes — barriers never bind first

Wave32. One barrier per multi-wave WG.

| Shape | Threads | Waves `N` | Barrier WGs allowed (WGP) | What actually binds |
|---|---|---|---|---|
| FA decode 128 | 128 | 4 | `min(64/4, 32) = 16` | VGPR mild; LDS decode tiles smaller |
| FA decode / prefill 256 | 256 | 8 | `min(64/8, 32) = 8` | **LDS 45–60 KB → 1 WG / 64 KB** |
| GDN o / wy ~56–58 KB | 128–256 | 4–8 | 8–16 | **LDS** |
| RMSNorm AOT | 128–1024 | 4–32 | `min(64/N, 32)` | tiny LDS; grid / VALU |
| Hypothetical 64-thr, 0 LDS | 64 | 2 | **32** (= barrier ceiling) | **barriers** (the rare case) |
| Single-wave 32-thr | 32 | 1 | **64** (no barrier cost) | wave slots / VGPR only |

So: shrinking BR/BC to free LDS is still the occupancy move for FA. Cutting threads to chase barriers is wrong for our tiles — you would only approach the barrier ceiling with **many tiny multi-wave WGs and near-zero LDS**.

## 4. Operational rules

1. Treat barrier budget as **`min(MaxWaves/N, 32)`** in WGP mode; ignore for `N == 1`.
2. When dumping occupancy ([occupancy-dump.md](occupancy-dump.md)), if LDS ≥ 32 KB on a 128–256 thr WG, **do not** attribute low `hipOccupancy*` to barriers.
3. Do **not** flip `-mcumode` to “get more barriers” — CU mode **halves** barrier count (32→16) and SIMDs; only measure for LDS-bandwidth-bound kernels ([hip-craft.md](hip-craft.md) §1.2).
4. Named barriers / `amdgpu.max_num_named_barrier` (newer LLVM `MCResourceInfo`) are **not** a gfx1030 extras dest. Stick to `__syncthreads()` / `s_barrier`.
5. `llvm-calc-occupancy -mcpu=gfx1030 --wg-size=… --vgprs=… --lds=…` already folds the WG/barrier packing via the same `GCNSubtarget` math; it has no separate `--barriers` flag — pass the real WG size.

## 5. Sources

1. LLVM `IsaInfo::getMaxWorkGroupsPerCU` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
2. LLVM `isSGPROccupancyLimited` (Major < 10) — same file
3. GPUOpen Occupancy explained (updated 2024-06-26) — https://gpuopen.com/learn/occupancy-explained/ — 16 barriers / SIMD pair; PIX limiters; Thread Group Size vs Barriers
4. `llvm-calc-occupancy` — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
5. RDNA 2 ISA 70648 §2.3.1 / §10.3 — WGP vs CU, `s_barrier` scope
6. Wiki priors: [architecture.md](architecture.md), [hip-craft.md](hip-craft.md), [lds-tiles.md](lds-tiles.md)

Idle pass 2026-09-03.
