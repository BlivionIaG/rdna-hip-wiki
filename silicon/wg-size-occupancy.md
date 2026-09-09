# Thread Group Size vs occupancy (gfx1030)

Lock: **PIX "Limited by Thread Group Size" is the atomic WG resource-lifetime limiter — not Barriers, not LDS, not VGPR.** On gfx1030 HIP default (**wave32 + WGP**), SPI allocates and frees **all** waves of a workgroup together (slots + VGPR + LDS + barrier). When one wave of a multi-wave WG retires, its SIMD slot can free while the rest of the WG still holds registers / LDS / barrier — that empty slot cannot start a *new* WG until the last wave finishes. That is Thread Group Size limited. Barriers only bind after a *different* WG frees enough slots/resources but every barrier is still held ([barrier-occupancy.md](barrier-occupancy.md)). FA / GDN fat tiles stay **LDS-bound** long before WG packing alone is the leftover. Do **not** retip `__launch_bounds__` or open UNC for this page.

Does **not** change extras HIP or UNC cards. No tok/s. Do not restate the FA pin.

Companions: [barrier-occupancy.md](barrier-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [lds-occupancy.md](lds-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [occupancy-dump.md](occupancy-dump.md), [hip-craft.md](hip-craft.md) §1.3 / §6, [architecture.md](architecture.md) §2.4 / §3, [fa-occupancy.md](fa-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat Thread Group Size and Barriers as **two different PIX counters**. Both need `N ≥ 2` waves/WG. TG-size = mid-WG retire left a free slot that cannot start a new WG. Barriers = slots+resources free, no barrier left. |
| **Take** | Pin `__attribute__((amdgpu_flat_work_group_size(min, max)))` to the **real** launch flat size (product of dim). Default IR is `1,1024`; Clang `0,0` implies `128,256`. Unpinned max budgets a 1024-thread block you never launch and under-states waves/EU. |
| **Take** | LLVM occupancy w.r.t. WG size is a **range**: `getOccupancyWithWorkGroupSizes(LDS, {min,max})` → `{min_waves/EU, max_waves/EU}`. `llvm-calc-occupancy` without a fixed `--wg-size` prints a range for the same reason (PR #123748). |
| **Take** | For multi-wave WGs, theoretical WGs/WGP (ignoring LDS) = `min(MaxWaves/N, MaxBarriers)` with `MaxWaves=64` WGP / `32` CU and `MaxBarriers=32` WGP / `16` CU — same `getMaxWorkGroupsPerCU` math as the barrier page. TG-size is the *dynamic* hole that formula does not show. |
| **Leave** | Do not shrink FA/GDN from 256→128 thr solely to chase PIX "Thread Group Size". Fat LDS (45–64 KB) already caps at **1 WG / 64 KB addressable** ([lds-occupancy.md](lds-occupancy.md)). Cutting threads without cutting LDS does not raise concurrent WGs. |
| **Leave** | Do not open UNC / retip extras for this consolidation. Occupancy leftover stays FA prefill LDS class. |

## 1. What Thread Group Size means (GPUOpen / PIX)

From GPUOpen Occupancy explained (updated 2024-06-26) and PIX `WaveOccupancyLimiters`:

| Limiter | Static / dynamic | Means on gfx1030 compute |
|---|---|---|
| VGPR | mostly static | not enough VGPRs for another wave ([vgpr-occupancy.md](vgpr-occupancy.md)) |
| LDS | mostly static | not enough LDS for another WG ([lds-occupancy.md](lds-occupancy.md)) |
| **Thread Group Size** | **dynamic** | a multi-wave WG holds **all** its resources until **every** wave retires; a freed wave slot mid-WG cannot start a new WG |
| Barriers | dynamic, rare | slots + VGPR/LDS free enough for a new multi-wave WG, but **no barrier** left ([barrier-occupancy.md](barrier-occupancy.md)) |

GPUOpen walkthrough (pair of SIMDs / one ISA CU, scale ×2 for WGP):

1. Multi-wave WG needs ≥ 2 wave slots + 1 barrier.
2. Wave A of WG1 finishes → one slot frees, but VGPR/LDS/barrier stay held for wave B → **Thread Group Size limited**.
3. Wave B finishes → whole WG frees. If another WG still holds the last barrier while two slots are free → **Barriers limited** (GPUOpen: "very specific … super rare").

Resources for a compute WG are allocated and deallocated **together** (GPUOpen large-thread-group article: registers, LDS, and wave slots). That atomic lifetime is why TG-size is a limiter even when the static `min(VGPR, LDS, barriers)` formula looks fine.

## 2. LLVM packing (theoretical WGs / waves)

`IsaInfo::getWavesPerWorkGroup` / `getMaxWorkGroupsPerCU` (`AMDGPUBaseInfo.cpp`, main tip 2026-09-09):

```text
N        = ceil(FlatWorkGroupSize / WavefrontSize)   // wave32 → WG/32
MaxWaves = getMaxWavesPerEU(gfx1030) * getNumWorkGroupSIMDs(WGP|CU)
         = 16 * 4 = 64   // WGP (HIP default, -mno-cumode)
         = 16 * 2 = 32   // CU (-mcumode)

if N == 1:
  MaxWGs = MaxWaves          // single-wave: no barrier; TG-size hole does not apply
else:
  MaxBarriers = 32 (WGP) / 16 (CU)
  MaxWGs = min(MaxWaves / N, MaxBarriers)
```

`AMDGPUSubtarget::getOccupancyWithWorkGroupSizes` (PR #123748 / `AMDGPUSubtarget.cpp`):

```text
MaxWGsLDS = LocalMemorySize / alignTo(LDSBytes, LDSAllocationGranularity)
WGsPerCU  = min(getMaxWorkGroupsPerCU(WGSize), MaxWGsLDS)
Waves/CU  = WavesPerWG * WGsPerCU
waves/EU  = spread Waves/CU across getNumWorkGroupSIMDs()   // clamp to [1, 16]

// Over FlatWorkGroupSizes = {min, max} this returns a *range*
// of achievable waves/EU (max WG size → usually min occupancy).
```

| Flat WG (wave32) | `N` waves | Max WGs WGP (no LDS) | Notes |
|---|---|---|---|
| 32 | 1 | **64** | no barrier; no TG-size atomic hole |
| 64 | 2 | `min(32, 32) = 32` | TG-size can show mid-pair retire |
| 128 | 4 | `min(16, 32) = 16` | FA decode / AWQ prefill class |
| 256 | 8 | `min(8, 32) = 8` | FA prefill class; **LDS 45–64 KB → 1–2 WGs** wins first |
| 512 | 16 | `min(4, 32) = 4` | rare on extras |
| 1024 | 32 | `min(2, 32) = 2` | IR default max; never leave this as the compile budget |

Pass the **real** `--wg-size=` to `llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32`. Unspecified / ranged WG size → ranged occupancy (tool docs).

## 3. `amdgpu_flat_work_group_size` craft

| Attribute | Effect |
|---|---|
| `amdgpu_flat_work_group_size(min, max)` | Pins legal flat launch size; better barrier codegen + scratch promotion estimate (Clang AttributeReference). |
| `(0, 0)` | Clang default **128, 256**. |
| IR omit | Backend default **1, 1024** (`AMDGPUUsage` / `getDefaultFlatWorkGroupSize` for kernels). |
| vs `amdgpu_waves_per_eu` | If incompatible, **WG-size occupancy bound wins** (`AMDGPUUsage`). |
| vs `__launch_bounds__(MAX_THREADS, MIN_WARPS)` | Set `MAX_THREADS` to the real block size (never 1024 on a 128/256-thr kernel). Prefer `waves_per_eu` + `flat_work_group_size` when you need an explicit VGPR ceiling ([vgpr-occupancy.md](vgpr-occupancy.md) §3). |

```cpp
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void k(...) { ... }
```

Assert `hipOccupancyMaxActiveBlocksPerMultiprocessor > 0` before graph capture ([occupancy-dump.md](occupancy-dump.md)). Runtime flat size outside `[min,max]` is undefined behavior (AMDGPUUsage).

## 4. Extras shapes — when TG-size matters

| Shape | Threads | `N` | Static binder | TG-size relevance |
|---|---|---|---|---|
| causal_conv1d update | 32 | 1 | VALU / grid | **none** (single-wave) |
| AWQ prefill `q_gemm_rdna2_awq_prefill` | 128 | 4 | LDS A-tile + VGPR | mild; LDS/VGPR first |
| FA decode 128 / split-K 256 | 128–256 | 4–8 | LDS / VGPR | possible mid-WG hole; not the leftover card |
| FA prefill 256, LDS 45–64 KB | 256 | 8 | **LDS → 1 WG / 64 KB** | TG-size secondary; shrinking thr without LDS does nothing |
| GDN wy ~56–58 KB | 128–256 | 4–8 | **LDS** | same |
| Hypothetical 64-thr, ~0 LDS | 64 | 2 | wave slots / barriers | **TG-size + barriers** are the rare PIX pair |

## 5. Operational rules

1. When PIX / RGP shows **Thread Group Size > 0%**, read it as binary ("this is a limiter"), not as severity — GPUOpen: limiter % is clocks failing to launch / total clocks, not an occupancy fraction.
2. Do **not** conflate with Barriers. Diagnose: if LDS ≥ 32 KB on a 128–256 thr WG, attribute low `hipOccupancy*` to LDS first ([lds-occupancy.md](lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md) §4).
3. Always pin `amdgpu_flat_work_group_size` to the launched flat size; pair with `amdgpu_waves_per_eu` / `__launch_bounds__` for the VGPR term.
4. Prefer **two** concurrent WGs on a WGP when LDS allows (≤ 32 KB each) — GPUOpen large-thread-group: overlapping allocate/free hides the TG-size ramp-down. Fat 64 KB tiles cannot do this; that is the LDS card, not a thr-count card.
5. Single-wave WGs (`N == 1`) skip both TG-size and barrier limiters — use when the algorithm fits 32 thr and needs max slot fill (causal_conv update path).
6. Peak occupancy ≠ peak perf when IC/L2 thrash ([infinity-cache.md](infinity-cache.md)); TG-size cleanup is latency-hiding capacity only.

## 6. Sources

1. GPUOpen Occupancy explained (updated 2024-06-26) — https://gpuopen.com/learn/occupancy-explained/ — PIX Thread Group Size vs Barriers walkthrough
2. GPUOpen Optimizing GPU occupancy with large thread groups — https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/ — atomic WG allocate/free
3. PIX hardware counters — https://devblogs.microsoft.com/pix/hardware-counters-in-gpu-captures/ — `CSLimitedByThreadGroupLimit`
4. LLVM `IsaInfo::getMaxWorkGroupsPerCU` / `getWavesPerWorkGroup` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
5. LLVM `AMDGPUSubtarget::getOccupancyWithWorkGroupSizes` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp — PR #123748
6. `llvm-calc-occupancy` — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html — ranged occupancy when WG size unspecified
7. Clang `amdgpu_flat_work_group_size` — https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
8. LLVM AMDGPUUsage `amdgpu-flat-work-group-size` / waves-per-eu precedence
9. Wiki priors: [barrier-occupancy.md](barrier-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [lds-occupancy.md](lds-occupancy.md), [hip-craft.md](hip-craft.md)

Idle pass 2026-09-09 (Paris). Researcher lane only.
