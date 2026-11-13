---
title: "amdgpu_waves_per_eu max as the scheduler's occupancy target, and the silent drop (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/include/llvm/TargetParser/AMDGPUTargetParser.h
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/SIMachineFunctionInfo.cpp
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/GCNSubtarget.cpp
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUOptions.td
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/docs/AMDGPUUsage.md
  - https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/clang/include/clang/Basic/AttrDocs.td
  - https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/fa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/gdn_prefill_delta_h_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu
  - https://github.com/razz97/Avalon-RDNA2/blob/f792cc6a25ddb2ba76d29dd7bed3b6307da7f18f/ggml/src/ggml-cuda/fattn-qsa-rdna2.cu
  - https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
---

# `amdgpu_waves_per_eu` max as the scheduler's occupancy target, and the silent drop (gfx1030)

Lock: **the second (max) argument of `amdgpu_waves_per_eu(min, max)` is not only a cap on reported occupancy. It sets the occupancy the default GCN machine scheduler aims for, so it is a register-pressure license.** With max = 8, the scheduler may use up to the 8-wave VGPR budget (128) to hoist loads and gain ILP, even when the same source fits 16 waves without the attribute. A second fact, also new to this wiki: **a range whose min is below the waves-per-SIMD that the flat work-group size implies is thrown away without any diagnostic.** For a 256-thread gfx1030 WGP-mode kernel, `(1, 4)` gives the same code as no attribute at all.[^eff][^wg][^probe]

Companions (the min / VGPR-ceiling side is already covered there, not repeated): [vgpr-occupancy.md](vgpr-occupancy.md) §3, [hip-craft.md](../hip-craft/hip-craft.md) §1.3, [fa-occupancy.md](../attention/fa-occupancy.md) §1 (`__launch_bounds__` lowering), [wg-size-occupancy.md](wg-size-occupancy.md), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [occupancy-dump.md](occupancy-dump.md).

Does **not** change extras HIP or a UNC card (see §5). No tok/s. Do not restate the FA pin.

Date: **2026-11-13** Europe/Paris.

## Take / Leave

| | |
|---|---|
| **Take (Now, craft)** | Treat `max` as a deliberate request: "below this many waves/SIMD, spend the registers on ILP". On the probe kernel `(4, 8)` took VGPRs from 25 to 103 and the largest `vmcnt(N)` wait from 4 to 58. Fewer resident waves, many more loads in flight per wave. Pick it when that is the trade you want, and dump the result ([occupancy-dump.md](occupancy-dump.md)). Do not add a max just to "document" a target.[^probe][^sched-init] |
| **Take (Now, craft)** | Keep `min ≥ ceil(ceil(flat_wg_max / 32) / 4)` on gfx1030 WGP mode (`/ 2` with `-mcumode`). Otherwise the whole attribute, max included, is replaced by the default range. Clang and the backend print nothing (checked with `-Wall -Rpass-analysis=kernel-resource-usage`).[^eff][^wg][^simds][^probe] |
| **Take (Later, A/B only)** | `-mllvm -amdgpu-sched-strategy=max-ilp` / `max-memory-clause` (or IR attribute `"amdgpu-sched-strategy"`) skip the occupancy stages altogether. On the probe they gave 104 VGPRs and 9 waves with no attribute. The option is accepted by ROCm 7.0.1 – 10.0 clang and rejected by ROCm 6.4. It is an undocumented backend knob (not in AMDGPUUsage), so it stays experimental.[^select][^opt][^stages][^probe] |
| **Leave** | Reading reported `Occupancy` as "the hardware occupancy". With a max, the compiler reports `min(VGPR-limited, max)`, e.g. 4 for `(2, 4)` at 103 VGPRs, although 103 VGPRs alone would allow 9.[^occ][^probe] |

## 1. Where the range is resolved

`AMDGPUSubtarget::getEffectiveWavesPerEU` builds a default range from the flat work-group size and LDS:

- **min** = `getWavesPerEUForWorkGroup(flat_wg_max)`;
- **max** = the occupancy the work-group sizes and LDS allow.

If the requested min is **below** the default min, above the default max, above the requested max, or the requested max is above the target maximum, it **returns the default range**. Otherwise it keeps the request, with the max clamped to the default max.[^eff]

`getWavesPerEUForWorkGroup` = `divideCeil(divideCeil(flat_wg_size, wave_size), getNumWorkGroupSIMDs(full_simd_mode))`.[^wg] `getNumWorkGroupSIMDs` is 4 in full-SIMD mode, 2 otherwise.[^simds] `isFullSIMDMode` is "not `FeatureCuMode`" on everything except gfx1250.[^fullsimd]

Implied minimum on gfx1030 wave32:

| flat WG max | WGP (HIP default) | `-mcumode` |
|---|---|---|
| 128 | 1 | 2 |
| 256 | 2 | 4 |
| 512 | 4 | 8 |
| 1024 (IR default with no `__launch_bounds__`) | 8 | 16 |

Arithmetic from the two functions above.[^wg][^simds] AMDGPUUsage says only that an incompatible range loses to the work-group-size bounds.[^usage-attr] The Clang attribute doc says incompatible values give an error.[^attrdocs] The probe shows no error and no warning in this case (§3).

## 2. How the max becomes the scheduler's target

1. `SIMachineFunctionInfo` stores `WavesPerEU = ST.getWavesPerEU(F)` and `Occupancy = ST.computeOccupancy(F, LDS).second`.[^mfi]
2. `computeOccupancy` takes the max from `getOccupancyWithWorkGroupSizes` and lowers it further for SGPR / VGPR use.[^occ]
3. `GCNSchedStrategy::initialize` sets `TargetOccupancy = MFI.getOccupancy()`. It then sets `VGPRCriticalLimit` from `getMaxNumVGPRs(TargetOccupancy)`. Source comment: this "effectively sets a lower bound on the 'Critical' register limits in the scheduler".[^sched-init]

So with max = 8 the critical VGPR limit is the 8-wave budget, not the 16-wave one. Below that limit the scheduler does not hold back on pressure. That is the 25 → 103 jump in §3. The step from `waves-per-eu` max to `getOccupancyWithWorkGroupSizes` is not traced line by line here. The probe shows the effect, and the reported `Occupancy` equals the requested max.

The default strategy, `GCNMaxOccupancySchedStrategy`, runs `OccInitialSchedule`, `UnclusteredHighRPReschedule`, `ClusteredLowOccupancyReschedule` and `PreRARematerialize` (plus an MFMA rewrite stage behind an option). `GCNMaxILPSchedStrategy` runs only `ILPInitialSchedule` and `GCNMaxMemoryClauseSchedStrategy` only `MemoryClauseInitialSchedule`.[^stages] `createMachineScheduler` picks a strategy by the function attribute `"amdgpu-sched-strategy"` first, then by the `amdgpu-sched-strategy` option. The values are `max-ilp`, `max-memory-clause`, `iterative-ilp`, `iterative-minreg`, `iterative-maxocc` and `coexec` (gfx1250/gfx950 only); anything else gets max-occupancy.[^select][^opt]

## 3. Probe (2026-11-13)

Compiler Explorer API, `--offload-arch=gfx1030 --cuda-device-only -O3`. Values are read from the asm comments `; NumVgprs` / `; Occupancy` / `; ScratchSize` and from the largest `s_waitcnt vmcnt(N)` immediate.[^probe]

Kernel: `__launch_bounds__(256)`, one row per work-group. The loop is unrolled 32× over `h2` (`_Float16` × 2) loads of `a` and `b` into 8 `__builtin_amdgcn_fdot2` accumulators. Indexing uses `__builtin_amdgcn_workgroup_id_x()` / `workitem_id_x()` and a constant stride of 256.

```cpp
#include <hip/hip_runtime.h>
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
// Probe: many independent half2 loads feeding fdot2 accumulators.
extern "C" __global__ void __launch_bounds__(256)
dot_rows(const h2* __restrict__ a, const h2* __restrict__ b,
         float* __restrict__ out, int n) {
  int row = __builtin_amdgcn_workgroup_id_x();
  int t = __builtin_amdgcn_workitem_id_x();
  const h2* ar = a + (size_t)row * n;
  float acc[8] = {0,0,0,0,0,0,0,0};
  for (int k = t; k < n; k += 256 * 32) {
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
      int kk = k + j * 256;
      acc[j & 7] = __builtin_amdgcn_fdot2(
          ar[kk],
          b[kk],
          acc[j & 7], false);
    }
  }
  float s = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) s += acc[j];
  out[row * 256 + t] = s;
}
```

Variants replace `__launch_bounds__(256)` with `__attribute__((amdgpu_waves_per_eu(...))) __launch_bounds__(N)` (or `__launch_bounds__(256, 4)`).

| ROCm clang | Attribute / flag | VGPR | Reported occ | Scratch | max `vmcnt` |
|---|---|---|---|---|---|
| 7.14.0 | none | 25 | 16 | 0 | 4 |
| 7.14.0 | `waves_per_eu(1, 4)` (min 1 < implied 2 → dropped) | 25 | 16 | 0 | 4 |
| 7.14.0 | `waves_per_eu(2, 4)` | 103 | 4 | 0 | 58 |
| 7.14.0 | `waves_per_eu(4, 8)` | 103 | 8 | 0 | 58 |
| 7.14.0 | `waves_per_eu(8, 16)` / `waves_per_eu(8)` / `__launch_bounds__(256, 4)` | 25 | 16 | 0 | 4 |
| 7.14.0 | `__launch_bounds__(128)` + `waves_per_eu(1, 4)` (implied 1 → kept) | 103 | 4 | 0 | 58 |
| 7.14.0 | `-mcumode` + `waves_per_eu(2, 4)` (implied 4 → dropped) | 25 | 16 | 0 | 4 |
| 7.14.0 | `-mcumode` + `waves_per_eu(4, 8)` | 103 | 8 | 0 | 58 |
| 6.4.0, 7.0.1, 10.0.0 | none / `(1, 4)` / `(2, 4)` / `(4, 8)` | 25 / 25 / 103 / 103 | 16 / 16 / 4 / 8 | 0 | 4 / 4 / 58 / 58 |
| 7.14.0, 7.0.1, 7.2.1, 10.0.0 | `-mllvm -amdgpu-sched-strategy=max-ilp` | 104 | 9 | 0 | 61 |
| 7.14.0, 7.0.1, 7.2.1, 10.0.0 | `-mllvm -amdgpu-sched-strategy=max-memory-clause` | 104 | 9 | 0 | 48 |
| 7.14.0, 7.0.1, 7.2.1, 10.0.0 | `-mllvm -amdgpu-sched-strategy=iterative-minreg` | 33 | 16 | 0 | 3 |
| 6.4.0 | `-mllvm -amdgpu-sched-strategy=…` | error: unknown command line argument | | | |

Second kernel: the same loop with stride `blockDim.x` instead of 256, on ROCm 7.14.0. It checks the 512-thread edge, where the implied min is 4. `__launch_bounds__(512)` + `(4, 8)` is kept: 79 VGPR, occupancy 8, versus 64 / 16 with no attribute. `(2, 4)` and `(3, 8)` are dropped: 64 / 16, the same as no attribute.[^probe]

On `(1, 4)` with `-Wall -Rpass-analysis=kernel-resource-usage`, the exit code is 0 and stderr shows only the resource remarks (VGPRs 25, Occupancy 16). There is no warning about the ignored attribute.[^probe]

Reading: 103 VGPRs round to 112 at granule 16, and 1024 / 112 = 9 waves. So `(4, 8)` is still inside its 8-wave budget, and the reported 8 is the max clamp, not the VGPRs ([vgpr-occupancy.md](vgpr-occupancy.md) for the granule). The 25 → 103 VGPR growth comes from loads being hoisted, as the `vmcnt` depth shows. This page makes **no** claim about whether that is faster.

## 4. Craft rules

1. When you add `amdgpu_waves_per_eu`, write the max only if you accept the scheduler spending up to `1024 / max` (granule 16) VGPRs. To keep only a VGPR ceiling, use the one-argument form or `__launch_bounds__(N, min)`; both leave the max at the default.[^attrdocs][^probe]
2. Check the min against the table in §1 for the **real** `__launch_bounds__` first argument and mode. If the min is too small, the max (the part you probably wanted) silently disappears.[^eff]
3. After an attribute change, dump `.vgpr_count` / `Occupancy` ([occupancy-dump.md](occupancy-dump.md)). Do not infer from source.
4. A scheduler-strategy flag is an A/B experiment, never a dest default (§Take, undocumented).

## 5. Dest / external audit (read-only, no edits)

Every `amdgpu_waves_per_eu(min, max)` in `csrc/rocm/*.cu` at dest `a63a6b00` was checked against §1, assuming HIP's default WGP mode:

| File / kernel | Flat WG | Range | Implied min | Kept? |
|---|---|---|---|---|
| `fa_rdna2.cu` `fa_decode_paged_splitk_kernel` / `_256` | 128 / 256 | (4, 8) | 1 / 2 | kept → 8-wave (128 VGPR) scheduler license[^d-fa] |
| `gdn_prefill_delta_h_rdna2.cu` `gdn_prefill_delta_h_pipe_kernel` | `BV*8`, launched with BV = 64 or 32 → 512 / 256 | (4, 8) | 4 / 2 | kept; the BV = 64 build sits exactly on the edge, so a min of 3 would be dropped[^d-gdn] |
| GDN decode / kkt / o / solve_wy (`GDN_THREADS` / `KKT_THREADS` = 256) | 256 | (2, 4) / (4, 8) | 2 | kept (not individually probed) |
| `mono_k2_rdna2.cu` K2 | 256 | (`WPE`, 16) | 2 | kept; max 16 = no license |
| `skinny_gemms.cu` `wvSplitKrc_` | — | (1, 1) | — | only inside `#if defined(__gfx950__)`. The gfx1030 build gets the `UNREACHABLE_CODE` stub with no attribute.[^d-skinny] |

External: razz97/Avalon-RDNA2 `k_fattn_qsa_rdna2` is `__launch_bounds__(256)` + `amdgpu_waves_per_eu(8)` (one-argument form). That is a 128-VGPR ceiling with max 16, so it gives no scheduler license.[^avalon]

Verdict: **no extras HIP change, no UNC change.** No dest kernel is silently dropped. The `(4, 8)` decode pins are working as designed. The `wvSplitKrc_` `(1, 1)` remark only bears on how the existing skinny occupancy card is worded for gfx1030; that is VLLM_FORK_Manager's call, not changed here.

## Sources

[^eff]: LLVM `AMDGPUSubtarget::getEffectiveWavesPerEU` / `getWavesPerEU` @ `39d7a5aae474940040c04556654e0079ca32569b`: default `{getWavesPerEUForWorkGroup(flat max), getOccupancyWithWorkGroupSizes(...).second}`; returns `Default` if requested min < default min, > default max, > requested max, or max > `getMaxWavesPerEU()`; else clamps max. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp#L190-L238
[^wg]: LLVM `getWavesPerEUForWorkGroup` / `getWavesPerWorkGroup` @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1187-L1196
[^simds]: LLVM `getNumWorkGroupSIMDs`: "All four SIMDs of the functional block in full-SIMD mode, half of them otherwise", returns 4 : 2 @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/include/llvm/TargetParser/AMDGPUTargetParser.h#L259-L263
[^fullsimd]: LLVM `isFullSIMDMode` = `isGFX1250(STI) || !FeatureCuMode` @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L2507-L2509
[^mfi]: LLVM `SIMachineFunctionInfo` constructor: `WavesPerEU = ST.getWavesPerEU(F)`, `Occupancy = ST.computeOccupancy(F, getLDSSize()).second` @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/SIMachineFunctionInfo.cpp#L58-L65
[^occ]: LLVM `GCNSubtarget::computeOccupancy`: `MaxOcc = min(work-group-size max, SGPROcc, VGPROcc)` @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/GCNSubtarget.cpp#L556-L566
[^sched-init]: LLVM `GCNSchedStrategy::initialize`: `TargetOccupancy = MFI.getOccupancy()` (unless relaxed-occupancy option), `VGPRCriticalLimit = min(getMaxNumVGPRs(TargetOccupancy), VGPRExcessLimit)`, comment "effectively sets a lower bound on the 'Critical' register limits" @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp#L80-L133
[^stages]: LLVM `GCNMaxOccupancySchedStrategy` / `GCNMaxILPSchedStrategy` / `GCNMaxMemoryClauseSchedStrategy` constructors (stage lists) @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp#L771-L879
[^select]: LLVM `AMDGPU::getSchedStrategy` (function attribute `"amdgpu-sched-strategy"` before option) and `GCNTargetMachine::createMachineScheduler` dispatch @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp#L441-L463 and https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp#L1224-L1256
[^opt]: LLVM `AMDGPUOptions.td`: `ValueField<"amdgpu-sched-strategy", "StringRef", "", "Select custom AMDGPU scheduling strategy">` @ `39d7a5aae474940040c04556654e0079ca32569b` (string not present in `AMDGPUUsage.md` at the same SHA, grep). https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/lib/Target/AMDGPU/AMDGPUOptions.td#L267-L268
[^usage-attr]: LLVM AMDGPUUsage, function attribute `"amdgpu-waves-per-eu"="m,n"`: "optimization hint … If the specified range is incompatible with the function's "amdgpu-flat-work-group-size" value, the occupancy bounds implied by the workgroup size take precedence." @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/llvm/docs/AMDGPUUsage.md#L3338-L3343
[^attrdocs]: Clang `AMDGPUWavesPerEUDocs`: `<max>` optional, "no restriction on the maximum … other than the one dictated by the hardware"; "An error will be given if … not compatible with values provided through other attributes" @ `39d7a5aae474940040c04556654e0079ca32569b`. https://github.com/llvm/llvm-project/blob/39d7a5aae474940040c04556654e0079ca32569b/clang/include/clang/Basic/AttrDocs.td#L3589-L3636
[^probe]: Compiler Explorer API probe 2026-11-13 (Europe/Paris), compilers `hipclang-rocm-60400`, `-70001`, `-70201`, `-71400`, `-100000`; flags `--offload-arch=gfx1030 --cuda-device-only -O3` (+ variants in §3). Kernel source in §3. https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
[^d-fa]: dest `fa_rdna2.cu` L304-L305 and L621-L622 (`__launch_bounds__(128|256)` + `amdgpu_waves_per_eu(4, 8)`) @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b`. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/fa_rdna2.cu#L304-L305
[^d-gdn]: dest `gdn_prefill_delta_h_rdna2.cu` L377-L378 (`__launch_bounds__(BV * 8)` + `(4, 8)`), L771-L784 (`GDN_PREFILL_PIPE` with BVN 64 / 32, block `BVN * 8`) @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b`. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/gdn_prefill_delta_h_rdna2.cu#L377-L378
[^d-skinny]: dest `skinny_gemms.cu` L1364 `#if defined(__gfx950__)`, L1368-L1369 `(1, 1)`, L1840-L1848 `#else` stub `{UNREACHABLE_CODE}` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b`. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu#L1364-L1369
[^avalon]: razz97/Avalon-RDNA2 `fattn-qsa-rdna2.cu` L14-L17 (`QSA_R2_NT 256`, `QSA_R2_WAVES_PER_EU 8`, "cap VGPRs at 128") and L49 @ `f792cc6a25ddb2ba76d29dd7bed3b6307da7f18f`. https://github.com/razz97/Avalon-RDNA2/blob/f792cc6a25ddb2ba76d29dd7bed3b6307da7f18f/ggml/src/ggml-cuda/fattn-qsa-rdna2.cu#L14-L49
