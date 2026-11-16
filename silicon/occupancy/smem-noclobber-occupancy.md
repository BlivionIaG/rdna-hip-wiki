---
title: "Uniform global loads: SMEM (noclobber / __restrict__) vs VMEM and effective occupancy (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/AMDGPUAnnotateUniformValues.cpp
  - https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/AMDGPUMemoryUtils.cpp
  - https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/SIISelLowering.cpp
  - https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
---

# Uniform global loads: SMEM (noclobber / `__restrict__`) vs VMEM and effective occupancy (gfx1030)

Lock: **whether a wave-uniform load from a global pointer (a per-block scale, a per-expert stride, a small table indexed by a loop counter) becomes `s_load_*` (SMEM → SGPR, K$) or `global_load_*` (VMEM → VGPR, `vmcnt`) is decided by an LLVM alias proof over the whole kernel, not by the load expression alone.** It is not a PIX / `llvm-calc-occupancy` MaxWaves row. What changes is VGPR count, how many vector loads can be in flight, and how often the wave parks on `s_waitcnt vmcnt(0)`.[^annotate][^isuniformload][^probe]

Does **not** change extras HIP or any UNC ticket: this is a craft and review rule for new kernels. No tok/s. Do not restate the FA pin.

Date: **2026-11-16** Europe/Paris.

Companions: [kcache-occupancy.md](kcache-occupancy.md) (K$, where SMEM lands), [waitcnt-occupancy.md](waitcnt-occupancy.md) (`vmcnt` vs `lgkmcnt` parks), [vmem-addressing-occupancy.md](vmem-addressing-occupancy.md) (SADDR form the VMEM fallback uses), [vgpr-occupancy.md](vgpr-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md) (SGPRs never bind on GFX10+).

## 1. The rule in LLVM (main @ `d0a9d2f1`)

1. **Uniform address.** `AMDGPUAnnotateUniformValues::visitLoadInst` returns early when the pointer is divergent.[^annotate]
2. **Kernel only.** The no-clobber walk runs only in entry functions (kernels). The pass comment says it cannot see past the function boundary.[^annotate]
3. **No clobber since kernel entry.** For a global (AS 1) load, `AMDGPU::isClobberedInFunction` walks MemorySSA from the load back to live-on-entry. Any `MemoryDef` that may alias is a clobber. If none is found the load gets `!amdgpu.noclobber`.[^annotate][^clobbered]
4. **Not clobbers:** fences, `s_barrier` / `__syncthreads`-class barrier intrinsics, `wave_barrier`, `sched_barrier` / `sched_group_barrier` / `iglp_opt`, and atomics (cmpxchg / RMW) whose pointer AA proves no-alias.[^reallyclobber]
5. **Selection.** `!amdgpu.noclobber` becomes the `MONoClobber` MMO flag. `SITargetLowering::isUniformLoad` accepts a load as scalar when it is uniform, aligned to `min(size, 4)`, and is invariant, or is in constant AS, or is a simple global load with `MONoClobber`. Vector loads also need `NumElements < 32` to stay scalar.[^mmoflags][^isuniformload][^vecload]

So a store through **any pointer that may alias** the table, on **any path** between kernel entry and the load, sends the load to VMEM. `__restrict__` on the kernel parameters is the usual way to give AA the no-alias proof. `__constant__` (AS 4) skips the alias check.

## 2. Probe (Compiler Explorer, 2026-11-16)

`--offload-arch=gfx1030 --cuda-device-only -O3`, all kernels `__launch_bounds__(256)`. Counts read from the emitted assembly; VGPR/SGPR from `.amdhsa_next_free_vgpr` / `_sgpr`.[^probe]

```cpp
// d (no restrict) / e (restrict): 16 uniform scales s[k], store y every step
for (int k = 0; k < 16; k++) { acc += x[i + k*n] * s[k]; y[i + k*n] = acc; }   // #pragma unroll
// c: same loop, restrict, single store after the loop
// f: s[] is a __constant__ float cs[16], no restrict on x/y
// a / b: two per-block scales s[blockIdx.x], s[blockIdx.x+gridDim.x], store y in between
// g: no restrict, LDS store + __syncthreads() between the two s[] loads, no global store before
// h: restrict, s[blockIdx.x] read inside a __device__ __attribute__((noinline)) helper
```

| Kernel | ROCm 7.14.0 (dest pin) | Scales via | `s_waitcnt vmcnt(0)` | VGPR |
|---|---|---|---|---|
| d: no restrict, store per step | 31 `global_load` (16 x + 15 s) | `global_load_dword v7, v5, s[0:1] offset:4k` (SADDR, VMEM) | 16 | 8 |
| e: restrict, store per step | 16 `global_load` | SMEM | 6, then `vmcnt(10)…(1)` ramp | 17 |
| c: restrict, store after loop | 16 `global_load` | SMEM | `vmcnt(15)…` ramp | 19 |
| f: `__constant__` table, no restrict | 16 `global_load` | `s_load_dwordx16 s[0:15]` | 16 | 6 |
| a: no restrict, store between | 3 `global_load` | 1st SMEM, 2nd **VMEM** | — | 6 |
| b: restrict, store between | 2 `global_load` | both SMEM | — | 6 |
| g: no restrict, barrier only | 1 `global_load` | both SMEM | — | 5 |
| h: restrict, noinline helper | helper does `flat_load_dword v0, v[0:1]` + `s_swappc` | **FLAT, VGPR address** | — | 8 (SGPR 33) |

ROCm 10.0.0 gave identical counts and VGPRs to 7.14.0. ROCm 6.4.0 gave the same load classes, but more VGPRs for the restrict kernels (c 27, e 48).[^probe]

Reading the table:

- **d vs e.** Without the proof, every `s[k]` after the first is a VMEM round trip, and each step serializes on `vmcnt(0)` because the next `x` load may also alias the `y` store. With `__restrict__`, scales come from SMEM and the `x` loads issue back to back.
- **f.** `__constant__` moves the scales to one `s_load_dwordx16`, but the `x` loads still serialize (16 × `vmcnt(0)`), because `x` vs `y` aliasing is a separate proof.
- **g.** A barrier between the loads does not cost the SMEM path, matching §1.4.
- **h.** A non-inlined helper loses both the uniform address space and the kernel-only walk: one generic `flat_load` with a VGPR pair, plus call overhead.
- **VGPR.** The restrict kernels use more VGPRs (8 → 17/19) because more `x` loads are in flight. At these sizes all variants stay at the 16-wave cap from [vgpr-occupancy.md](vgpr-occupancy.md) (`min(16, floor(1024 / alignTo(v, 16)))`). On a kernel that is already near a granule edge, the extra in-flight loads can cost a wave per SIMD. Check the dump ([occupancy-dump.md](occupancy-dump.md)).

## 3. Take / Leave

| | |
|---|---|
| **Take (Now, craft)** | New gfx1030 kernels: put `__restrict__` on every global pointer parameter whose buffers really do not overlap, so per-block or per-expert scales, zero-points, strides and small LUTs go to SMEM / K$ and `x`/`y` loads can be pipelined. |
| **Take (Now, review)** | In `-S` output, grep for `global_load_dword vN, vM, s[..]` with a constant offset, or `flat_load` inside a helper. That pattern usually means a uniform value went down VMEM because the alias proof failed or a helper was not inlined. Fix with `__restrict__`, `__forceinline__`, `__constant__`, or one explicit load into a local before the store loop. |
| **Take (Now, correctness)** | `__restrict__` is a promise. If the "scale" buffer does alias an output written in the same kernel, the SMEM path reads K$ and can return stale data. The compiler does not check this. |
| **Leave** | Not an occupancy-calculator input: keep `min(VGPR, LDS, WG/barrier)` in [occupancy-composite.md](occupancy-composite.md) unchanged. |
| **Leave** | No extras edit, no ticket. This pass did not audit extras kernels for the pattern. Whether any extras kernel hits it is (unverified). |

## Sources

[^annotate]: LLVM `AMDGPUAnnotateUniformValues.cpp` @ `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, `visitLoadInst` (divergent-pointer early return, entry-function-only no-clobber walk, `setNoClobberMetadata` for global loads). https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/AMDGPUAnnotateUniformValues.cpp#L67-L83
[^clobbered]: LLVM `AMDGPUMemoryUtils.cpp` @ `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, `isClobberedInFunction` (MemorySSA walk to live-on-entry through MemoryDefs and MemoryPhis). https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/AMDGPUMemoryUtils.cpp#L419-L470
[^reallyclobber]: LLVM `AMDGPUMemoryUtils.cpp` @ `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, `isReallyAClobber` (fences, barrier / sched intrinsics, and no-alias atomics are not clobbers). https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/AMDGPUMemoryUtils.cpp#L377-L417
[^mmoflags]: LLVM `SIISelLowering.cpp` @ `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, `getTargetMMOFlags` (`amdgpu.noclobber` → `MONoClobber`). https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/SIISelLowering.cpp#L21713-L21719
[^isuniformload]: LLVM `SIISelLowering.cpp` @ `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, `isUniformLoad` (uniform, alignment ≥ min(size, 4), invariant / constant AS / simple global with `MONoClobber`). https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/SIISelLowering.cpp#L2547-L2568
[^vecload]: LLVM `SIISelLowering.cpp` @ `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, vector load lowering keeps scalar loads when uniform, align ≥ 4 and `NumElements < 32`. https://github.com/llvm/llvm-project/blob/d0a9d2f12c154a5d32425dbc6c91ee799dc36737/llvm/lib/Target/AMDGPU/SIISelLowering.cpp#L13749-L13760
[^probe]: Compiler Explorer compile API probe run 2026-11-16 (Paris) against `hipclang-rocm-60400`, `hipclang-rocm-71400`, `hipclang-rocm-100000`; args `--offload-arch=gfx1030 --cuda-device-only -O3`; kernels a–h as in §2; counts of `s_load_dword*` / `global_load*` / `flat_load*` / `s_waitcnt` and `.amdhsa_next_free_vgpr` / `_sgpr` read from emitted assembly (kernels g, h on 7.14.0 only). Compiler Explorer builds are not pinned to a SHA; re-run to confirm. https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
