---
title: "Warp id (threadIdx.x >> 5) is divergent in HIP: SGPR vs VGPR, SMEM vs VMEM, scalar vs EXEC loops (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/6eed2284f0482ef6be2de0a69fef3c0cfe878b2e/llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp
  - https://github.com/llvm/llvm-project/blob/6eed2284f0482ef6be2de0a69fef3c0cfe878b2e/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
  - https://github.com/llvm/llvm-project/commit/9470113495a79ee4f857b566c60a0adcd48c9d2a
  - https://github.com/llvm/llvm-project/blob/6eed2284f0482ef6be2de0a69fef3c0cfe878b2e/clang/include/clang/Basic/Attr.td
  - https://github.com/ROCm/llvm-project/blob/46fcb339fb61119b337f973c7ca9e710a319fdd0/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
  - https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
  - https://godbolt.org/api/compiler/llctrunk/compile
---

# Warp id (`threadIdx.x >> 5`) is divergent in HIP: SGPR vs VGPR, SMEM vs VMEM, scalar vs EXEC loops (gfx1030)

Lock: **in HIP on gfx1030, the warp id `threadIdx.x >> 5` (or `threadIdx.x / warpSize`, or `threadIdx.y` in a `(32, N)` block) is never proven wave-uniform by the compiler.** LLVM treats it as uniform only when the kernel carries `reqd_work_group_size` metadata, and HIP source cannot attach that attribute. A per-warp table load therefore stays VMEM (`global_load` + `vmcnt(0)`), and a warp-strided loop becomes an EXEC-mask loop with its counter in a VGPR. Wrapping the warp id in `__builtin_amdgcn_readfirstlane` turns both back into SMEM and SALU/SCC code. This is not a PIX / `llvm-calc-occupancy` MaxWaves row: it moves VGPR count, `vmcnt` parks and EXEC bookkeeping inside a held slot.[^tti][^xdim][^probe]

Does **not** change extras HIP or any UNC ticket: this is a craft and review rule for new kernels. extras was not audited for this idiom in this pass. No tok/s. Do not restate the FA pin.

Date: **2026-11-18** Europe/Paris.

Companions: [smem-noclobber-occupancy.md](smem-noclobber-occupancy.md) (the second condition for SMEM: alias proof), [exec-divergence-occupancy.md](exec-divergence-occupancy.md) (EXEC mechanics), [kcache-occupancy.md](kcache-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [salu-occupancy.md](salu-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md) (readlane loops).

## 1. The rule in LLVM (main @ `6eed2284`)

1. **Pattern.** `GCNTTIImpl::isAlwaysUniform` accepts `workitem.id.x >> C` with `C >= log2(wavesize)` (5 on wave32), and `workitem.id.x & Mask` when the mask has at least that many trailing zeros. Both need `XDimDoesntResetWithinWaves`.[^tti]
2. **Gate.** `XDimDoesntResetWithinWaves` comes from `AMDGPUSubtarget::hasWavefrontsEvenlySplittingXDim`. That function returns `false` unless the function has a 3-operand `reqd_work_group_size` node. With the node, it returns true for a 1-D size (Y and Z at most 1) or a power-of-two X.[^xdim] The source comment gives the reason: with uneven sizes such as `(65, 2)`, one wave holds X ids from two rows, so `tid.x / wavesize` differs across lanes.[^tti]
3. **History.** The gate came with LLVM #152581 (`9470113`, 2025-08-29). That change also marks the Y and Z ids uniform when the required X is a power of two at least the wave size.[^pr152581] The ROCm 7.14.0 compiler on Compiler Explorer is ROCm llvm-project `46fcb339` (from the emitted `clang version` string), and it carries the same function.[^rocmfork][^probe]
4. **HIP cannot set the gate.** Clang spells `reqd_work_group_size` as a GNU attribute with no `[[]]` spelling, marked "OpenCL-related".[^attrtd] In a HIP `__global__` function, ROCm 6.4.0, 7.14.0 and 10.0.0 all reject it with `attribute 'reqd_work_group_size' can only be applied to an OpenCL kernel function`.[^probe] `__launch_bounds__(256)` emits only `"amdgpu-flat-work-group-size"="1,256"`, and the device IR has no `reqd_work_group_size` node.[^probe]

## 2. Probe (Compiler Explorer, 2026-11-18)

HIP: `--offload-arch=gfx1030 --cuda-device-only -O3`, all kernels `__launch_bounds__(256)`, `__restrict__` on every pointer (so the alias proof from [smem-noclobber-occupancy.md](smem-noclobber-occupancy.md) is not what fails). IR rows: the 7.14.0 device IR (`-emit-llvm -S -g0`) fed to `llc` trunk (`-mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -O3`), unchanged, with only the flat size set to `256,256`, or with `!reqd_work_group_size !{i32 256, i32 1, i32 1}` added. VGPR/SGPR come from `.amdhsa_next_free_vgpr` / `_sgpr`.[^probe][^llc]

```cpp
// a: int w = threadIdx.x >> 5;  float v = s[w];  if (w == 0) v *= 2.f;  y[gid] = v;
// c: same, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5)
// d: (32,8) block shape, w = threadIdx.y
// h: w = threadIdx.x / warpSize
// f: for (int k = w; k < n; k += 8) acc += x[k*32 + (threadIdx.x & 31)];   w = threadIdx.x >> 5
// i: f with w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5)
```

| Kernel | Build | `s[w]` / loop control | `s_waitcnt vmcnt(0)` on scale | VGPR |
|---|---|---|---|---|
| a: `tid.x >> 5` | HIP 7.14.0 | `global_load_dword v3, v1, s[0:1]` (VMEM, SADDR) | 1 | 5 |
| c: readfirstlane | HIP 7.14.0 | `v_readfirstlane_b32` once, then `s_load_dword` (SMEM) | 0 (`lgkmcnt`) | 3 |
| d: `threadIdx.y` | HIP 7.14.0 | VMEM, same as a | 1 | 5 |
| h: `tid.x / warpSize` | HIP 7.14.0 | VMEM, same as a | 1 | 5 |
| a | IR → llc trunk, flat `256,256` | VMEM (no change) | 1 | 5 |
| a | IR → llc trunk + `reqd_work_group_size` | compiler adds `v_readfirstlane_b32 s4`, then `s_load_dword s0, s[0:1], s4` | 0 | 4 |
| f: warp-strided loop | HIP 7.14.0 | `v_cmpx_gt_i32` + `s_cbranch_execz`; loop exit by `s_andn2_b32 exec_lo` / `s_cbranch_execnz`; counter `v4` | — | 7 |
| i: readfirstlane loop | HIP 7.14.0 | `s_cmp_ge_i32` + `s_cbranch_scc1/scc0`; counter `s4` (`s_add_i32 s4, s4, 8`); no EXEC writes | — | 6 |
| f | IR → llc trunk + `reqd_work_group_size` | entry `s_cbranch_scc1`, back edge `v_cmp_gt_i32` → `s_cbranch_vccnz` (uniform branch, counter still `v4`) | — | 7 |

ROCm 10.0.0 gave the same instruction classes as 7.14.0. VGPRs matched except `f`, which used 6 on 10.0.0 and 7 on 7.14.0. ROCm 6.4.0 gave the same classes, with SGPR 10 for `c` and 9 for `f`.[^probe]

Reading the table:

- **a / d / h.** The table index is a VGPR, so the load goes to VMEM even with `__restrict__`, and the branch on `w == 0` becomes `v_cmp` + `v_cndmask`. That is the same result as a per-lane index.
- **c.** One `v_readfirstlane` moves the id to an SGPR. The load goes to SMEM (K$ path, `lgkmcnt`) and the kernel uses 2 fewer VGPRs.
- **Exact flat size does not help.** A `256,256` flat size with no `reqd_work_group_size` node still gives VMEM. Only the metadata node opens the gate (§1.2).
- **i vs f.** With readfirstlane, the warp loop is pure SALU control: no `exec` save, no `execz` skip, counter in an SGPR. Without it, every trip does `s_or_b32` / `s_andn2_b32 exec_lo` bookkeeping, and the counter costs a VGPR.
- **Metadata alone is weaker than readfirstlane for loops.** With `reqd_work_group_size`, LLVM trunk branches uniformly on VCC but keeps the counter and compare in VALU (VGPR 7). The explicit readfirstlane form `i` is the only one in this probe with an SGPR counter.

## 3. Take / Leave

| Call | Item |
|---|---|
| **Take — Now** (review rule) | In new gfx1030 HIP kernels, compute the warp id once as `__builtin_amdgcn_readfirstlane(threadIdx.x >> 5)` (or `/ 32`) and derive per-warp table indices, row bases and warp-strided loop bounds from it. This keeps per-warp scales on SMEM and loop control on SALU. |
| **Take — Now** (correctness gate) | readfirstlane returns the first active lane's value and does not check that the value is uniform. Use it only when each wave has a single warp id: 1-D blocks (`blockDim.y == blockDim.z == 1`), or `blockDim.x` a multiple of 32. For example, a `(48, 2)` block puts X ids 32–47 of row 0 and 0–15 of row 1 in the same wave, so the lanes need warp ids 1 and 0 and readfirstlane gives wrong answers with no error. This is the same reset case the LLVM comment describes.[^tti] Add a host-side `TORCH_CHECK` or `static_assert` on the block shape next to the readfirstlane. |
| **Take — Later** | With a 2-D `(32, N)` block, `threadIdx.y` is the warp id. It is also divergent in HIP (row d), so wrap it in readfirstlane under the same gate. |
| **Leave** | Expecting `__launch_bounds__`, an exact flat size, or `warpSize` to make the warp id uniform. None of them emits the `reqd_work_group_size` node, and the HIP spelling is a compile error. |
| **Leave** | Hand-written IR metadata or OpenCL just to get the gate. It is weaker than readfirstlane for loops (§2), and it is not a dest compile path. |

## 4. Cross-links

| Page | Role vs this lock |
|---|---|
| [smem-noclobber-occupancy.md](smem-noclobber-occupancy.md) | SMEM needs a uniform address **and** a no-clobber proof. This page covers the first condition. |
| [exec-divergence-occupancy.md](exec-divergence-occupancy.md) | What EXEC-mask loops cost inside a held slot. |
| [salu-occupancy.md](salu-occupancy.md) | SALU issue for scalar loop control. |
| [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md) | `readlane` / `readfirstlane` in reductions. |
| [hip-craft.md](../hip-craft/hip-craft.md) | Builtin table (`readfirstlane` to make a pointer or scale uniform). |

## Sources

[^tti]: LLVM `GCNTTIImpl::isAlwaysUniform`, `llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp` lines 1472–1518 @ `6eed2284f0482ef6be2de0a69fef3c0cfe878b2e`. https://github.com/llvm/llvm-project/blob/6eed2284f0482ef6be2de0a69fef3c0cfe878b2e/llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp#L1472-L1518
[^xdim]: LLVM `AMDGPUSubtarget::hasWavefrontsEvenlySplittingXDim`, `llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp` lines 250–266 @ `6eed2284f0482ef6be2de0a69fef3c0cfe878b2e`. https://github.com/llvm/llvm-project/blob/6eed2284f0482ef6be2de0a69fef3c0cfe878b2e/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp#L250-L266
[^pr152581]: LLVM commit `9470113495a79ee4f857b566c60a0adcd48c9d2a`, "[AMDGPU] Mark workitem IDs uniform in more cases (#152581)", committed 2025-08-29. https://github.com/llvm/llvm-project/commit/9470113495a79ee4f857b566c60a0adcd48c9d2a
[^attrtd]: Clang `def ReqdWorkGroupSize`, `clang/include/clang/Basic/Attr.td` @ `6eed2284f0482ef6be2de0a69fef3c0cfe878b2e` (GNU spelling only, "OpenCL-related attribute"). https://github.com/llvm/llvm-project/blob/6eed2284f0482ef6be2de0a69fef3c0cfe878b2e/clang/include/clang/Basic/Attr.td
[^rocmfork]: ROCm llvm-project `AMDGPUSubtarget.cpp` @ `46fcb339fb61119b337f973c7ca9e710a319fdd0` (contains `hasWavefrontsEvenlySplittingXDim`). https://github.com/ROCm/llvm-project/blob/46fcb339fb61119b337f973c7ca9e710a319fdd0/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp
[^probe]: Compiler Explorer API probe run 2026-11-18 (Europe/Paris), compilers `hipclang-rocm-60400`, `hipclang-rocm-71400`, `hipclang-rocm-100000`, flags `--offload-arch=gfx1030 --cuda-device-only -O3`. Instruction classes and register counts were read from the returned assembly, and the attribute error from the returned stderr. https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
[^llc]: Compiler Explorer `llctrunk` on 2026-11-18, `-mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -O3`, input = 7.14.0 device IR with the edits in §2. The verifier rejected `reqd_work_group_size` 256×1×1 with flat size `1,256` ("amdgpu-flat-work-group-size must equal the product of reqd_work_group_size operands"), so the reqd rows also set `256,256`. https://godbolt.org/api/compiler/llctrunk/compile
