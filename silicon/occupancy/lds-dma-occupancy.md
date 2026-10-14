---
title: "VMEM-to-LDS direct loads (buffer/global `lds` bit) vs effective occupancy (gfx1030)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://github.com/ROCm/aiter/pull/3336
  - https://github.com/ROCm/aiter/commit/a86fb0fab139ce78bce40e2ac51659a27ca161dd
  - https://github.com/opengfx1030/vllm-rdna/blob/8bb8ff36cdb2e885553a151a824706d2d920fe3a/csrc/rocm/skinny_gemms.cu
---

# VMEM-to-LDS direct loads (buffer/global `lds` bit) vs effective occupancy (gfx1030)

Lock: **gfx1030 can load from memory straight into LDS without a destination VGPR, but only one dword (or narrower) per lane per instruction.** RDNA2 MUBUF has an `LDS` bit ("Return read-data to LDS instead of VGPRs") for `BUFFER_LOAD_{ubyte, sbyte, ushort, sshort, dword, format_x}`, and GLOBAL/SCRATCH loads carry the same `LDS` bit.[^isa-mubuf-lds][^isa-flat-lds] LLVM puts `vmem-to-lds-load-insts` in `FeatureGFX10` (so gfx1030 has it) and not in `FeatureGFX11` or `FeatureGFX12`.[^llvm-td-feature][^llvm-td-gfx10] It is **not** a PIX / `llvm-calc-occupancy` MaxWaves row. What it changes is the **VGPR term**: a prefetched tile no longer occupies VGPRs while it is in flight. It does not shrink the LDS allocation, and it costs one SALU `M0` write per load plus four times the VMEM instruction count of a `dwordx4` copy.[^llvm-calc-occupancy][^probe-ldsdma]

Sibling of [vgpr-occupancy.md](vgpr-occupancy.md) (the term it moves), [lds-occupancy.md](../lds-and-tiles/lds-occupancy.md) (the term it does not move), [waitcnt-occupancy.md](waitcnt-occupancy.md), the SALU page from wiki PR #22 (the `M0` writes), and [lds-bank-occupancy.md](../lds-and-tiles/lds-bank-occupancy.md) (why the fixed lane layout matters).

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-10-14** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [vgpr-occupancy.md](vgpr-occupancy.md), [lds-tiles.md](../lds-and-tiles/lds-tiles.md), [skinny-gemm.md](../../kernels/gemm-moe/skinny-gemm.md), [hip-craft.md](../hip-craft/hip-craft.md).

## Take / Leave

| | |
|---|---|
| **Take** | Correct the old "hardware hole" note: VMEM-to-LDS is **present** on gfx1030. The AITER #3336 description says RDNA gfx10xx/gfx11xx/gfx12xx lack it; its test target was gfx1201, and that is right for gfx11/gfx12 only. gfx1030 assembles `buffer_load_dword … lds` and LLVM's GFX10 feature set includes `vmem-to-lds-load-insts`.[^aiter-3336][^llvm-td-gfx10][^probe-ldsdma] |
| **Take** (Later) | Treat it as a **VGPR-relief lever for double-buffered tile prefetch**, where loaded data would otherwise stay live across a compute phase. Probe §3: an 8-dword-per-lane prefetch loop drops from 36 to 27 VGPRs.[^probe-ldsdma] It matters only when that cut crosses a 16-VGPR allocation rung (for example 72 rounds to 80 for 12 waves/SIMD, while 64 allows 16; formula on [vgpr-occupancy.md](vgpr-occupancy.md)). In the probe both variants already sit at 16 waves, so occupancy did not change. Measure before adopting. |
| **Take** | Budget **one `s_mov`/`s_or` to `M0` per load** (LDS byte offset comes from `M0[15:0]`) and a VMEM read that is waited with `vmcnt`, then `s_barrier` before another wave reads the data. Probe: clang 22 emits a separate `M0` write before each of the 8 loads.[^isa-lds-addr][^isa-vmcnt][^probe-ldsdma] |
| **Take** | Fatbin rule: keep LDS-DMA code behind a gfx10 guard. gfx1100 (W7800 / 7900 XTX class) and gfx12 reject both the MUBUF `lds` form and the clang builtins (`needs target feature vmem-to-lds-load-insts`). gfx1013 (BC-250) assembles it, same as gfx1030.[^probe-ldsdma][^llvm-td-feature] |
| **Must** | Before any extras use, re-probe on the **ROCm 7.14.0** compiler. Upstream clang 19 rejects `__builtin_amdgcn_global_load_lds` on gfx1030 ("needs target feature gfx940-insts") and has no `__builtin_amdgcn_raw_ptr_buffer_load_lds`, while `llc` 19 still lowers both LLVM intrinsics on gfx1030. Upstream clang 22 accepts both builtins. The gate is the front end, not the silicon, and the pinned amd-clang was not checked here.[^probe-ldsdma] |
| **Leave** | Do **not** port the gfx950 16-byte path. dest `skinny_gemms.cu` calls `__builtin_amdgcn_global_load_lds(…, 16, …)` only under `#if defined(__gfx950__)`; on gfx1030 it takes the `bigType` VGPR copy, and `buffer_load_dwordx4 … lds` does not assemble for gfx1030.[^dest-skinny][^probe-ldsdma] Those copies are a prologue before `__syncthreads()`, not a prefetch overlapped with compute, so the VGPR lever above does not apply to them. |
| **Leave** | Do **not** expect a per-lane LDS layout. The LDS side adds the lane id to the `M0` offset (ISA: "LDS adds it"), so each instruction writes a lane-contiguous dword row. Padded or swizzled tiles from [lds-bank-occupancy.md](../lds-and-tiles/lds-bank-occupancy.md) can only be built at instruction granularity, not per lane.[^isa-lds-addr] |
| **Leave** | Do **not** subtract it from `--lds=`. The data still lands in the workgroup's LDS allocation, and the ISA clamps writes to the wave's own LDS space (out-of-range lanes are masked, with no memory violation).[^isa-lds-addr][^isa-memviol] |
| **Leave** | Do **not** invent latency or bandwidth numbers. Neither ISA 70648 nor the LLVM feature text gives a cycle cost for the `lds` path.[^isa-mubuf-lds][^llvm-td-feature] |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. |

## 1. What the ISA allows

| Path | Widths with `lds` | LDS address | Source |
|---|---|---|---|
| MUBUF `BUFFER_LOAD_*` | ubyte, sbyte, ushort, sshort, dword, format_x; TFE must be 0 | `M0[15:0]` byte offset + lane id, inside the wave's allocation (LDSbase in 32-dword units) | ISA §8.1.9 [^isa-lds-addr] |
| GLOBAL / SCRATCH | load only; must be 0 for FLAT | same `M0` scheme in the LLVM output | ISA §9.1 field table [^isa-flat-lds] |

The data is a VMEM read, so it is counted by `VM_CNT`, which is incremented for every vector-memory read and decremented in issue order.[^isa-vmcnt] ISA §10.2 describes the same split: data reaches LDS through VGPRs with DS instructions, or "for some types of loads" directly from memory.[^isa-dataflow]

## 2. Compiler side

| Check (upstream LLVM, not the ROCm pin) | gfx1030 | gfx1013 | gfx1100 | gfx1201 | gfx942 |
|---|---|---|---|---|---|
| `llvm-mc-22` `buffer_load_dword off, s[0:3], 0 lds` | ok | ok | error | error | ok |
| `llvm-mc-22` `buffer_load_dwordx4 … lds` | error | error | error | error | error |
| `llvm-mc-22` `global_load_lds_dword` | error | – | error | – | ok |
| clang 22 `__builtin_amdgcn_raw_ptr_buffer_load_lds` | `buffer_load_dword … offen lds` | same | feature error | – | `buffer_load_dword … lds` |
| clang 22 `__builtin_amdgcn_global_load_lds` (size 4) | `global_load_dword … lds` | same | feature error | – | `global_load_lds_dword` |
| clang 19 `__builtin_amdgcn_global_load_lds` | error: needs `gfx940-insts` | – | – | – | – |
| `llc-19` `llvm.amdgcn.{raw.buffer,global}.load.lds` | both lower | – | – | – | – |

All rows come from the box probe.[^probe-ldsdma] The LLVM source matches: `vmem-to-lds-load-insts` is defined as "global_load w/lds bit set, buffer_load w/lds bit set or global_load_lds" and is listed in `FeatureGFX9` and `FeatureGFX10` but not `FeatureGFX11` / `FeatureGFX12`.[^llvm-td-feature][^llvm-td-gfx10]

## 3. Probe (2026-10-14, clang 22.1.8, `-O3 -nogpulib`, gfx1030)

`db.c`: 256-thread workgroup, two 8 KiB LDS buffers (16 KiB group segment), each iteration prefetches 8 dwords per lane of the next tile while running 32 `fdot2` from the current one, then waits and barriers.[^probe-ldsdma]

| Variant | `.vgpr_count` | Rounded (granule 16) | VMEM loads/iter | `M0` writes/iter | DS writes/iter |
|---|---|---|---|---|---|
| VGPR prefetch + `ds_write` | 36 | 48 | 8 | 0 | 4 |
| `global_load_dword … lds` | 27 | 32 | 8 | 8 | 0 |

Both variants land at the 16-wave cap, so the 9-VGPR saving bought nothing in this toy. It becomes occupancy only for a kernel already sitting just over a rung. The probe does not measure time.

## 4. Where it sits in the occupancy composite

Out of the theoretical min. It is an input that lowers `vgpr_count`, not a separate limiter. Count the extra `M0` SALU writes and the barrier wait inside the held slot, like [waitcnt-occupancy.md](waitcnt-occupancy.md). Leave LDS bytes unchanged.

## Sources

1. AMD "RDNA 2" ISA, doc 70648 (local text `/workspace/rdna2-src/rdna2-isa-amd.txt`): §8.1 MUBUF field table (p.58), §8.1.9 "Memory Buffer Load to LDS" (p.65), §9.1 FLAT/GLOBAL/SCRATCH field table (p.83), §4 `VM_CNT` (p.29), §3.11 memory violations (p.22), §10.2 data flow (p.91).
2. LLVM `AMDGPU.td` at eb61ef62df4fd5151c4c379302f636ac430bc32d: `VMemToLDSLoad` definition and generation feature lists.
3. ROCm/aiter PR #3336, merged as a86fb0fab139ce78bce40e2ac51659a27ca161dd (gfx1201 fallback for `opus::_async_load`).
4. opengfx1030/vllm-rdna `rdna_extras` at 8bb8ff36cdb2e885553a151a824706d2d920fe3a, `csrc/rocm/skinny_gemms.cu`.
5. Box probe `/workspace/ldsdma-probe/` (`k.c`, `db.c`, `ir.ll`, `.s` outputs).

[^isa-mubuf-lds]: AMD "RDNA 2" ISA Reference Guide, doc 70648, 2020-11-30, §8.1 MUBUF fields (p.58): "LDS … 1 = Return read-data to LDS instead of VGPRs"; §8.1.9 (p.65) lists the supported loads and "illegal to set the instruction's TFE bit for loads to LDS". https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-14)
[^isa-flat-lds]: Same guide, §9.1 FLAT field table (p.83): "LDS … When set, data is moved from memory to LDS instead of to VGPRs. Available only for loads. For Global and Scratch only; must be zero for Flat." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-14)
[^isa-lds-addr]: Same guide, §8.1.9 (p.65): "LDS_offset = 16-bit unsigned byte offset from M0[15:0]"; "TIDinWave is only added if the resource (V#) has the ADD_TID_ENABLE field set to 1, whereas LDS adds it"; "the return data cannot be written outside the LDS space allocated to this wave"; "The LDSbase (alloc) is in units of 32 Dwords". https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-14)
[^isa-vmcnt]: Same guide, §4 `VM_CNT` (p.29): "Incremented every time a vector-memory read … (MIMG, MUBUF, MTBUF, or FLAT/Scratch/Global format) instruction is issued", decremented when that load and those before it complete. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-14)
[^isa-memviol]: Same guide, §3.11 (p.22): "Memory Buffer to LDS does NOT return a memory violation if the LDS address is out of range, but masks off EXEC bits of threads that would go out of range." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-14)
[^isa-dataflow]: Same guide, §10.2 (p.91): "When loading from memory, the data may be loaded into VGPRs first or for some types of loads it may be loaded directly into LDS from memory." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-14)
[^llvm-td-feature]: LLVM `AMDGPU.td` at eb61ef62df4fd5151c4c379302f636ac430bc32d, `defm VMemToLDSLoad` ("The platform has memory to lds instructions (global_load w/lds bit set, buffer_load w/lds bit set or global_load_lds…"). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L1631-L1634 (checked 2026-10-14)
[^llvm-td-gfx10]: Same file, `def FeatureGFX10` feature list includes `FeatureVMemToLDSLoad`; `FeatureGFX11` (L1839) and `FeatureGFX12` (L1874) lists do not. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L1804-L1838 (checked 2026-10-14)
[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide (options are `--wg-size`, `--vgprs`, `--sgprs`, `--lds`; no VMEM-to-LDS input). https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-10-14)
[^aiter-3336]: ROCm/aiter PR #3336 "[csrc/opus] add synchronous fallback to _async_load for non-buffer-to-LDS archs", merged as a86fb0fab139ce78bce40e2ac51659a27ca161dd; description states RDNA gfx10xx/gfx11xx/gfx12xx lack the feature and reports the failure on gfx1201. https://github.com/ROCm/aiter/pull/3336 ; https://github.com/ROCm/aiter/commit/a86fb0fab139ce78bce40e2ac51659a27ca161dd (checked 2026-10-14)
[^dest-skinny]: `csrc/rocm/skinny_gemms.cu`, opengfx1030/vllm-rdna `rdna_extras` at 8bb8ff36cdb2e885553a151a824706d2d920fe3a: `#if defined(__gfx950__)` → `__builtin_amdgcn_global_load_lds((int*)(&A[k]), (int*)(&s[k]), 16, 0, 0)`, `#else` → `*((bigType*)(&s[k])) = *((bigType*)(&A[k]))`, then `__syncthreads()`. https://github.com/opengfx1030/vllm-rdna/blob/8bb8ff36cdb2e885553a151a824706d2d920fe3a/csrc/rocm/skinny_gemms.cu#L403-L410 (same `__gfx950__` guard at L645, L905, L990, L2099, L2294; the L1549–L1754 calls sit inside the `wvSplitKrc_` block opened by `#if defined(__gfx950__)` at L1340, with an `#else` stub at L1816)
[^probe-ldsdma]: Box compiler probe `/workspace/ldsdma-probe/` (`k.c`, `db.c`, `ir.ll` and `.s` output; Debian clang/llvm 22.1.8 and 19.1.7, upstream, not the ROCm 7.14.0 pin). Local box path, not a public URL; re-run to reproduce.
