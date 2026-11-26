---
title: "Buffer V# range check (OOB_SELECT) as tail mask vs effective occupancy (gfx1030)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://gitlab.freedesktop.org/mesa/mesa/-/blob/9b0a38cabb83e9f82c2f831123064720200b1a36/src/amd/registers/gfx10-rsrc.json
  - https://gitlab.freedesktop.org/mesa/mesa/-/blob/9b0a38cabb83e9f82c2f831123064720200b1a36/src/amd/common/ac_descriptors.c
  - https://github.com/ROCm/composable_kernel/blob/8577ecdf9b00eb50c5363221f7c7af4a5a944608/include/ck/ck.hpp
  - https://github.com/llvm/llvm-project/blob/869eeb1e8a5a81164c9b7d722732937aa88e72b8/clang/include/clang/Basic/BuiltinsAMDGPU.td
  - https://godbolt.org/
---

# Buffer V# range check (OOB_SELECT) as tail mask vs effective occupancy (gfx1030)

Lock: **a buffer descriptor's hardware range check can replace the `if (i < n)` tail guard, so the tail wave keeps full EXEC and needs no `s_and_saveexec` / `s_cbranch_execz`.** On RDNA 2 an out-of-range buffer read returns zero and an out-of-range write is dropped; the check is chosen by the 2-bit `OOB_SELECT` field in the V#.[^isa-range] This is not a PIX MaxWaves row. It removes a divergent branch and the exec save/restore, and (with the address-mode effect already on the address-mode page in wiki PR #41) trims address VGPRs. It moves `.vgpr_count` only when that crosses an allocation rung on [vgpr-occupancy.md](vgpr-occupancy.md).[^probe-oob]

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [exec-divergence-occupancy.md](exec-divergence-occupancy.md) (the lane-idle cost this avoids), [vgpr-occupancy.md](vgpr-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md) (4 SGPRs per V#), [cache-policy.md](../arch-and-memory/cache-policy.md) §3.4 (V# bits 62–63), [infinity-cache-occupancy.md](infinity-cache-occupancy.md) (the `LLC_NOALLOC` field below), and the address-mode page in wiki PR #41 (`address-mode-occupancy.md`, `offen` + `soffset` VGPR savings).

## 1. What the ISA says (gfx1030 = RDNA 2)

RDNA 2 ISA §8.1, "Range Checking", Table 35:[^isa-range]

| `OOB_SELECT` | Out-of-bounds when | ISA "use" |
|---|---|---|
| 0 | `(index >= NumRecords) \| (offset >= stride)` | structured buffers |
| 1 | `index >= NumRecords` | raw buffers |
| 2 | `NumRecords == 0` | do not check bounds |
| 3 | if `swizzle_en & stride != 0`: `index >= NumRecords \|\| offset+payload > stride`; else `offset+payload > NumRecords` | raw; "num_records is reduced by sgpr_offset" |

Notes under the table:[^isa-notes]

- Out-of-range reads return zero (except components with `dst_sel = SEL_1`, which return 1). Out-of-range writes write nothing.
- `load/store_format_*` and atomics are checked all-or-nothing.
- `load/store_dword_x{2,3,4}` are range-checked **per component**.

`NumRecords` is in bytes when stride is 0 (or swizzle is off) and in units of stride otherwise; the field is 32 bits (V# bits 95:64).[^isa-v-table]

**Source conflict, flagged.** Mesa's `ac_descriptors.c` comment describes GFX10 mode 3 as `offset >= NUM_RECORDS` and says the `offset+payload > NUM_RECORDS` form is GFX11+.[^mesa-oob-comment] The RDNA 2 ISA text above already uses `offset+payload`.[^isa-range] We have no silicon run on gfx1030 to settle which applies to a straddling `dwordx4`. Until probed on the V620, keep tails dword-aligned and do not rely on a partially in-range x4 returning its in-range lanes. (unsourced for gfx1030 silicon behaviour)

## 2. V# word 3 on gfx103x (the 4th dword you pass to `make_buffer_rsrc`)

Mesa's register JSON has a separate `SQ_BUF_RSRC_WORD3_gfx103` layout:[^mesa-word3]

| Bits (word 3) | Field | Note |
|---|---|---|
| 0–11 | `DST_SEL_X/Y/Z/W` | 3 bits each |
| 12–18 | `FORMAT` | `GFX10_FORMAT` enum |
| 21–22 | `INDEX_STRIDE` | swizzled only |
| 23 | `ADD_TID_ENABLE` | |
| 24 | `RESOURCE_LEVEL` | "must be 1" |
| **26–27** | **`LLC_NOALLOC`** | gfx103 only in this file; not in plain GFX10 `SQ_BUF_RSRC_WORD3` |
| 28–29 | `OOB_SELECT` | enum: 0 STRUCTURED_WITH_OFFSET, 1 STRUCTURED, 2 DISABLED, 3 RAW[^mesa-oob-enum] |
| 30–31 | `TYPE` | "must be 0" |

The RDNA 2 ISA Table 37 lists `Resource Level` (bit 120), `OOB_SELECT` (125:124) and `Type` (127:126) but leaves 123:121 unnamed, so `LLC_NOALLOC` (V# bits 123:122) comes only from Mesa's gfx103 register file.[^isa-v-table][^mesa-word3] At the pinned Mesa SHA, `ac_descriptors.c` sets `OOB_SELECT` and `RESOURCE_LEVEL` for buffers but never references `LLC_NOALLOC`.[^mesa-oob-comment] Its effect on Infinity Cache allocation for gfx1030 compute is therefore (unsourced): treat it as a field to A/B on silicon, not a known MALL bypass.

**Composable Kernel's constant decodes cleanly.** CK uses `CK_BUFFER_RESOURCE_3RD_DWORD = 0x31014000` for `__gfx101__`/`__gfx103__` (0x31004000 for gfx11/gfx120, 0x00020000 for gfx9).[^ck-dword3] Against the gfx103 layout: `OOB_SELECT = 3` (RAW), `RESOURCE_LEVEL = 1`, `FORMAT = 20` (`GFX10_FORMAT_32_UINT` in Mesa's enum), `LLC_NOALLOC = 0`, all `DST_SEL = 0`.[^mesa-word3][^mesa-fmt] So CK's gfx103 buffer loads are raw, byte-bounded, range-checked loads. It is the right default for gfx1030.

## 3. Probe (Compiler Explorer, gfx1030, ROCm 7.14.0 and 10.0.0)

Source (no HIP headers; `-x hip --offload-arch=gfx1030 --offload-device-only -nogpulib -nogpuinc -O3 -g0 -S`, compilers `clang-rocm-71400` and `clang-rocm-100000`):[^probe-oob]

```cpp
#define __global__ __attribute__((global))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1,n)))
extern "C" __global__ void __launch_bounds__(256) guarded(const float* __restrict__ x, float* __restrict__ y, int n){
  int i = __builtin_amdgcn_workgroup_id_x()*256 + __builtin_amdgcn_workitem_id_x();
  if (i < n) y[i] = x[i]*2.0f;
}
extern "C" __global__ void __launch_bounds__(256) bufoob(const float* __restrict__ x, float* __restrict__ y, int n){
  int i = __builtin_amdgcn_workgroup_id_x()*256 + __builtin_amdgcn_workitem_id_x();
  __amdgpu_buffer_rsrc_t rx = __builtin_amdgcn_make_buffer_rsrc((void*)x, 0, n*4, 0x31014000);
  __amdgpu_buffer_rsrc_t ry = __builtin_amdgcn_make_buffer_rsrc((void*)y, 0, n*4, 0x31014000);
  float v = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rx, i*4, 0, 0));
  __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, v*2.0f), ry, i*4, 0, 0);
}
// bufoob_noalloc: same as bufoob, rx word3 = 0x31014000 | (1<<26)
```

| Kernel | Branch / EXEC | Address VALU | `next_free_vgpr` | `next_free_sgpr` |
|---|---|---|---|---|
| `guarded` | `v_cmp_gt_i32` + `s_and_saveexec_b32` + `s_cbranch_execz` | `v_ashrrev` + `v_lshlrev_b64` + 2×(`v_add_co`/`v_add_co_ci`) | 4 | 7 |
| `bufoob` | none | `v_lshlrev_b32` + `v_lshl_or_b32` | 2 | 9 |
| `bufoob_noalloc` | none | same | 2 | 9 |

The ISA body is byte-identical on 7.14.0 and 10.0.0.[^probe-oob] Observations from that listing:

- The V# is built in SALU: `s_and_b32 s5, s1, 0xffff` (base high 16 bits, stride 0), `s_mov_b32 s6, n*4` (num_records, from `s_lshl_b32`), `s_mov_b32 s7, 0x31014000`. Loads are `buffer_load_dword v1, v0, s[4:7], 0 offen`.
- The compiler passes word 3 through unchecked: `| (1<<26)` shows up as `s_mov_b32 s7, 0x35014000`. Nothing stops you from setting reserved or wrong bits, so a bad constant fails on silicon, not at compile time.
- In this toy kernel the VGPR drop (4 to 2) is far below any allocation rung, so occupancy is unchanged. The real gain is no divergent tail branch, plus fewer address VGPRs when many loads are live (see PR #41).

Builtin signatures (clang `BuiltinsAMDGPU.td`): `__builtin_amdgcn_make_buffer_rsrc(void *, short stride, uint64_t num_records, int flags)` and `__builtin_amdgcn_raw_buffer_load_b32(rsrc, int offset, int soffset, _Constant int aux)` returning `unsigned int`.[^llvm-builtins] Bit-cast the result for float data; an implicit conversion emits `v_cvt_f32_u32` (seen in a first draft of this probe).

## 4. Take / Leave

| Verdict | Item |
|---|---|
| **Take** (Later) | For K/V-cache, weight, and activation tails in our HIP kernels, use a raw V# (`OOB_SELECT = 3`, CK's `0x31014000`) with `num_records` = exact valid bytes. The tail tile needs no `if`, keeps the same instruction stream as full tiles, and its out-of-range lanes read zeros, which is the padding a GEMM/attention tail wants.[^isa-range][^ck-dword3] |
| **Take** (Later) | Clamp windows per tile by rebasing the V# (base + tile offset, `num_records` = remaining bytes). Offsets and `num_records` are 32-bit, so tensors over 4 GiB must be rebased per tile anyway.[^isa-v-table] |
| **Take** (Later, needs silicon) | A/B `LLC_NOALLOC` (word 3 bits 26–27) on streaming read-once weights to see whether it keeps the 128 MiB Infinity Cache for KV. Unknown effect; no source we hold documents it for compute. |
| **Leave** | Do not use `OOB_SELECT = 2` ("do not check bounds") to save the check; it removes the zero-fill the tail relies on.[^isa-range] |
| **Leave** | Do not copy gfx11/gfx12 word-3 constants (CK's 0x31004000 or 0) onto gfx1030; the field layout differs per family.[^ck-dword3] |
| **Leave** | Do not count on partial `dwordx4` tails until a V620 probe settles the ISA/Mesa conflict in §1. |

Occupancy fold: like address mode, this is an input to `vgpr_count` and to the EXEC/divergence term, not a separate limiter in [occupancy-composite.md](occupancy-composite.md). Each V# costs 4 SGPRs; SGPRs are not the gfx1030 limiter ([sgpr-occupancy.md](sgpr-occupancy.md)).

## Sources

[^isa-range]: AMD "RDNA 2" ISA Reference Guide (doc 70648), §8.1 "Range Checking", Table 35 "Buffer Out Of Bounds Selection" (local text `/workspace/rdna2-src/rdna2-isa-amd.txt` lines 3658–3698). https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-11-11)
[^isa-notes]: Same guide, notes 1–4 after Table 35 (local text lines 3703–3709). https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-11-11)
[^isa-v-table]: Same guide, Table 33 (V# addressing fields) and Table 37 "Buffer Resource Descriptor" (local text lines 3603–3628, 3827–3863). https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-11-11)
[^mesa-word3]: Mesa `src/amd/registers/gfx10-rsrc.json` at 9b0a38cabb83e9f82c2f831123064720200b1a36, `SQ_BUF_RSRC_WORD3_gfx103` (lines 343–357, `LLC_NOALLOC` line 353). https://gitlab.freedesktop.org/mesa/mesa/-/blob/9b0a38cabb83e9f82c2f831123064720200b1a36/src/amd/registers/gfx10-rsrc.json#L343-357 (checked 2026-11-11)
[^mesa-oob-enum]: Same file, `SQ_BUF_RSRC_WORD3__OOB_SELECT` enum (lines 187–192). https://gitlab.freedesktop.org/mesa/mesa/-/blob/9b0a38cabb83e9f82c2f831123064720200b1a36/src/amd/registers/gfx10-rsrc.json#L187-192 (checked 2026-11-11)
[^mesa-fmt]: Same file, `GFX10_FORMAT` enum: `GFX10_FORMAT_32_UINT` = 20. https://gitlab.freedesktop.org/mesa/mesa/-/blob/9b0a38cabb83e9f82c2f831123064720200b1a36/src/amd/registers/gfx10-rsrc.json (checked 2026-11-11)
[^mesa-oob-comment]: Mesa `src/amd/common/ac_descriptors.c` at 9b0a38cabb83e9f82c2f831123064720200b1a36, OOB_SELECT comment and word-3 build (lines 815–838); no `LLC_NOALLOC` reference in the file. https://gitlab.freedesktop.org/mesa/mesa/-/blob/9b0a38cabb83e9f82c2f831123064720200b1a36/src/amd/common/ac_descriptors.c#L815-838 (checked 2026-11-11)
[^ck-dword3]: Composable Kernel `include/ck/ck.hpp` at 8577ecdf9b00eb50c5363221f7c7af4a5a944608, `CK_BUFFER_RESOURCE_3RD_DWORD` (lines 86–96). https://github.com/ROCm/composable_kernel/blob/8577ecdf9b00eb50c5363221f7c7af4a5a944608/include/ck/ck.hpp#L86-L96 (checked 2026-11-11)
[^llvm-builtins]: LLVM `clang/include/clang/Basic/BuiltinsAMDGPU.td` at 869eeb1e8a5a81164c9b7d722732937aa88e72b8, lines 249 and 258. https://github.com/llvm/llvm-project/blob/869eeb1e8a5a81164c9b7d722732937aa88e72b8/clang/include/clang/Basic/BuiltinsAMDGPU.td#L249-L258 (checked 2026-11-11)
[^probe-oob]: Compiler Explorer API probe, 2026-11-11, compilers `clang-rocm-71400` (ROCm 7.14.0) and `clang-rocm-100000` (ROCm 10.0.0), source and flags as in §3; ISA and `.amdhsa_next_free_vgpr/sgpr` read from the returned assembly. https://godbolt.org/ (checked 2026-11-11)
