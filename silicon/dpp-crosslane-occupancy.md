# Cross-lane ops (DPP / permlane / ds_bpermute / wave.reduce) vs effective occupancy (gfx1030)

Lock: **Cross-lane data movement is not a PIX / `llvm-calc-occupancy` MaxWaves row.** DPP and `v_permlane*` are plain VALU (no LDS bytes, no waitcnt). `ds_bpermute_b32` / `ds_permute_b32` / `ds_swizzle_b32` run on the **LDS pipe** but allocate **no** LDS (`group_segment_fixed_size` stays 0), so they never lower `MaxWGsLDS`. What they cost is time inside a slot you already hold: each `__shfl_xor` step is an LDS-pipe op plus an `s_waitcnt lgkmcnt(0)` park, and `__builtin_amdgcn_wave_reduce_*` on gfx1030 is a **serial `v_readlane` loop**, not DPP. Sibling of [waitcnt-occupancy.md](waitcnt-occupancy.md) (lgkm parks), [exec-divergence-occupancy.md](exec-divergence-occupancy.md) (DPP/permlane want full EXEC) and [lds-bank-occupancy.md](lds-bank-occupancy.md) (bpermute shares the LDS pipe with real tile traffic).

**Closes an open item:** [hip-craft.md](hip-craft.md) §5.3 listed "DPP `0x118/0x114/0x112/0x111` wave32 reduce encodings vs ISA 70648 — not verified". Verified this pass: they are `row_shr:8/4/2/1`, legal on gfx1030, and the extras `REDUCE_SUM_DPP_WAVE32` (`csrc/rocm/skinny_gemms_int4.cu`, `rdna_extras` @ `c6d99ca3`, read 2026-10-08) is correct as written for `THRDS == 32` (result in lane 31). No extras change is needed.

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-10-08** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [lds-bank-occupancy.md](lds-bank-occupancy.md), [lds-occupancy.md](lds-occupancy.md), [atomic-occupancy.md](atomic-occupancy.md) (register pre-reduce before the CAS), [hip-craft.md](hip-craft.md) §5.2, [lds-tiles.md](lds-tiles.md) (row-max / row-sum), [../toolchain/compiler.md](../toolchain/compiler.md) (`__shfl_xor` `v_cndmask` fix, rocm-systems#11618), [../kernels/skinny-gemm.md](../kernels/skinny-gemm.md).

## Take / Leave

| | |
|---|---|
| **Take** | Full wave32 reduce, every lane gets the result: **`row_xmask:1,2,4,8` + `v_permlanex16_b32`**. 4 DPP-fused VALU + 1 permlane + 1 add, zero LDS ops, zero waitcnt. This is exactly what LLVM's own atomic optimizer (DPP strategy) emits on gfx1030 (probe §3). |
| **Take** | Reduce-to-one-lane (store from the last lane): `row_shr:8,4,2,1` (`0x118/0x114/0x112/0x111`) leaves each 16-lane row's sum in lanes 15 and 31; one cross-row step finishes it. That is the extras `REDUCE_SUM_DPP_WAVE32` shape — correct on gfx1030. |
| **Take** | For the cross-row step prefer `v_permlanex16_b32` (VALU) over `__shfl_xor(v, 16)` (`ds_bpermute` + `lgkmcnt(0)`). **Later** craft only: one LDS-pipe op per reduced value at the end of the K loop; A/B, not a fix. Fork call (VLLM_FORK_Manager), not this page. |
| **Take** | Butterflies written as `for (off = 16; off; off >>= 1) v += __shfl_xor(v, off)` cost **5 `ds_bpermute` + 5 `lgkmcnt(0)` parks** per value on gfx1030 (probe). Hot per-row loops (`rdna_fused_glue.cu` `wave_reduce<MT>`, skinny int4 MoE, sparse-MLA partials) are the places a DPP rewrite would bite. **Later**, A/B only. |
| **Take** | For max / min reductions pass the identity as `old` (`-inf` / `+inf`) and keep `bound_ctrl` off; `0` is only an identity for sum. Without `nnan` (`-ffinite-math-only` / fast-math) `fmaxf` adds a `v_max x,x` canonicalize per step and DPP does not fuse into the max (probe). |
| **Take** | No `s_nop` padding is needed on gfx1030 between a VALU write and a DPP read of that VGPR (ISA 70648 §4.5: "Inserting S_NOP is not required to achieve correct operation"; probe: gfx906 gets `s_nop 1`, gfx1030 none). Copied gfx9 inline asm with `v_nop; v_nop` before `v_*_dpp` is dead weight on gfx1030. |
| **Take** | Keep DPP / permlane under **full EXEC** (ISA §6.9: scan needs EXEC all 1s; seed unused lanes with the identity). ISA §6.10 also says `V_PERMLANE` may not immediately follow `V_CMPX` (insert any VALU). clang 22 does **not** pad this on gfx1030 (its `vcmpx-permlane-hazard` feature is gfx101x-only; forcing it inserts `v_mov_b32 v1, v1`). Doc and compiler disagree; not silicon-tested. If you hand-write permlane inside a branch, add a dummy VALU yourself. |
| **Leave** | Do **not** use `row_bcast:15/31` (`0x142/0x143`) or `wave_shl/shr/rol/ror` (`0x130–0x13C`) on gfx1030 — removed in GFX10. `llvm-mc` rejects them; clang hard-errors "Invalid dpp_ctrl value: broadcasts are not supported on GFX10+". A gfx9 reduce copy fails at compile time, not silently. |
| **Leave** | Do **not** trust `__builtin_amdgcn_wave_reduce_*` (`llvm.amdgcn.wave.reduce.*`) strategy hint 2 = DPP on gfx1030. clang 22.1.8 emits the same `s_ff1` / `v_readlane` / SALU-add loop for hints 0, 1 and 2 (up to 32 serial trips for a divergent value; `fadd_f32` adds a `v_add` + `v_readfirstlane` per trip). The ROCm 7.14 amd-llvm pin is not verified here. |
| **Leave** | Do **not** count `ds_bpermute` / `ds_swizzle` as LDS allocation in `--lds=` or `MaxWGsLDS` (ISA §10.4.4: "use the LDS hardware but do not use any memory storage, and may be used by waves which have not allocated any LDS space"). Effective occupancy only. |
| **Leave** | Do **not** invent DPP / permlane / bpermute latency or throughput numbers. ISA 70648 gives none per op. |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. |

## 1. What the ISA says (RDNA 2, Document 70648; local text `/workspace/rdna2-src/rdna2-isa.txt`)

- §6.9 DPP: two forms, **DPP8** (arbitrary swizzle inside 8-lane groups) and **DPP16** (fixed swizzles inside 16-lane rows). VOP1/VOP2 only (and VOPC for DPP8). Scan needs EXEC all 1s; "Readlane, readfirstlane and writelane cannot be used with DPP."
- Table 91 DPP_CTRL (~16670): `QUAD_PERM 000–0FF`, `ROW_SL 101–10F`, `ROW_SR 111–11F`, `ROW_RR 121–12F`, `ROW_MIRROR 140`, `ROW_HALF_MIRROR 141`. **No** `ROW_BCAST` / `WAVE_*`. The table is also silent on `ROW_SHARE 150–15F` / `ROW_XMASK 160–16F`; LLVM accepts both for GFX10+ and AMD's own compiler emits `row_xmask` on gfx1030 (§3), so treat them as legal with that caveat.
- §6.10: `V_PERMLANE` may not occur immediately after a `V_CMPX`; insert any VALU.
- §4.5: "Inserting S_NOP is not required to achieve correct operation."
- §10.4.4 LDS lane-permute ops: `ds_permute_b32` (scatter) / `ds_bpermute_b32` (gather) across 32 lanes; index in **bytes** (×4), only bits [6:2] used, wraps; EXEC honored; reading a disabled lane returns 0; no LDS storage used.

## 2. Cost model inside a held slot

| Op | Pipe | Waitcnt | LDS bytes | Notes |
|---|---|---|---|---|
| `v_*_dpp` (DPP16 / DPP8) | VALU | none | 0 | Usually fuses into the add/max (`v_add_f32_dpp`). Needs a temp VGPR when it cannot fuse. |
| `v_permlane16_b32` / `v_permlanex16_b32` | VALU (VOP3) | none | 0 | Selects are SGPR/literal; LLVM adds a `v_mov` copy of the source (+1 VGPR live). |
| `ds_bpermute_b32` / `ds_permute_b32` / `ds_swizzle_b32` | LDS pipe | `lgkmcnt` | 0 | Shares the WGP LDS pipe with tile `ds_read`/`ds_write`; every dependent step parks on `lgkmcnt(0)`. HIP `__shfl*` lowers here. |
| `v_readlane_b32` loop (`wave_reduce_*`) | VALU→SALU per lane | none | 0 | Serial, ≤ 32 trips for wave32; SALU-bound, the wave sits in its slot. |

## 3. Probe (box, upstream clang/LLVM 22.1.8 — not the ROCm 7.14 pin)

Files: `/workspace/dpp-probe/` (`p.hip`, `q.hip`, `m.hip`, `h.hip`, `ao.hip`, `*.s`). `clang-22 -x hip --cuda-device-only -nogpulib -nogpuinc --offload-arch=gfx1030 -O3 -S`; `llvm-mc-22 -mcpu=gfx1030 -show-encoding`.

| Case | gfx1030 | Contrast |
|---|---|---|
| `row_shr:1/8`, `row_ror:4`, `row_mirror`, `quad_perm`, `row_share:3`, `row_xmask:1`, `dpp8` | encode (`row_shr:8` byte `0x18`, `row_xmask:1` `0x61`) | gfx906: `row_share`/`row_xmask`/`dpp8` invalid |
| `row_bcast:15/31`, `wave_shl/shr/rol/ror` | `llvm-mc`: not a valid operand; clang: "broadcasts are not supported on GFX10+" | gfx906 encodes `0x142/0x143/0x130…` |
| `update_dpp` `0x111,0x112,0x114,0x118` after a VALU write | `v_add_f32_dpp … row_shr:N bound_ctrl:1`, **no `s_nop`** | gfx906: `s_nop 1` before each DPP |
| `__builtin_amdgcn_mov_dpp(float, …)` (extras macro form) | `bitcast float→i32` (no `v_cvt`), fuses to `v_add_f32_dpp` | — |
| extras `REDUCE_SUM_DPP_WAVE32` + `__shfl_xor(v,16)` (bpermute) | 4× `v_add_f32_dpp row_shr` + 1 `ds_bpermute` + `lgkmcnt(0)` | same with `permlanex16`: 0 LDS ops |
| 5-step `ds_bpermute` butterfly | 5 `ds_bpermute_b32`, 5 `s_waitcnt lgkmcnt(0)`, `group_segment_fixed_size: 0` | — |
| `row_xmask:1,2,4,8` + `permlanex16` sum | 4 fused `v_add_f32_dpp row_xmask` + `v_permlanex16_b32`, no waitcnt | — |
| same, fmax with `-inf` old | 4× `v_mov_b32_dpp` + `v_max` + canonicalize `v_max x,x`; with `-ffinite-math-only` the canonicalize goes | — |
| `__builtin_amdgcn_wave_reduce_add_u32(v, 0/1/2)`, `…_fadd_f32(v, 2)` | `s_ff1` / `v_readlane` / `s_add` (`v_add_f32` + `v_readfirstlane`) loop for **all** hints | — |
| atomic optimizer `-amdgpu-atomic-optimizer-strategy=DPP`, divergent i32 add | `v_add_nc_u32_dpp row_xmask:1,2,4,8` + `v_permlanex16_b32` + one `global_atomic_add` | — |
| `v_cmpx` → `v_permlanex16` (divergent branch) | no VALU inserted (SALU only between) | `+vcmpx-permlane-hazard`: `v_mov_b32 v1, v1` inserted; gfx1010 avoids `v_cmpx` entirely |

## 4. Extras read (no edit; `rdna_extras` @ `c6d99ca3`, 2026-10-08)

- `skinny_gemms_int4.cu` `REDUCE_SUM_DPP_WAVE32` under `__HIP__GFX1X__`: `row_shr:8,4,2,1` + `__shfl_xor(val,16)`, store from `threadIdx.x == THRDS-1` with launch `THRDS = 32` → **correct**. Cross-row step is a bpermute (Later craft above).
- `__shfl_xor` butterflies (5 steps) in `rdna_fused_glue.cu` `wave_reduce<MT>`, skinny int4 MoE/dense tails, `sparse_mla_rdna2.cu`, `layernorm.cu`; short 1–8 xor steps in the GDN kernels. All correct; all on the LDS pipe.
- `attention.cu` `v_nop; v_nop; v_*_dpp row_ror` inline asm sits under `__HIP__GFX9__` (MFMA paged attention) → **Leave**; not a gfx1030 path.
- No `wave_reduce_*`, `permlane*`, or `row_bcast` in the 52 rdna/rocm sources read.

## Sources

1. RDNA 2 ISA Reference Guide (AMD, Document 70648) — §4.5, §6.9, §6.10, Table 91 DPP_CTRL, §10.4.4, DS_BPERMUTE_B32 / V_PERMLANEX16_B32 opcode entries.
2. LLVM AMDGPU backend as shipped in clang/LLVM 22.1.8 (Debian) — DPP operand validation, `GCNHazardRecognizer` (`vcmpx-permlane-hazard`), wave.reduce lowering, `AMDGPUAtomicOptimizer` DPP strategy; observed via the probe, not source-read this pass.
3. LLVM AMDGPUUsage — https://llvm.org/docs/AMDGPUUsage.html — DPP / permlane intrinsics.
4. `opengfx1030/vllm-rdna` `rdna_extras` @ `c6d99ca3` — `csrc/rocm/skinny_gemms_int4.cu`, `rdna_fused_glue.cu`, `attention.cu` (read-only via GitHub API).
