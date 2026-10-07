# Atomics (global FP CAS loops / LDS atomics) vs effective occupancy (gfx1030)

Lock: **Atomics are not a PIX / `llvm-calc-occupancy` MaxWaves row.** A wave running an atomic still holds its VGPR∩LDS slot. What changes is how long it holds it: on gfx1030 a global/flat **fp32 `atomicAdd` is always a `global_atomic_cmpswap` retry loop** (returning, `glc`, `s_waitcnt vmcnt(0)` per trip), because the ISA has **no** global/flat FP add atomic. LDS fp32 add is native (`ds_add_f32`). Sibling of [waitcnt-occupancy.md](waitcnt-occupancy.md) (the loop is a chain of full L2 round-trip parks) and [exec-divergence-occupancy.md](exec-divergence-occupancy.md) (losing lanes retry under a shrinking EXEC).

**Correction this page lands:** extras `csrc/rocm/moe_accum_rdna2.cuh` (`rdna_extras`, read 2026-10-07) says the fp32 mode is "native `v_global_atomic_add_f32` … no CAS". On gfx1030 that instruction does not exist; `atomicAdd(float*, float)` there compiles to **four** independent CAS-32 loops per `moe_accum_row` (one per column), vs **one** CAS-64 loop for the packed-fp16 `atomic_add_pk4_f16` mode. The result is still a correct atomic add (fp32 accumulation and the one-cast epilogue are unchanged); only the "no CAS" rationale is wrong. Wiki pages that repeated "native" / "HW op" are fixed in the same push ([kernels/w4a8.md](../kernels/w4a8.md), [hip-craft.md](hip-craft.md) §3.3). Whether the comment or the mode choice changes is a fork call (VLLM_FORK_Manager), not this page.

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-10-07** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [l2-occupancy.md](l2-occupancy.md), [cache-policy.md](cache-policy.md) (atomics execute at L2), [hip-craft.md](hip-craft.md) §1 item 5 / §3.3, [lds-bank-occupancy.md](lds-bank-occupancy.md), [../kernels/w4a8.md](../kernels/w4a8.md) (MoE epilogue), [../kernels/w4a16.md](../kernels/w4a16.md) (packed fp16 CAS).

## Take / Leave

| | |
|---|---|
| **Take** | On gfx1030, read every global/flat **fp32 / fp16 / bf16 / f64 add** atomic as a **CAS loop**: `global_atomic_cmpswap … glc` → `s_waitcnt vmcnt(0)` → compare → `s_andn2 exec` → `s_cbranch_execnz`. Each trip is a full returning L2 round trip; the wave sits in its slot for all of them. |
| **Take** | Count CAS loops, not "atomics". `moe_accum_row<float>`: 4 loops per row (cols j=0..3). `atomic_add_pk4_f16`: 1 CAS-64 loop per row. Minimum trips per loop = max lanes in the wave hitting the **same address** (one winner per address per trip), plus cross-wave collisions. |
| **Take** | Pre-reduce in LDS with **native** `ds_add_f32` / `ds_add_rtn_f32` (ISA DS op 21 / 85) or in registers (DPP / `__shfl_xor`), then issue **one** global CAS per output element per WG. Same idea as the packed CAS-64: fewer global loops. **Later** craft; A/B only. |
| **Take** | fp32 **fmin / fmax** global atomics are native on gfx1030 (`GLOBAL_ATOMIC_FMIN/FMAX`, also `_X2` f64) — **only** when LLVM sees `!amdgpu.no.fine.grained.memory` (HIP `-munsafe-fp-atomics` / `unsafeAtomicMax`, or the clang atomic attribute). Without it: CAS loop too (probe below). |
| **Take** | Integer atomics (`GLOBAL_ATOMIC_ADD/SUB/…/INC/DEC/CSUB`) are native single instructions; a uniform-value add may also get the LLVM atomic optimizer (`v_mbcnt` + one lane issues). Non-returning ones count on **vscnt**, returning ones on **vmcnt** ([waitcnt-occupancy.md](waitcnt-occupancy.md)). |
| **Take** | Keep `-munsafe-fp-atomics` out of the "makes it native" story for gfx1030: it changes nothing for fp add here (probe: identical CAS loops with and without). It does matter on gfx11 (`global_atomic_add_f32` appears only with it) — contrast only. |
| **Leave** | Do **not** write "native `v_global_atomic_add_f32`" / "no CAS" for gfx1030. That is gfx11 / gfx90a / gfx94x / gfx12 ISA (`FeatureAtomicFaddRtnInsts` / `NoRtnInsts` are absent from `FeatureISAVersion10_3_0` in LLVM `AMDGPU.td`). |
| **Leave** | Do **not** put atomic count, CAS retries, or contention into `llvm-calc-occupancy` / PIX `WaveOccupancyLimiters`. Effective occupancy only. |
| **Leave** | Do **not** use `ds_pk_add_f16` / `global_atomic_pk_add_{f16,bf16}` on gfx1030 — absent (no `FeatureAtomicDsPkAdd16Insts` / `…PkAdd…`); LDS `<2 x half>` add is a `ds_cmpst_rtn_b32` loop (probe). |
| **Leave** | Do **not** invent L2 atomic throughput, per-channel atomic rate, or retry latency numbers. The ISA and LLVM give none. |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. Comment fix or mode choice → VLLM_FORK_Manager. |

## 1. What the ISA says (RDNA 2, Document 70648)

- §9.1 Flat / Global opcodes table (local text `/workspace/rdna2-src/rdna2-isa.txt` ~5080–5120): `SWAP, CMPSWAP, ADD, SUB, SMIN, UMIN, SMAX, UMAX, AND, OR, XOR, INC, DEC, FMIN, FMAX, FCMPSWAP`, plus global-only `CSUB`; all also `_X2`. `ADD` is integer. **There is no FP add** in the flat/global (or MUBUF) atomic list.
- DS opcodes: `DS_ADD_F32` (21) / `DS_ADD_RTN_F32` (85), "Floating point add that handles NaN/INF/denormal values" (~11722, ~12365). LDS fp32 add is a real instruction.
- Atomics execute at **L2** ([cache-policy.md](cache-policy.md)); `glc` on an atomic means "return the pre-op value", not "bypass".

## 2. What LLVM does (main, read 2026-10-07)

| Piece | gfx1030 |
|---|---|
| `FeatureISAVersion10_3_0` (`AMDGPU.td`) | `FeatureGFX10` + `FeatureAtomicFMinFMaxF32/F64Global/FlatInsts`. **No** `FeatureAtomicFaddRtnInsts`, `FeatureAtomicFaddNoRtnInsts`, `FeatureFlatAtomicFaddF32Inst`, `FeatureAtomicDsPkAdd16Insts`, `FeatureAtomicBufferGlobalPkAddF16Insts`. (`FeatureScalarAtomics` is gfx101x only.) |
| `SITargetLowering::shouldExpandAtomicRMWInIR`, `FAdd` | LOCAL f32 → `None` (native, `hasLDSFPAtomicAddF32()` = GFX8+). Global/flat f32: native only with `hasAtomicFadd{No,}RtnInsts` / `hasFlatAtomicFaddF32Inst` → else **`CmpXChg`**. |
| `FMin` / `FMax` | LOCAL f32/f64 native. Global/flat native only if `getGlobalMemoryFPAtomicLegality` ≠ Illegal, i.e. `!amdgpu.no.fine.grained.memory` present (gfx1030 has no agent-scope fine-grained remote FP atomics). |
| HIP `unsafeAtomicAdd(float*)` (`amd_hip_unsafe_atomics.h`, ROCm/clr develop) | HW builtin path is `#if defined(__gfx90a__)`; every other target falls to `__hip_atomic_fetch_add(…, AGENT)` → on gfx1030 the same CAS loop. |

## 3. Probe (box, upstream LLVM/clang 22.1.8 — not the ROCm 7.14 pin; the instruction absence is ISA-level, so the pin cannot do better)

Files: `/workspace/atomic-probe/` (`a.ll`, `b.ll`, `row.hip`, `*.s`). `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -O3`; `clang -x hip --offload-arch=gfx1030 -nogpuinc -nogpulib --cuda-device-only -O3 [-munsafe-fp-atomics]`.

| Case | gfx1030 | gfx1100 (contrast) |
|---|---|---|
| global f32 fadd, no-fine-grained md, result unused | `global_atomic_cmpswap glc` loop | `global_atomic_add_f32` |
| global f32 fadd, result used | CAS loop | `global_atomic_add_f32 … glc` |
| `row_f32` (= `moe_accum_row<float>` shape, 4 cols) | **4** CAS loops (`offset:0/4/8/12`), same with `-munsafe-fp-atomics` | 4× `global_atomic_add_f32` (with `-munsafe-fp-atomics`) |
| LDS f32 fadd | `ds_add_f32` | `ds_add_f32` |
| LDS `<2 x half>` fadd | `ds_cmpst_rtn_b32` loop | `ds_cmpstore_rtn_b32` loop |
| global f32 fmax, no-fine-grained md | `global_atomic_fmax` | `global_atomic_max_f32` |
| global f32 fmax, no md | CAS loop | CAS loop |
| global i32 add | `global_atomic_add` | `global_atomic_add_u32` |

Divergent-address fp32 add on gfx1030 (`b.ll`, 32 lanes → 8 addresses):

```text
.LBB0_1:
  s_waitcnt vmcnt(0)
  v_add_f32_e32 v2, v3, v4
  global_atomic_cmpswap v2, v[0:1], v[2:3], off glc
  s_waitcnt vmcnt(0)              ; full L2 round trip every trip
  v_cmp_eq_u32_e32 vcc_lo, v2, v3
  v_mov_b32_e32 v3, v2
  s_or_b32 s0, vcc_lo, s0
  s_andn2_b32 exec_lo, exec_lo, s0 ; winners retire, losers retry
  s_cbranch_execnz .LBB0_1
```

## 4. Relation to occupancy — out of the PIX min

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)    // unchanged by atomics
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| **Global FP atomic (CAS loop)** | **No** | Wave parks on `vmcnt(0)` once per trip; trips ≥ same-address lanes; EXEC shrinks as lanes win; slot held throughout |
| **LDS atomic (`ds_add_f32`)** | **No** | One `lgkmcnt` op; same-address lanes serialize inside the LDS, not in the shader |
| **Integer global atomic** | **No** | One VMEM op; vscnt (no-return) or vmcnt (return) |

Mental model:

```text
fp32 atomicAdd on gfx1030  = loop { load-old-from-reg ; add ; CAS@L2 ; wait ; retry losers }
4 columns per row          = 4 loops back to back (compiler does not fuse them)
packed fp16 CAS-64         = 1 loop per row (4 halves in one 64-bit CAS)
LDS pre-reduce + 1 CAS/elt = fewer global loops per WG; LDS fp32 add is native
```

## 5. Craft checklist (HIP → ISA)

1. `--save-temps` and grep the kernel `.s` for `global_atomic_cmpswap` + `s_cbranch_execnz`. If you expected `global_atomic_add_f32`, you are reading a gfx11 recipe.
2. Count loops per output element; prefer packing (CAS-64 over 4 halves) or an LDS/DPP pre-reduce when an epilogue does many fp32 atomics per lane.
3. For max/min reductions (e.g. scale / amax), `-munsafe-fp-atomics` or `unsafeAtomicMax` on **coarse-grained** `hipMalloc` buffers gets native `global_atomic_fmax`; never on fine-grained / host-visible memory ([hip-craft.md](hip-craft.md) §3.3).
4. Size occupancy as before ([occupancy-composite.md](occupancy-composite.md)); profile atomic-heavy epilogues as **effective** occupancy (waves resident but parked on `vmcnt`), the same signature as [waitcnt-occupancy.md](waitcnt-occupancy.md).

## Sources

1. AMD "RDNA 2" ISA Reference Guide (Document 70648), §9.1 Flat/Global opcode table; DS opcodes 21 / 85 — local `/workspace/rdna2-src/rdna2-isa.txt`.
2. LLVM `llvm/lib/Target/AMDGPU/AMDGPU.td` (main): `FeatureGFX10`, `FeatureISAVersion10_3_0`, gfx11 feature sets.
3. LLVM `llvm/lib/Target/AMDGPU/SIISelLowering.cpp` (main): `shouldExpandAtomicRMWInIR` FAdd / FMin / FMax cases, `getGlobalMemoryFPAtomicLegality`.
4. LLVM `llvm/lib/Target/AMDGPU/DSInstructions.td`, `GCNSubtarget.h` (`hasLDSFPAtomicAddF32` = GFX8 insts).
5. ROCm/clr `hipamd/include/hip/amd_detail/amd_hip_unsafe_atomics.h` (develop): `unsafeAtomicAdd(float*)` HW path gated on `__gfx90a__`.
6. Box probe `/workspace/atomic-probe/` (Debian LLVM/clang 22.1.8), 2026-10-07.
7. `opengfx1030/vllm-rdna` `rdna_extras` `csrc/rocm/moe_accum_rdna2.cuh` (read-only, 2026-10-07).
