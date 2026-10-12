---
title: "Scalar ALU (SALU) issue vs effective occupancy (gfx1030)"
sources:
  - https://gpuopen.com/download/RDNA_Architecture_public.pdf
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://gpuopen.com/learn/occupancy-explained/
  - https://gpuopen.com/download/GPC24_Occupancy_explained.pdf
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
---

# Scalar ALU (SALU) issue vs effective occupancy (gfx1030)

Lock: **SALU issue / scalar-pipe contention is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** On gfx1030 the SALU is a **shared** scalar unit (RDNA Architecture deck draws **one SALU beside each SIMD32** → treat as **~1 SALU per CU / 2 per WGP**; ISA says the WGP has “scalar & vector ALU’s” without pinning 1 vs 2). Control flow, `EXEC` updates, scalar arithmetic, `s_mov` / compares, and waits after SMEM / `s_load` all consume that pipe while the wave still holds its VGPR∩LDS slot. Dense SALU or SALU-bound divergence handling lowers **effective** occupancy / utilization; sibling waves help only if they have **non-SALU** work the arbiter can issue.[^llvm-calc-occupancy][^rdna-arch][^isa70648]

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **SALU / Scalar ALU** | Operates on **one value per wavefront**; “manages all control flow” (ISA terminology + ch. 5). Encodes SOP1 / SOP2 / SOPC / SOPK / SOPP-class scalar ops (`s_add`, `s_and`, `s_cmp`, `s_cbranch_*`, `s_mov`, …). Cannot use VGPRs or LDS. [^isa70648] |
| **VALU / Vector ALU** | Per-lane arithmetic on VGPRs (`v_fma`, packed DOT, …). Separate pipe; RDNA deck: **1 VALU instruction / cycle / SIMD32**. [^rdna-arch] |
| **SMEM / K$ / scalar memory** | `s_load_*` into SGPRs through the **scalar data cache (K$ / SQC DCache, 16 KB/WGP)**. Misses raise `lgkmcnt` wait on the scalar side — companion [kcache-occupancy.md](kcache-occupancy.md), not a second SALU file. [^rdna-arch] |
| **CDNA “5-issue”** | HIP Hardware implementation paragraph: arbiter issues VALU + VMEM + SALU/SMEM + LDS + branch per cycle on a **CDNA CU** diagram. **Not** an RDNA 2 WGP fact ([hip-craft.md](../hip-craft/hip-craft.md) §7.5). [^hip-hw-impl] |

Rate / CU layout already live in [architecture.md](../arch-and-memory/architecture.md) §2.1–2.2. This page is the **occupancy framing** only.

Date: **2026-10-12** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [architecture.md](../arch-and-memory/architecture.md) §2.1–2.2 / SALU rows, [valu.md](../arch-and-memory/valu.md), [kcache-occupancy.md](kcache-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [trans-occupancy.md](trans-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [vgpr-bank-occupancy.md](vgpr-bank-occupancy.md), [hip-craft.md](../hip-craft/hip-craft.md) §7.5 / rocprof `SALUBusy`, [sgpr-occupancy.md](sgpr-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat SALU as **effective scalar throughput / issue contention**, not MaxWaves: PIX / `llvm-calc-occupancy` book VGPR∩LDS∩WG/barrier slots; a wave issuing dense `s_*` control / arithmetic still holds those slots while the **shared SALU** is the bottleneck (deck + ISA). [^llvm-calc-occupancy][^rdna-arch] |
| **Take** | When VGPR∩LDS look full but rocprof / RGP show high **`SALUBusy`** / Wave Issue Wait with low VALU busy, fix **scalar shape / schedule** (hoist loop counters, keep hot DOT/K paths SALU-light, prefer `v_cmpx` over long `s_and_saveexec` chains where the compiler already does on gfx1030 — [hazard-occupancy.md](hazard-occupancy.md)) **before** shaving another VGPR wave. [^gpuopen-occupancy] |
| **Take** | After SMEM / `s_load` / kernarg: correlate with [kcache-occupancy.md](kcache-occupancy.md) — a K$ miss parks on `lgkmcnt`; the wave then often burns SALU to unpack bases into SGPRs. Same slot, two out-of-min effects. |
| **Take** | Keep hot DOT / K loops **SALU-light**: hoist scalar address / stride / compare work **outside** the inner packed-DOT streak; do not sprinkle `s_mov` / `s_add` / branches through a clause-eligible VMEM/SMEM burst ([hard-clause-occupancy.md](hard-clause-occupancy.md) — SALU is illegal inside `S_CLAUSE`). |
| **Take** | Divergence handling is SALU-heavy by ISA design (all CF is scalar; `EXEC` changed by SALU / `v_cmpx`). Prefer uniform control and lane-masked VALU over deep scalar branch nests ([exec-divergence-occupancy.md](exec-divergence-occupancy.md)). [^isa70648] |
| **Leave** | Do **not** put SALU issue rate, SALU count, or “SALU occupancy %” into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. [^llvm-calc-occupancy] |
| **Leave** | Do **not** treat HIP’s CDNA “issue arbiter can issue five instructions per cycle (VALU + VMEM + SALU/SMEM + LDS + branch)” as an RDNA 2 WGP fact. Deck: 1 VALU/cycle/SIMD32 + SALU/VMEM/LDS as **separate** pipes; exact RDNA 2 issue-group width beyond that is **not quantified in the ISA**. [^hip-hw-impl][^rdna-arch] |
| **Leave** | Do **not** invent gfx1030 SALU cycles-per-op, scalar queue depth, or dual-issue tables. ISA ch. 5 lists formats and opcodes; the deck pins layout (“2x scalar execution resources” vs GCN) and draws one SALU per SIMD32 pair — stop there. [^isa70648][^rdna-arch] |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. |

## 1. What the deck / ISA say

### 1.1 RDNA Architecture deck (layout + 2× scalar)

GPUOpen RDNA Architecture presentation:[^rdna-arch]

- Highlight: **“2x scalar execution resources”** vs GCN (GCN drew one SALU per CU of 4×SIMD16; RDNA WGP draws **SALU + SIMD32** twice per dual-CU WGP).
- CU/WGP diagrams: each SIMD32 has a neighboring **SALU** block → wiki lock **~1 SALU per CU, two per WGP** (same wording as [architecture.md](../arch-and-memory/architecture.md) §2.1; ISA does not pin the integer).
- Issue model stated for VALU: “Each SIMD32 issues 1 instruction every cycle”; SALU is drawn as a **separate** block, not folded into the VALU rate row.
- Occupancy / PIX limiters stay VGPR / LDS / Thread Group Size / Barriers — SALU is drawn in the CU, never as a MaxWaves row (same pattern as TRANS in GPC24). [^gpuopen-occupancy]

Source: https://gpuopen.com/download/RDNA_Architecture_public.pdf (local `/workspace/rdna2-src/rdna-arch.pdf`).[^rdna-arch]

### 1.2 GPC24 CU diagram

GPUOpen GPC24 “Occupancy explained through the AMD RDNA architecture” labels **SALU / Scalar** beside VALU in the CU diagrams (same family of slides as the Occupancy explained article). PIX `WaveOccupancyLimiters` remain the four resource rows only.[^gpuopen-download-gpc24-occupancy-explained-pdf][^gpuopen-occupancy]

Sources: https://gpuopen.com/download/GPC24_Occupancy_explained.pdf ; https://gpuopen.com/learn/occupancy-explained/

### 1.3 ISA (Document 70648)

Local text `/workspace/rdna2-src/rdna2-isa.txt`:[^isa70648]

| Claim | Where |
|---|---|
| “Scalar ALU (SALU) — The scalar ALU operates on one value per wavefront and manages all control flow.” | §1.1 Terminology |
| WGP = “scalar & vector ALU’s and memory, as well as LDS and scalar caches.” | §1.1 |
| CU = “2 SIMD32’s which share one path to memory.” | §1.1 |
| “All kernel control flow is handled using scalar ALU instructions” (if/else, branches, looping). | ch. 2 |
| SALU + scalar memory work on an entire wavefront; operate on up to two SGPRs + literals. | ch. 2 |
| `EXEC` changed by Scalar ALU instructions; does **not** affect scalar ALU/memory or branches themselves. | ch. 2 / §3.3 |
| SALU ops = 32-bit integer arithmetic and 32-/64-bit bitwise; can operate on PC; many set SCC. **Cannot use VGPRs or LDS.** | ch. 5 |
| SMEM loads into SGPRs through the scalar cache; returns may be out-of-order → `lgkmcnt(0)` when all prior SMEM needed. | ch. 7 |
| `S_CLAUSE`: SALU (and branch / message / GDS / export) **illegal** inside a hard clause. | §4.1.1 / [hard-clause-occupancy.md](hard-clause-occupancy.md) |

### 1.4 HIP CDNA paragraph (warn only)

HIP Hardware implementation documents a CDNA CU “five instructions per cycle” arbiter including SALU/SMEM. Cite **only** to refuse importing that width onto gfx1030 craft ([hip-craft.md](../hip-craft/hip-craft.md) §7.5). [^hip-hw-impl]

## 2. What LLVM / HIP do

| Piece | Role |
|---|---|
| `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` | `min(WG+LDS, SGPR, VGPR)` only — **no** SALU term ([occupancy-composite.md](occupancy-composite.md)) [^llvm-calc-occupancy] |
| SGPR occupancy | GFX10+ always full SGPR file; SGPR is **not** a MaxWaves limiter ([sgpr-occupancy.md](sgpr-occupancy.md)). SALU **uses** those SGPRs; file size ≠ SALU issue. |
| `AMDGPUInsertDelayAlu` | Distinguishes VALU / SALU / TRANS delay kinds on newer gens — scheduling, not MaxWaves; do not invent gfx1030 delay tables from this page. |
| rocprof / RGP | `SALUBusy` (hip-craft §6.4 example PMC list); RGP instruction-timing: memory-wait bar green when other waves’ VALU ran, **yellow hatch = SALU** ([waitcnt-occupancy.md](waitcnt-occupancy.md)). [^gpuopen-occupancy] |
| gfx1030 branch idiom | Prefer `v_cmpx` (writes `EXEC` directly); gfx101x often emits `v_cmp` + `s_and_saveexec` (extra SALU) — [hazard-occupancy.md](hazard-occupancy.md). |

hipcc for `--offload-arch=gfx1030` owns emission. Craft = keep SALU out of the DOT/K inner streak; inspect `.s` for unexpected `s_cbranch` / `s_and_saveexec` / dense `s_mov` chains in the hot path. **No box probe this pass** (optional; prior occupancy pages that ship `.s` tables re-ran clang on a fixed probe tree — not repeated here).

## 3. Relation to occupancy — out of the PIX min

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)    // unchanged by SALU density
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect [^llvm-calc-occupancy] |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| **SALU / scalar issue** | **No** | Shared scalar pipe busy; wave holds VGPR∩LDS while issuing CF / scalar math / post-SMEM SGPR work |
| **K$ miss → `lgkmcnt`** | **No** | Often followed by SALU unpack ([kcache-occupancy.md](kcache-occupancy.md)) |
| **TRANS / SFU** | **No** | Separate ¼-rate vector pipe ([trans-occupancy.md](trans-occupancy.md)) — do not conflate with SALU |

Mental model:

```text
dense SALU on one wave  = holds slot; arbiter can still issue VALU/VMEM/LDS from siblings
all resident waves SALU-bound  = high theoretical occupancy, weak VALU util (SALUBusy high)
hot K loop with mid-burst s_mov/branch  = breaks S_CLAUSE adjacency + burns shared SALU
```

## 4. Craft checklist (HIP → ISA)

1. Size occupancy as before ([occupancy-composite.md](occupancy-composite.md)); do **not** invent a SALU term for `llvm-calc-occupancy`.
2. Profile: if theory full and measured full but VALU util low / `SALUBusy` high, open this page + [exec-divergence-occupancy.md](exec-divergence-occupancy.md) / [kcache-occupancy.md](kcache-occupancy.md) before another VGPR shave.
3. `--save-temps` and skim the kernel `.s` for dense `s_*` in the inner DOT/K region; hoist scalars; keep load streaks clause-clean (no SALU mid-burst).
4. Do not schedule as “5-issue CDNA”; schedule as “1 VALU/cycle/SIMD + independent SALU/VMEM/LDS pipes” ([hip-craft.md](../hip-craft/hip-craft.md) §7.5).

## Sources

1. GPUOpen RDNA Architecture PDF — https://gpuopen.com/download/RDNA_Architecture_public.pdf (local `/workspace/rdna2-src/rdna-arch.pdf`): WGP diagram SALU+SIMD32, “2x scalar execution resources,” 1 VALU/cycle/SIMD32.
2. AMD “RDNA 2” ISA Reference Guide, Document **70648** — terminology SALU; ch. 2 control flow / EXEC; ch. 5 SALU ops; ch. 7 SMEM; §4.1.1 clauses. Local `/workspace/rdna2-src/rdna2-isa.txt`. Landing: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
3. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/ (PIX four limiters; RGP timing colors).
4. GPUOpen GPC24 Occupancy explained PDF — https://gpuopen.com/download/GPC24_Occupancy_explained.pdf (CU diagrams label SALU/Scalar beside VALU).
5. `llvm-calc-occupancy` — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (no SALU term).
6. ROCm HIP Hardware implementation — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (CDNA 5-issue paragraph — **warn only**).
7. Wiki siblings: [architecture.md](../arch-and-memory/architecture.md) §2.1 SALU row; [kcache-occupancy.md](kcache-occupancy.md); [waitcnt-occupancy.md](waitcnt-occupancy.md); [hard-clause-occupancy.md](hard-clause-occupancy.md); [exec-divergence-occupancy.md](exec-divergence-occupancy.md); [trans-occupancy.md](trans-occupancy.md); [hip-craft.md](../hip-craft/hip-craft.md) §7.5.

Footnotes:

[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-10-12)
[^rdna-arch]: GPUOpen, "RDNA Architecture" public PDF. https://gpuopen.com/download/RDNA_Architecture_public.pdf (checked 2026-10-12)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-12)
[^gpuopen-occupancy]: GPUOpen, "Occupancy explained". https://gpuopen.com/learn/occupancy-explained/ (checked 2026-10-12)
[^gpuopen-download-gpc24-occupancy-explained-pdf]: GPUOpen GPC24, "Occupancy explained through the AMD RDNA architecture" PDF. https://gpuopen.com/download/GPC24_Occupancy_explained.pdf (checked 2026-10-12)
[^hip-hw-impl]: ROCm HIP docs, "Hardware implementation". https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (checked 2026-10-12)
