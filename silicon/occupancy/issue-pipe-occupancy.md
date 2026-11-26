---
title: "Independent issue pipes (VALU ∥ VMEM ∥ LDS ∥ SALU) vs effective occupancy (gfx1030)"
sources:
  - https://gpuopen.com/download/RDNA_Architecture_public.pdf
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://gpuopen.com/learn/occupancy-explained/
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://github.com/ROCm/HIP/blob/eab500faddcece1dcaf9b051d6928a453fbca75b/docs/understand/hardware_implementation.rst
  - https://github.com/llvm/llvm-project/issues/186179
---

# Independent issue pipes vs effective occupancy (gfx1030)

Lock: **issue-pipe independence (VALU ∥ VMEM ∥ LDS ∥ SALU) is not a PIX / `llvm-calc-occupancy` MaxWaves row.** It allocates nothing. What it affects is **how well sibling waves / same-wave ILP fill parked time inside a held VGPR∩LDS slot** — the latency-hiding story next to [waitcnt-occupancy.md](waitcnt-occupancy.md) (why a wave parks) and [spi-ace-occupancy.md](spi-ace-occupancy.md) (whether slots are filled at all). Completes the occupancy-adjacent set that [architecture.md](../arch-and-memory/architecture.md) §2.2 already states as issue facts but had no dedicated occupancy sibling for.[^llvm-calc-occupancy]

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **VALU pipe** | Per-lane vector ALU on a SIMD32: **1 instruction / cycle / SIMD32** (wave32). Dest latency **5 cycles** exposed; dependency stalls filled by other waves (RDNA Architecture deck). [^rdna-arch] |
| **SALU pipe** | Wave-uniform scalar ALU + control flow; drawn **separate** from each SIMD32 on the RDNA WGP diagram (deck). Dedicated occupancy sibling: [salu-occupancy.md](salu-occupancy.md) (open PR path). [^rdna-arch] |
| **VMEM pipe** | Vector memory (global / buffer / flat / scratch / texture). Outstanding ops tracked by `vmcnt` / `vscnt` ([waitcnt-occupancy.md](waitcnt-occupancy.md)). Independent of VALU issue so another wave can issue VALU while a load is in flight. |
| **LDS pipe** | Local Data Share (`ds_*`, some `lgkmcnt`-class ops). ISA WGP: “scalar & vector ALU’s and memory, as well as LDS and scalar caches.” [^isa70648] |
| **Issue-group / “5-issue”** | HIP Hardware implementation’s theoretical max of five co-issued ops (VALU + VMEM + SALU/SMEM + LDS + branch) — written against a **CDNA CU** diagram. **Not** an RDNA2 WGP quantified fact. [^hip-hw-impl][^hip-hw-rst] |
| **VOPD / `v_dual_*`** | RDNA3 dual-issue of selected VALU pairs in wave32. LLVM `FeatureVOPDInsts` is on **FeatureGFX11**, absent from FeatureGFX10 / FeatureGFX10_3. Dead on gfx1030 ([valu.md](../arch-and-memory/valu.md)). [^llvm-vopd] |

Date: **2026-11-06** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [architecture.md](../arch-and-memory/architecture.md) §2.2, [valu.md](../arch-and-memory/valu.md), [hip-craft.md](../hip-craft/hip-craft.md) §7.5, [salu-occupancy.md](salu-occupancy.md), [dot-issue-occupancy.md](dot-issue-occupancy.md), [trans-occupancy.md](trans-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [lds-dma-occupancy.md](lds-dma-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat multi-pipe issue as **effective latency hiding**, not MaxWaves: PIX / `llvm-calc-occupancy` book VGPR∩LDS slots; a wave with VMEM (or LDS) in flight still holds those slots while **another wave’s VALU** (or same-wave independent ILP) can issue on the SIMD. Deck: 1 VALU/cycle/SIMD32; “Dependency stalls can be filled by other waves.” [^llvm-calc-occupancy][^rdna-arch] |
| **Take** | Keep hot loops **mixable across pipes**: independent address / DOT / FMA next to outstanding loads; SALU/CF off the critical VALU beat when possible ([salu-occupancy.md](salu-occupancy.md)); wait with **non-zero** `vmcnt`/`lgkmcnt` thresholds so parks stay short ([waitcnt-occupancy.md](waitcnt-occupancy.md); [hip-craft.md](../hip-craft/hip-craft.md) §2). |
| **Take** | When VGPR∩LDS look full but RGP / rocprof shows long memory-wait with little VALU busy (or VALU busy with starved VMEM progress), fix **schedule / pipe mix / wait placement**, not another `__launch_bounds__` rung. GPUOpen Occupancy explained: occupancy exists to hide memory latency by switching to another resident wave. [^gpuopen-occupancy] |
| **Leave** | Do **not** treat HIP/CDNA “issue arbiter … five instructions per cycle (VALU + VMEM + SALU/SMEM + LDS + branch)” as an RDNA2 WGP fact. Exact RDNA2 issue-group width beyond “VALU every cycle, SALU independent, VMEM independent” is **not quantified in the ISA**. [^hip-hw-impl][^hip-hw-rst][^rdna-arch] |
| **Leave** | Do **not** expect RDNA3 VOPD / `v_dual_*` / `FeatureVOPDInsts` on gfx1030. Wave64 “dual-issue” on the RDNA deck is **half-wave replay**, not VOPD. [^llvm-vopd][^rdna-arch] |
| **Leave** | Do **not** trade VGPRs down “to create ILP” without measuring. Occupancy floors (VGPR∩LDS∩WG) still dominate reservation; ILP only fills **already reserved** parked time. Cite no invented cycle tables from this page. |
| **Leave** | Do **not** put issue-pipe width, “5-issue %”, or VOPD into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. [^llvm-calc-occupancy] |
| **Take Later (measurement only, Must/unverified on this box)** | On a V620 probe: `llvm-mca` / rocprof **SQ** counters for VALU vs VMEM (and LDS) busy under a controlled mix kernel. **Not probed this pass** — box has no `clang` / `llvm-mca` / `rocprof` in PATH. Do not invent numbers. |

## 1. What the deck / ISA say

### 1.1 RDNA Architecture deck (issue + latency hiding)

GPUOpen RDNA Architecture presentation:[^rdna-arch]

- “Each SIMD32 issues **1 instruction every cycle**. Vector instruction throughput is **1 every cycle** (for Wave32).”
- “**5 cycles** of latency are exposed (automatic dependency check in hardware). Dependency stalls can be filled by **other waves**.”
- WGP diagrams draw **SALU** beside each SIMD32 pair; LDS sits as a shared WGP resource — separate from the VALU issue beat.
- Transcendental co-issue with non-trans VALU is a **different** story ([trans-occupancy.md](trans-occupancy.md)); packed DOT is ordinary Write32Bit VALU ([dot-issue-occupancy.md](dot-issue-occupancy.md); [valu.md](../arch-and-memory/valu.md)).

Source: https://gpuopen.com/download/RDNA_Architecture_public.pdf (local `/workspace/rdna2-src/rdna-arch.pdf` / `rdna-arch.txt`).

### 1.2 ISA 70648 — WGP pipes + arbiter interleave

RDNA 2 ISA (Document **70648**):[^isa70648]

- **WGP**: “scalar & vector ALU’s and memory, as well as LDS and scalar caches.”
- **CU**: “2 SIMD32’s which share one path to memory.”
- **Instruction**: “vector ALU, scalar ALU, memory transfer, and control flow operations.”
- §4.1.1 Instruction Clauses: normally HW **interleaves** instructions from different waves; a hard clause forces service of **only one wave for a given instruction type** for the clause duration ([hard-clause-occupancy.md](hard-clause-occupancy.md)).

That interleave model is exactly why multi-pipe + multi-wave occupancy hides VMEM/LDS latency: when wave A is waiting on `vmcnt`, wave B can still own the VALU issue slot on that SIMD.

Local text: `/workspace/rdna2-src/rdna2-isa.txt`.

### 1.3 GPUOpen Occupancy explained (why slots exist)

Occupancy = assigned waves / max slots (16 per SIMD on RDNA2). The article’s latency-hiding diagram: wave A hits a memory fetch; the SIMD switches to another resident wave for ALU work while A’s request is outstanding. Reservation limiters stay VGPR / LDS / threadgroup / barriers — **not** “issue pipes.” [^gpuopen-occupancy]

https://gpuopen.com/learn/occupancy-explained/

## 2. What HIP / LLVM say (and what they do not)

| Piece | Role on gfx1030 |
|---|---|
| HIP Hardware implementation — “Instruction issuing” | Theoretical max **five** ops/cycle to different units (VALU, VMEM, SALU/SMEM, LDS, branch). CDNA-framed; **Leave** as RDNA2 WGP craft ([hip-craft.md](../hip-craft/hip-craft.md) §7.5). [^hip-hw-impl][^hip-hw-rst] |
| `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` | `min(WG+LDS, SGPR, VGPR)` only — **no** issue-pipe term. [^llvm-calc-occupancy] |
| `FeatureVOPDInsts` | On **FeatureGFX11** generation list; **absent** from FeatureGFX10. gfx1030 = GFX10.3 class → no `v_dual_*`. [^llvm-vopd] |
| Wave64 “dual-issue” (deck) | Half-wave replay of the same VALU/VMEM — **not** VOPD ([architecture.md](../arch-and-memory/architecture.md) §2.2). |

## 3. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed [^llvm-calc-occupancy] |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Waitcnt / caches / SPI / EXEC / TRANS / SALU / clauses | **No** | Park / thrash / refill / lane-idle / SFU-starve / scalar-pipe / arbiter lock (siblings) |
| **Independent issue pipes** | **No** | Slots stay reserved; **effective** fill of parked cycles depends on whether another wave (or ILP) can issue on a free pipe while this wave waits |

Mental model (no invented cycles):

```text
high reserved occupancy (VGPR∩LDS look full)
    + wave A: global_load outstanding (VMEM pipe busy for A)
    + wave B: ready VALU / DOT on same SIMD
    → B issues VALU while A’s load flies
    → measured VALUBusy can stay high while PIX occupancy looks fine

hard S_CLAUSE on wave A (same instruction type)
    → arbiter briefly locks that pipe onto A
    → B’s same-type issue waits — multi-pipe interleave fights the clause briefly
    → see hard-clause-occupancy.md
```

That is **not** “raise MaxWaves.” Fix mix/schedule/waits; do not chase another occupancy rung to “buy ILP.”

## 4. Contrast: TRANS co-issue and packed DOT

| Mechanism | Pipe story | Occupancy sibling |
|---|---|---|
| **TRANS ∥ non-trans VALU** | Same SIMD: ¼-rate 8-wide SFU can run **beside** ordinary VALU (deck). Still **not** a MaxWaves row. | [trans-occupancy.md](trans-occupancy.md) |
| **Packed DOT (`fdot2` / `sdot4`)** | Ordinary **Write32Bit VALU** issue (1/SIMD inferred; GPUOpen ops/clk); dependency chains bind the VALU pipe — not a separate “DOT pipe.” | [dot-issue-occupancy.md](dot-issue-occupancy.md); [valu.md](../arch-and-memory/valu.md) |
| **VALU ∥ VMEM ∥ LDS ∥ SALU** | Cross-pipe / cross-wave fill of waitcnt parks (this page). | — |
| **VOPD `v_dual_*`** | Two selected VALU ops / cycle in wave32 — **gfx11+ only**. | Leave ([valu.md](../arch-and-memory/valu.md)) |

## 5. Hard clauses vs multi-pipe interleave

Hard `S_CLAUSE` bursts temporarily **lock the instruction arbiter onto one wave for one instruction type** so a same-type VMEM/SMEM/… run is uninterrupted ([hard-clause-occupancy.md](hard-clause-occupancy.md); ISA §4.1.1). That is a deliberate, short fight with the interleave that this page relies on: slots stay booked; **same-type** issue from sibling waves pauses for the clause length. Craft = keep clauses where burst/coherency wins; do not sprinkle waits mid-burst you wanted claused; do not confuse clause length with a PIX limiter.

## 6. Craft checklist (HIP → ISA)

1. Size occupancy with `llvm-calc-occupancy` + runtime query first — pipes do not appear there. [^llvm-calc-occupancy]
2. Software-pipeline loads: issue several `global_load_*`, do independent VALU/DOT/address, then `s_waitcnt vmcnt(N)` with **N > 0** when older results are ready ([waitcnt-occupancy.md](waitcnt-occupancy.md)).
3. Keep SALU/CF and dense scalar post-SMEM work from serializing a VALU-heavy inner loop ([salu-occupancy.md](salu-occupancy.md)).
4. Do not invent a “5-issue” dependency graph from HIP CDNA docs ([hip-craft.md](../hip-craft/hip-craft.md) §7.5).
5. Do not emit or expect `v_dual_*` on `--offload-arch=gfx1030`.
6. If a hard VMEM clause is huge and sibling waves starve on that pipe, treat it as a **measured** schedule trade vs coherency — not a `__launch_bounds__` ticket.

## 7. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](../arch-and-memory/architecture.md) §2.2 | 1 VALU/cycle; SALU/VMEM/LDS separate; Leave CDNA 5-issue; Leave VOPD | Explicit **not a PIX row**; effective vs theoretical framing |
| [hip-craft.md](../hip-craft/hip-craft.md) §7.5 | Anti-CDNA-5-issue craft | Occupancy-adjacent: pipe mix fills parks inside held slots |
| [valu.md](../arch-and-memory/valu.md) | DOT issue; no VOPD | Pipes framing points here for occupancy |
| [trans-occupancy.md](trans-occupancy.md) | TRANS ∥ non-trans | Orthogonal: SFU vs VALU mix ≠ VMEM/LDS/SALU cross-pipe |
| [dot-issue-occupancy.md](dot-issue-occupancy.md) | DOT dependency chains on VALU | DOT stays on VALU pipe; not a fifth limiter |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Arbiter lock one type | Clauses briefly fight the interleave this page wants |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Why a wave parks | Pipes answer **who works** during that park |
| [salu-occupancy.md](salu-occupancy.md) | SALU contention | One pipe in the set; this page is the cross-pipe fold |
| [occupancy-composite.md](occupancy-composite.md) | Leave caches/clauses/waitcnt/… out of min | Issue-pipe independence as named *(out of min)* sibling |

## Sources

1. GPUOpen RDNA Architecture presentation — 1 VALU/cycle/SIMD32; 5-cycle dest latency; other waves fill dependency stalls; SALU drawn separate — https://gpuopen.com/download/RDNA_Architecture_public.pdf
2. RDNA 2 ISA Reference Guide, Document ID **70648** — WGP/CU/instruction terminology; §4.1.1 Instruction Clauses (interleave vs clause). Local: `/workspace/rdna2-src/rdna2-isa.txt`
3. GPUOpen Occupancy explained — latency hiding via resident waves; PIX limiters VGPR/LDS/threadgroup/barriers — https://gpuopen.com/learn/occupancy-explained/
4. LLVM `llvm-calc-occupancy` — no issue-pipe term — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
5. LLVM `AMDGPU.td` @ `eb61ef62df4fd5151c4c379302f636ac430bc32d` — `FeatureVOPDInsts` on FeatureGFX11, not FeatureGFX10
6. ROCm HIP Hardware implementation (docs + RST @ `eab500faddcece1dcaf9b051d6928a453fbca75b`) — CDNA “five instructions per cycle” issue-arbiter paragraph (Leave for RDNA2)
7. LLVM #186179 — VOPD `v_dual_dot2acc_*` is a gfx11 concern
8. Wiki companions listed above

Footnotes:

[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-11-06)
[^rdna-arch]: AMD, "RDNA Architecture" (GPUOpen, 2019). https://gpuopen.com/download/RDNA_Architecture_public.pdf (checked 2026-11-06)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-11-06)
[^gpuopen-occupancy]: GPUOpen, "Occupancy explained". https://gpuopen.com/learn/occupancy-explained/ (checked 2026-11-06)
[^hip-hw-impl]: ROCm HIP docs, "Hardware implementation". https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (checked 2026-11-06)
[^hip-hw-rst]: ROCm/HIP `docs/understand/hardware_implementation.rst` at `eab500faddcece1dcaf9b051d6928a453fbca75b` — "Instruction issuing" / five instructions per cycle list. https://github.com/ROCm/HIP/blob/eab500faddcece1dcaf9b051d6928a453fbca75b/docs/understand/hardware_implementation.rst (checked 2026-11-06)
[^llvm-vopd]: LLVM `AMDGPU.td` `FeatureVOPDInsts` / FeatureGFX11 includes it; FeatureGFX10 does not — snapshot `eb61ef62df4fd5151c4c379302f636ac430bc32d`. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td (checked 2026-11-06). Also LLVM #186179 (VOPD gfx11). https://github.com/llvm/llvm-project/issues/186179

## 8. Cross-links

| Page | Role vs this lock |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold page; issue pipes are *(out of min)* |
| [architecture.md](../arch-and-memory/architecture.md) §2.2 | Issue-model facts this page frames for occupancy |
| [valu.md](../arch-and-memory/valu.md) | Ordinary VALU / DOT; no VOPD on gfx1030 |
| [hip-craft.md](../hip-craft/hip-craft.md) §7.5 | Anti-CDNA-5-issue |
| [salu-occupancy.md](salu-occupancy.md) | SALU pipe contention sibling |
| [dot-issue-occupancy.md](dot-issue-occupancy.md) | DOT stays on VALU |
| [trans-occupancy.md](trans-occupancy.md) | TRANS ∥ non-trans (same SIMD, different unit) |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Temporary same-type arbiter lock vs interleave |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Park mechanism pipes fill around |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Slot fill / lack of work — not pipe mix inside a slot |
| [vgpr-occupancy.md](vgpr-occupancy.md) | Reservation floor; do not trade VGPRs blindly for ILP |
