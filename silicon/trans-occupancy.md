# Transcendental / SFU issue vs effective occupancy (gfx1030)

Lock: **Transcendental (SFU) issue — ¼ rate, 8-wide unit, co-issue with non-transcendental VALU — is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** Dense `v_exp` / `v_rcp` / `v_rsq` / `v_sqrt` / `v_log` / `v_sin` / `v_cos` sequences starve the special-function path while the wave still holds its VGPR∩LDS slot; sibling waves help only if they have **non-trans** work. Completes the “out of min” set next to [waitcnt-occupancy.md](waitcnt-occupancy.md) (time-idle park) and [exec-divergence-occupancy.md](exec-divergence-occupancy.md) (lane-idle NOP): here the slot is filled but **TRANS-throughput-starved**.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **Transcendental / TRANS / SFU** | Special-function vector ops on a dedicated **8-wide** unit per SIMD32: `rcp` / `rsq` / `sqrt` / `log` / `exp` / `sin` / `cos` (RDNA deck + GPC24 CU diagram). |
| **¼ rate** | Deck framing: transcendental instructions run at **quarter** the issue rate of ordinary VALU (like GCN). Wave32 completion on an 8-wide unit is the same story (~4 clocks / wave for a full-mask TRANS op) — do not invent finer cycle tables. |
| **Co-issue** | Non-transcendental VALU (`v_fma`, packed DOT, integer/address math) can run **in parallel** with a transcendental on the same SIMD (deck: “Non-transcendental instructions can execute in parallel”). |
| **VOPD** | RDNA3 dual-issue of *selected* VALU pairs (`v_dual_*`) — **absent** on gfx1030 ([valu.md](valu.md)). Not a substitute for TRANS co-issue. |

Rate / opcode lists already live in [architecture.md](architecture.md) §2.2 and [lds-tiles.md](lds-tiles.md). This page is the **occupancy framing** only.

Date: **2026-10-02** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [valu.md](valu.md), [architecture.md](architecture.md) §2.2, [hip-craft.md](hip-craft.md) §7.5 / `sched_barrier`, [lds-tiles.md](lds-tiles.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [fa-occupancy.md](fa-occupancy.md), [fa-gqa.md](fa-gqa.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat TRANS as **effective VALU throughput**, not MaxWaves: PIX / `llvm-calc-occupancy` book VGPR∩LDS slots; a wave issuing dense `v_exp` / `v_rcp` / `v_rsq` / … still holds those slots while the **8-wide / ¼-rate** path is the bottleneck (RDNA deck + GPC24). |
| **Take** | Prefer **co-issue packing**: interleave non-trans VALU (`v_fma` / `fdot2` / address math) with transcendentals so the parallel non-trans pipe stays fed (deck schedule pairs `v_rcp_f32` with `v_fma_f32`). |
| **Take** | Keep hot DOT / K loops **TRANS-light**: hoist softmax `exp` / `rcp` (and norms / `rsqrt`) **outside** the K inner loop ([lds-tiles.md](lds-tiles.md)). Prefer `v_cndmask` / scale math over `sin` / `cos` / `tan` in decode paths. |
| **Take** | When VGPR∩LDS look full but rocprof / RGP shows high **transcendental / `SQ_INSTS_VALU_TRANS`** with low non-trans VALU busy, fix **math shape / schedule**, not another MaxWaves rung. |
| **Take** | HIP: `__builtin_amdgcn_sched_barrier` / `sched_group_barrier` **TRANS** bit is real on gfx1030; the MFMA/WMMA sched bit is **Dead** here ([hip-craft.md](hip-craft.md)). |
| **Leave** | Do **not** put TRANS rate, SFU width, or “transcendental occupancy %” into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. |
| **Leave** | Do **not** invent gfx1030 cycles-per-`v_exp`, SFU queue depth, or tok/s from this page. Deck gives ¼ rate + co-issue; stop there. ISA ch. 12 lists opcodes and accuracy (1ULP, …), not an occupancy formula. |
| **Leave** | Do **not** open UNC / retip extras / restate the FA pin. Softmax register path is FA craft; this page only names TRANS as out-of-min. |
| **Leave** | Do **not** treat CDNA “5-issue” or RDNA3 VOPD as gfx1030 TRANS craft ([hip-craft.md](hip-craft.md) §7.5). |

## 1. What the deck / ISA say

### 1.1 RDNA Architecture deck (primary rate + co-issue)

GPUOpen RDNA Architecture presentation:

- Transcendental math co-execution: `rcp` / `rsq` / `sqrt` / `log` / `exp` / `sin` / `cos`
- “Transcendental instructions are **¼ rate** (like GCN).”
- “**Non-transcendental instructions can execute in parallel.**”

Same deck: 1 VALU instruction / cycle / SIMD32 for ordinary wave32 VALU; dependency stalls filled by other waves. TRANS does **not** appear as a reservation resource next to VGPR / LDS.

Source: https://gpuopen.com/download/RDNA_Architecture_public.pdf (local `/workspace/rdna2-src/rdna-arch.pdf`).

### 1.2 8-wide unit (GPC24)

GPUOpen GPC24 “Occupancy explained through the AMD RDNA architecture” CU diagrams label **Transcendental (8-wide)** beside the main VALU / SALU. That width matches the ¼-rate framing for wave32 (32 lanes / 8 = 4 beats). PIX `WaveOccupancyLimiters` in the sibling Occupancy explained article stay **VGPR / LDS / Thread Group Size / Barriers** only — TRANS is drawn in the CU, never as a MaxWaves row.

Sources: https://gpuopen.com/download/GPC24_Occupancy_explained.pdf ; https://gpuopen.com/learn/occupancy-explained/

### 1.3 ISA opcodes (Document 70648, ch. 12 / VOP1)

Representative compute-relevant ops (accuracy notes in ISA; not occupancy math):

| ISA | Role |
|---|---|
| `V_EXP_F32` | Base-2 exp |
| `V_LOG_F32` | Log |
| `V_RCP_F16` / `V_RCP_F32` / `V_RCP_F64` | Reciprocal |
| `V_RSQ_*` | Reciprocal sqrt |
| `V_SQRT_F32` | Sqrt |
| `V_SIN_*` / `V_COS_*` | Trig |

Local text: `/workspace/rdna2-src/rdna2-isa.txt`. HIP maps many of these through OCML / builtins; craft = keep them off the K-critical path.

### 1.4 Performance guide (secondary craft)

GPUOpen RDNA Performance Guide: minimize `sin` / `cos` / `sqrt` / `log` / `rcp` (quarter rate); RDNA can co-execute transcendentals; `tan` expands to **three** transcendentals (`sin`/`cos`); avoid arc (`atan` / `acos`) expansions that can cost 100+ cycles of codegen.

https://gpuopen.com/learn/rdna-performance-guide/

## 2. What LLVM / HIP do

| Piece | Role |
|---|---|
| `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` | `min(WG+LDS, SGPR, VGPR)` only — **no** TRANS / SFU term ([occupancy-composite.md](occupancy-composite.md)) |
| Sched classes | LLVM D81011 / `SISchedule.td`: separate transcendental sched classes (GFX10SpeedModel; “F16 or F32 transcendental instructions (these are quarter rate)”) — scheduling, not MaxWaves |
| `AMDGPUInsertDelayAlu` | Distinguishes VALU / SALU / TRANS delay kinds on newer gens; gfx1030 craft still “¼ rate + co-issue,” not invent delay tables |
| `__builtin_amdgcn_sched_barrier` / `sched_group_barrier` | Mask bits include **TRANS**; MFMA/WMMA bit Dead on gfx1030 ([hip-craft.md](hip-craft.md)) |
| rocprofiler-compute | `SQ_INSTS_VALU_TRANS` / “Instructions - Transcendental” separate from ordinary VALU mix |

hipcc for `--offload-arch=gfx1030` owns emission. Craft = hoist TRANS out of DOT/K loops; inspect `.s` for unexpected `v_exp` / `v_rcp` / OCML calls in the inner path.

## 3. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Waitcnt / caches / clauses / SPI / EXEC | **No** | Park / thrash / interleave / refill / lane-idle (siblings) |
| **TRANS / SFU** | **No** | Slots stay reserved; ¼-rate path saturated; non-trans co-issue idle if the wave is TRANS-only — weak **effective** VALU work while theory looks full |

Mental model (no invented cycles):

```text
high reserved occupancy (VGPR∩LDS look full)
    + dense softmax / rcp / rsqrt / trig in the hot loop
    → 8-wide TRANS pipe busy; non-trans VALU underfed
    → sibling waves hide little unless they have non-trans work
    → measured useful FLOP low while PIX occupancy looks fine
```

That is **not** “raise MaxWaves.” Fix math placement (hoist, approximate with FMA/scale, avoid `tan`/`atan` expansions), or accept the cost as algorithm — do not chase another occupancy rung.

## 4. Craft checklist (HIP → ISA)

1. Softmax / attention normalize: keep `exp` / `rcp` (and running-max updates that force them) **outside** the K DOT loop ([lds-tiles.md](lds-tiles.md)).
2. Norms / RMS: prefer one `rsqrt` (or reciprocal of reduced sum) per tile, not per-element TRANS in the multiply loop.
3. Decode: avoid `sin`/`cos`/`tan` for gating or rotary if a rotate/`v_cndmask`/precomputed table path exists; `tan` → three TRANS.
4. When a TRANS burst is unavoidable, leave **independent non-trans** work (address, DOT, integer) adjacent so co-issue can fire; do not serialize pure TRANS chains if FMA can sit beside them.
5. Profile with `SQ_INSTS_VALU_TRANS` vs ordinary VALU (and `VALUBusy`) before blaming VGPR pressure.
6. Occupancy still sized with `llvm-calc-occupancy` + `hipOccupancyMaxActiveBlocksPerMultiprocessor` — TRANS density does not appear in either.

## 5. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](architecture.md) §2.2 | ¼ rate + co-issue; opcode list; no VOPD | Explicit **not a PIX row**; effective vs theoretical framing |
| [hip-craft.md](hip-craft.md) | TRANS sched_barrier bit; no CDNA 5-issue | Occupancy-adjacent: TRANS-starved while VGPR∩LDS look full |
| [lds-tiles.md](lds-tiles.md) | Keep exp/rcp outside K loop | Elevates that craft note to out-of-min sibling |
| [valu.md](valu.md) | DOT / FMA issue table | TRANS is a **different** pipe class, not a DOT substitute |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Scoreboard park | Dual: waitcnt = **time-idle**; TRANS = **throughput-starved** on filled slot |
| [exec-divergence-occupancy.md](exec-divergence-occupancy.md) | Lane-idle NOP | Orthogonal: EXEC decides lanes; TRANS decides SFU vs VALU mix |
| [occupancy-composite.md](occupancy-composite.md) | Leave caches / clauses / waitcnt / EXEC / SPI out of min | TRANS as named *(out of min)* sibling |
| [fa-occupancy.md](fa-occupancy.md) / [fa-gqa.md](fa-gqa.md) | FA LDS/VGPR/`__launch_bounds__` | Softmax uses exp/rcp — craft only; do not restate FA pin |

## 6. Sources

1. GPUOpen RDNA Architecture presentation — transcendental ¼ rate + co-issue — https://gpuopen.com/download/RDNA_Architecture_public.pdf
2. GPUOpen GPC24 Occupancy explained (PDF) — Transcendental (8-wide) on CU diagram; not a reservation limiter — https://gpuopen.com/download/GPC24_Occupancy_explained.pdf
3. GPUOpen Occupancy explained — PIX limiters VGPR / LDS / threadgroup / barriers — https://gpuopen.com/learn/occupancy-explained/
4. GPUOpen RDNA Performance Guide — minimize TRANS; `tan`→3; avoid arc expansions — https://gpuopen.com/learn/rdna-performance-guide/
5. RDNA 2 ISA Reference Guide, Document ID **70648** — ch. 12 VOP1 opcodes (`V_EXP_F32`, `V_RCP_*`, …). Local: `/workspace/rdna2-src/rdna2-isa.txt`
6. LLVM `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` — no TRANS term — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
7. LLVM D81011 — transcendental sched classes (quarter rate) — https://reviews.llvm.org/D81011
8. Wiki companions: [occupancy-composite.md](occupancy-composite.md), [architecture.md](architecture.md), [hip-craft.md](hip-craft.md), [lds-tiles.md](lds-tiles.md)

## 7. Cross-links

| Page | Role vs this lock |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold page; TRANS is *(out of min)* |
| [valu.md](valu.md) | Ordinary VALU / DOT issue; no VOPD on gfx1030 |
| [architecture.md](architecture.md) | Issue model §2.2 — rate facts this page frames |
| [hip-craft.md](hip-craft.md) | `sched_barrier` TRANS bit; anti-CDNA-5-issue |
| [lds-tiles.md](lds-tiles.md) | Hoist exp/rcp outside K |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Time-idle park vs TRANS throughput starve |
| [exec-divergence-occupancy.md](exec-divergence-occupancy.md) | Lane-idle vs SFU-starve |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Clause is same-type burst; TRANS mix is a different pipe story |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Slot fill — not SFU mix inside a slot |
| [fa-occupancy.md](fa-occupancy.md) | Softmax TRANS class — craft only, no FA pin |
