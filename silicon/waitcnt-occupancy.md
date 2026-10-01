# Waitcnt / memory-dependence scoreboard vs effective occupancy (gfx1030)

Lock: **`s_waitcnt` / `s_waitcnt_vscnt` / memory-dependence scoreboards are not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** They are the **wave-local** mechanism that parks a wave until outstanding VMEM / LDS / SMEM / store counts drop — the same stalls occupancy exists to **hide**. Early or universal drains (`vmcnt(0)` “to be safe”) lengthen that park while VGPR∩LDS still looks full. Completes the occupancy-adjacent set next to [hard-clause-occupancy.md](hard-clause-occupancy.md) (arbiter burst) and the cache-thrash pages (what fills the counters).

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **Scoreboard / waitcnt** | Per-wave counters + `S_WAITCNT*` that resolve **memory** (and some export) dependences the HW does not auto-wait (ISA §4.4). |
| **VMCNT** | Outstanding **vector memory loads** / atomics-with-return (ISA: 6-bit counter). |
| **VSCNT** | Outstanding **vector memory stores** / atomics-without-return (ISA: 6-bit; separate from VMCNT on RDNA). |
| **LGKMCNT** | Outstanding **LDS, GDS, constant (SMEM), message** (ISA abbrev: **4-bit** counter). |
| **EXPCNT** | Export / GDS / some VMEM write-data (graphics; ignore in compute craft). |
| **DEPCTR** | Separate VALU/SALU dependency counters waited via `S_WAITCNT_DEPCTR` (debug / workaround; not a PIX row). |

Craft of *which* wait / when lives in [hip-craft.md](hip-craft.md) §2. This page is the **occupancy framing** only.

Date: **2026-09-30** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [hip-craft.md](hip-craft.md) §2, [hard-clause-occupancy.md](hard-clause-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [kcache-occupancy.md](kcache-occupancy.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [l2-occupancy.md](l2-occupancy.md), [infinity-cache-occupancy.md](infinity-cache-occupancy.md), [lds-bank-occupancy.md](lds-bank-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [architecture.md](architecture.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat waitcnt as **effective latency hiding**, not MaxWaves: PIX / `llvm-calc-occupancy` book VGPR∩LDS slots; the wave still parks on `s_waitcnt` until the named counter drops. Sibling waves hide that park **only if** they have useful work and are not themselves waiting on the same pipe. |
| **Take** | Software-pipeline with **non-zero** thresholds (`vmcnt(N)` / `lgkmcnt(N)`) so older results become usable while younger ops stay in flight (ISA §4.4: schedule long-latency, do unrelated work, wait when results are needed). Match the wait class to the pipe ([hip-craft.md](hip-craft.md) §2.2). |
| **Take** | On RDNA / gfx1030 keep the **VMCNT / VSCNT split**: loads → `vmcnt`; stores → `s_waitcnt_vscnt`. A GCN-style universal `s_waitcnt vmcnt(0)` does **not** drain stores and can pin the wave longer than needed for a load use. |
| **Take** | When decode looks “full” on VGPR∩LDS but RGP / thread-trace shows long **memory-wait** bars with little green (unhidden) latency, fix **wait placement / pipeline depth / cache thrash** before chasing another MaxWaves rung (GPUOpen Occupancy explained: wait ISA latency column + color = how much was hidden by other waves’ ALU). |
| **Leave** | Do **not** put waitcnt depth, VMCNT/VSCNT/LGKMCNT sizes, or “scoreboard occupancy %” into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. |
| **Leave** | Do **not** invent outstanding-request capacity for TCP/L0, miss-cycle tables, or a gfx1030 “max in-flight VMEM” number from this page — ISA states **counter widths and wait semantics**, not a published TCP queue depth for occupancy math. |
| **Leave** | Do **not** hand-author mid-burst `s_waitcnt` inside a hot same-type load streak you wanted hard-claused ([hard-clause-occupancy.md](hard-clause-occupancy.md) — waitcnt breaks `S_CLAUSE`). |
| **Leave** | Do not open UNC / retip extras for this page. |

## 1. What ISA says (RDNA 2, Document 70648)

### 1.1 Data dependency resolution (§4.4)

> Shader hardware can resolve most data dependencies, but a few cases must be explicitly handled by the shader program. In these cases, the program must insert **S_WAITCNT** instructions …
>
> These allow the shader writer to **schedule long-latency instructions, execute unrelated work, and specify when results** of long-latency operations are needed.

Same-type returns are ordered for VMEM loads (wait `VM_CNT < N` ⇒ prior loads done). Different types (e.g. LDS vs GDS on LGKM) can complete out-of-order. Scalar-memory reads on LGKM can return out-of-order — ISA: only **`S_WAITCNT` LGKM==0** is the legitimate full wait for that case (§7.3).

### 1.2 Counter widths (abbrev / status registers)

| Counter | ISA size (bits) | Counts (compute-relevant) |
|---|---|---|
| **VMCNT** | **6** | VMEM **loads** / atomics-with-return issued, not yet completed |
| **VSCNT** | **6** | VMEM **stores** / atomics-without-return issued, not yet completed |
| **LGKMCNT** | **4** | LDS, GDS, constant-fetch (SMEM), message |
| **EXPCNT** | **3** | Export / GDS / some VMEM write-data (graphics) |

VS_CNT decrements when store data has been written to **L2** (ISA §4.4) — fire-and-forget vs load completion.

### 1.3 Wait instructions (SOPP / SOPK)

| Op | Role |
|---|---|
| `S_WAITCNT` | Wait until `vmcnt` / `expcnt` / `lgkmcnt` are **≤** the packed SIMM16 fields (vmcnt bits 15:14+3:0; lgkmcnt 13:8; expcnt 6:4) |
| `S_WAITCNT_VSCNT` | Wait until `vscnt ≤` threshold (separate; not a field of `S_WAITCNT` on gfx10) |
| `S_WAITCNT_VMCNT` / `_LGKMCNT` / `_EXPCNT` | SOPK forms with null/literal thresholds |
| `S_WAITCNT_DEPCTR` | Bitmask wait on **VALU/SALU** dependency counters (`va_vdst`, `vm_vsrc`, …) — ISA: “intended for debug and bug-workarounds” |
| `S_ENDPGM` | Implicitly waits all counters (RDNA deck / hip-craft) |

`N` means “continue when **at most N** of that class remain outstanding.” `vmcnt(0)` = drain that class.

## 2. What LLVM / docs say on gfx1030

| Piece | Role |
|---|---|
| LLVM gfx1030 `waitcnt` operand | Documents packed fields: `vmcnt` 0..63, `lgkmcnt` 0..63, `expcnt` 0..7 — encoding ranges for the immediate ([ROCm llvm-project gfx1030_waitcnt](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.0/LLVM/llvm/html/AMDGPU/gfx1030_waitcnt.html)). Cite next to ISA **hardware** widths above; do not invent saturation policy. |
| `SIInsertWaitcnts` | Backend inserts waits for data + memory legalizer needs. Names: `LOAD_CNT`→VMcnt (pre-gfx12), `DS_CNT`→LGKMcnt, `STORE_CNT`→VScnt on gfx10/11. |
| `vscnt` use | Memory dependencies (SIMemoryLegalizer), not ordinary data deps; no `vscnt(0)` on function entry/return (`f2c164c` — hip-craft). |
| Barriers | Without auto-wait-before-barrier: insert all-zero waits **including vscnt** before `s_barrier` (hip-craft §2.2). |
| Hard clauses | `s_waitcnt` is a **clause breaker** in `SIInsertHardClauses` ([hard-clause-occupancy.md](hard-clause-occupancy.md)). |

hipcc owns emission for normal HIP. Craft = avoid wrong-pipe / early-zero waits in hot loops; check `.s` when using inline asm.

## 3. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Scratch / I$ / K$ / L0–IC | **No** | Latency / thrash (sibling pages) |
| Hard clause / SPI-ACE | **No** | Interleave / refill |
| **Waitcnt / scoreboard park** | **No** | Wave idle until counter ≤ N; slots stay reserved. Early drains / shallow pipeline → more park time → weaker latency hiding while theory still “full” |

GPUOpen Occupancy explained: occupancy = assigned waves / 16 slots per SIMD (RDNA2); purpose is **hiding memory latency** by switching waves. The RGP instruction-timing view colors a **memory wait** bar green when other waves’ VALU (yellow hatch = SALU) ran instead — that is effective hiding. A full reservation with every wave parked on `s_waitcnt` is high **theoretical** occupancy and poor **effective** hiding.

Mental model (no invented cycles):

```text
high reserved occupancy (VGPR∩LDS look full)
    + early s_waitcnt vmcnt(0) / wrong-pipe wait / no software pipeline
    → this wave parks until the counter drains
    → siblings help only if they have non-waiting work
    → measured ALU busy low / mem-wait high while PIX occupancy looks fine
```

That is **not** “raise MaxWaves.” Fix wait placement, overlap independent work, or reduce the miss traffic that keeps counters high ([l0-gl1-occupancy.md](l0-gl1-occupancy.md) / [l2-occupancy.md](l2-occupancy.md) / [kcache-occupancy.md](kcache-occupancy.md)).

## 4. Craft checklist (HIP → ISA)

1. Inner K / KV / weight loop: issue loads, do independent VALU / LDS address math, then `s_waitcnt vmcnt(N)` / `lgkmcnt(N)` with **N = still-useful in-flight**, not always 0 ([hip-craft.md](hip-craft.md) §2.3).
2. LDS reuse: wait **`lgkmcnt`**, not `vmcnt`. Global load use: **`vmcnt`**. Publish stores before barrier/atomic: **`vscnt`** (compiler usually emits this beside `__syncthreads` / atomics).
3. Do **not** insert waits mid same-type burst you want `S_CLAUSE`’d — wait **after** the clause.
4. SMEM / kernarg: LGKM; out-of-order scalar returns ⇒ full `lgkmcnt(0)` when you need all prior SMEM (ISA §7.3).
5. Confirm with disassembly + RGP wait bars. Occupancy still sized with `llvm-calc-occupancy` + `hipOccupancyMaxActiveBlocksPerMultiprocessor` — waitcnt does not appear in either.

## 5. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [hip-craft.md](hip-craft.md) §2 | Which counter / when; VMCNT vs VSCNT; pipeline `vmcnt(N)` | Explicit **not a PIX row**; effective vs theoretical occupancy framing |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Waitcnt **breaks** clauses | Scoreboard park as the dual: clauses reduce interleave; waits create the idle that occupancy hides |
| [kcache-occupancy.md](kcache-occupancy.md) / L0–IC pages | Miss → wave waits on waitcnt | Waitcnt as the **common scoreboard** mechanism, independent of which cache missed |
| [occupancy-composite.md](occupancy-composite.md) | Leave caches / clauses / SPI out of min | Waitcnt / scoreboard as named *(out of min)* sibling |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Lack of work / launch-rate | Orthogonal: SPI fills slots; waitcnt decides whether a filled slot is **running or parked** |

## 6. Sources

1. RDNA 2 ISA Reference Guide, Document ID **70648** — §4.4 Data Dependency Resolution (counters, schedule/wait contract); abbrev table VMCNT/VSCNT 6-bit, LGKMCNT 4-bit, EXPCNT 3-bit; SOPP `S_WAITCNT` / `S_WAITCNT_DEPCTR`; SOPK `S_WAITCNT_VSCNT` / `_VMCNT` / `_LGKMCNT`; §7.3 SMEM LGKM out-of-order → wait 0. Local text: `/workspace/rdna2-src/rdna2-isa.txt`.
2. LLVM gfx1030 waitcnt operand — https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.0/LLVM/llvm/html/AMDGPU/gfx1030_waitcnt.html
3. GPUOpen Occupancy explained — reservation limiters only; latency hiding via wave switch; RGP wait-instruction color = hidden vs parked — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
4. GPUOpen RDNA architecture deck — VMCNT/VSCNT split, fire-and-forget stores — https://gpuopen.com/download/RDNA_Architecture_public.pdf
5. LLVM `SIInsertWaitcnts` / hip-craft citations (`LOAD_CNT`/`DS_CNT`/`STORE_CNT`, `f2c164c`, barrier allZero incl. vscnt)
6. Wiki companions: [hip-craft.md](hip-craft.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [occupancy-composite.md](occupancy-composite.md)

## 7. Cross-links

| Page | Role vs this lock |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold page; waitcnt is *(out of min)* |
| [hip-craft.md](hip-craft.md) | Concrete wait class / pipeline patterns |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Mid-burst waitcnt breaks `S_CLAUSE` |
| [kcache-occupancy.md](kcache-occupancy.md) | Scalar miss → `lgkmcnt` park |
| [l0-gl1-occupancy.md](l0-gl1-occupancy.md) / [l2-occupancy.md](l2-occupancy.md) / [infinity-cache-occupancy.md](infinity-cache-occupancy.md) | Vector miss traffic that keeps VMCNT high |
| [lds-bank-occupancy.md](lds-bank-occupancy.md) | LDS bank conflicts stretch `lgkmcnt` wait — latency, not MaxWGsLDS |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Slot fill / launch-rate — not the scoreboard park |
| [exec-divergence-occupancy.md](exec-divergence-occupancy.md) | Lane-idle under EXEC vs time-idle on waitcnt — dual effective-occupancy sibling |
| [vgpr-occupancy.md](vgpr-occupancy.md) | Deeper software pipelines can raise live VGPR — measure the cliff; don’t invent a waitcnt MaxWaves row |
