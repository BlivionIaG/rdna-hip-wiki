# Hard clauses (`S_CLAUSE`) vs effective occupancy (gfx1030)

Lock: **hard memory/VALU clauses are not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** On gfx10+ (including **gfx1030**), HW no longer auto-detects pre-gfx10 **soft clauses**; LLVM inserts **`S_CLAUSE`** to mark **hard clauses**. A clause briefly **locks the instruction arbiter onto one wave** for that instruction type so a same-type burst runs uninterrupted — cache-coherency / burst benefit, **not** a fifth PIX resource. Completes the occupancy-adjacent set next to [spi-ace-occupancy.md](spi-ace-occupancy.md) (interleave / refill) and the cache thrash pages.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **Soft clause** (pre-gfx10) | HW auto-detected same-type memory sequences. **Removed on gfx10** (LLVM `SIInsertHardClauses` header). |
| **Hard clause** | Explicit `S_CLAUSE` + 2–63 same-type instructions (ISA §4.1.1 / SOPP `S_CLAUSE`). |
| **XNACK restartable group** | Separate LLVM concern (`SIFormMemoryClauses` / hazard recognizer). Rules differ (e.g. `s_nop` breaks a restartable group but may sit inside a hard clause). |

Date: **2026-09-29** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [hip-craft.md](hip-craft.md) § waitcnt, [cache-policy.md](cache-policy.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [l2-occupancy.md](l2-occupancy.md), [infinity-cache-occupancy.md](infinity-cache-occupancy.md), [kcache-occupancy.md](kcache-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [architecture.md](architecture.md).

## Take / Leave

| | |
|---|---|
| **Take** | On gfx1030, treat **hard clauses as compiler craft**: keep same-type VMEM / FLAT / SMEM loads (or stores) **adjacent** in the inner burst so `SIInsertHardClauses` can emit `S_CLAUSE`. Do not sprinkle `s_waitcnt` / SALU / branches through a hot load streak. |
| **Take** | Read ISA §4.1.1 literally: a clause forces the HW to **service only one wave for that instruction type** for the clause duration — temporary **less** interleave on that pipe, in exchange for burst / intra-clause cache behavior (ISA §2.4: overlapping loads in a load clause are cached w.r.t. each other). Occupancy slots stay booked; effective latency hiding on *other* waves for that pipe pauses. |
| **Take** | gfx1030 LLVM max hard-clause length is **63** (`FeatureMaxHardClauseLength63` on `FeatureGFX10`). Length encoding: `(SIMM16[5:0] + 1)`, must be **≥ 2** (ISA SOPP `S_CLAUSE`). GFX11 drops the feature max to **32** — do not copy that limit onto V620. |
| **Take** | When dumping ISA (`-save-temps` / `llvm-objdump -d`), expect `s_clause` before clustered `global_*` / `buffer_*` / `flat_*` / `s_load_*` runs. Absence in a hot decode loop is a **scheduler adjacency** smell, not a PIX occupancy miss. |
| **Leave** | Do **not** put clause length / `S_CLAUSE` into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. LLVM `computeOccupancy` still mins WG+LDS, VGPR, SGPR only. |
| **Leave** | Do **not** assume pre-gfx10 soft-clause auto-detection still runs on gfx1030. Do not hand-author `S_CLAUSE` in HIP inline asm as a default craft move. |
| **Leave** | Do **not** conflate hard clauses with XNACK restartable groups (`SIFormMemoryClauses` — pointer live-range / soft-clause rename when XNACK is on; RP tracker there is for **not dropping occupancy while bundling**, not a PIX row). V620 produce does not need an XNACK craft ticket from this page. |
| **Leave** | Do not invent clause-break cycle tables, arbiter lock latencies, or “clause occupancy %”. Do not open UNC / retip extras for this page. |

## 1. What ISA says (RDNA 2, Document 70648)

### 1.1 Instruction clauses (§4.1.1)

> An instruction clause is a group of instructions of the **same type** which are to be executed in an **uninterrupted** sequence. Normally the shader hardware may **interleave** instructions from different waves … a clause can … force the hardware to service **only one wave** for a given instruction type for the duration of the clause.

Clause types (type = first instruction after `S_CLAUSE`):

- VALU
- SMEM
- LDS
- FLAT
- Texture, buffer, global and scratch

Must contain **only one** instruction type.

### 1.2 `S_CLAUSE` encoding (SOPP)

| Field | Rule |
|---|---|
| Length | `(SIMM16[5:0] + 1)` → **2 … 63** instructions |
| Break stride | `SIMM16[11:8]` = instructions per clause break (0 = no breaks) |
| Mix bans | Texture/Buffer/Global/Scratch: **may not mix** atomics, loads & stores. Flat: loads, stores, atomics **may not** combine |
| Illegal inside | SALU, Export, Branch, Message, GDS |
| Halt / kill | Breaks the clause |

### 1.3 Arbiter lock + load-clause cache (§7.4 / §2.4)

- §7.4 (SMEM): a clause is `S_CLAUSE` + **2–63** instructions; **locks the instruction arbiter onto this wave** until the clause completes. A *group* (same-type consecutive ops **without** `S_CLAUSE`) is **not** forced to run as one wave’s exclusive burst.
- §2.4: load requests that **overlap within the clause** are **cached with respect to each other** (clause-local coherency / reuse framing — not a published miss-cycle table).

## 2. What LLVM does on gfx1030

| Piece | Role |
|---|---|
| `FeatureGFX10` → `FeatureMaxHardClauseLength63` | `hasHardClauses()` true; `maxHardClauseLength()` = **63** (`AMDGPU.td` / `GCNSubtarget`) |
| `SIInsertHardClauses` | After scheduling: insert `S_CLAUSE` over adjacent same-type mem ops. Header: soft auto-detect **removed on gfx10**; hard clauses for **cache coherency benefits**. |
| gfx10 clause kinds (insert pass) | `HARDCLAUSE_VMEM` (texture/buffer/global/scratch, incl. segment-specific FLAT), `HARDCLAUSE_FLAT` (true flat), `HARDCLAUSE_SMEM`; LDS TODO in pass; VALU clauses **not** formed (“not clear what benefit”) |
| Illegal / breakers | `s_waitcnt`, SALU, branch, message, GDS, export; `s_nop` is **INTERNAL** (allowed mid-clause) |
| `amdgpu-hard-clause-length-limit` | Hidden cl::opt / fn attr; insert pass also lies to `shouldClusterMemOps` about cluster size **after RA** (RP limit no longer needed) |
| `SIFormMemoryClauses` | **Different**: XNACK restartable groups; default `amdgpu-max-memory-clause=15`; uses `GCNDownwardRPTracker` so bundling **does not cut** recorded occupancy |

hipcc / ROCm LLVM for `--offload-arch=gfx1030` owns emission. Craft = keep bursts clause-friendly; do not fight the pass with mid-burst waits.

## 3. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Scratch / I$ / K$ / L0–IC | **No** | Latency / thrash (sibling pages) |
| SPI / ACE / grid fill | **No** | Lack of work / launch-rate |
| **Hard clause (`S_CLAUSE`)** | **No** | Temporarily **reduces interleave** on one instruction pipe for the issuing wave; slots stay reserved. Helps burst/coherency; can **starve** other waves’ issue on that pipe for the clause length |

GPUOpen Occupancy explained still lists only VGPR / (fixed) SGPR / LDS / threadgroup / barriers as reservation limiters. Clauses are a **runtime issue-arbiter** behavior on top of that booking model — closer to [spi-ace-occupancy.md](spi-ace-occupancy.md) (how waves get cycles) than to VGPR file math.

Mental model (no invented cycles):

```text
high reserved occupancy
    + long hard VMEM clause on wave A
    → wave A owns that pipe for N same-type ops
    → waves B.. on the same WGP wait for that pipe
    → measured “ALU busy / mem busy” can look uneven while PIX occupancy looks full
```

That is **not** “raise MaxWaves.” If a skinny decode loop never clauses, fix **source adjacency** / avoid waitcnt in the burst; if clauses are huge and other waves starve on a shared pipe, that is a **measured** schedule trade (shorter bursts vs coherency), not a `__launch_bounds__` ticket by itself.

## 4. Craft checklist (HIP → ISA)

1. Inner K/KV / weight burst: consecutive `global_load_*` / `__ldg`-style loads with **one** base + offsets; no scalar math / `__syncthreads` / atomics mid-burst.
2. Split **load** clauses from **store** / **atomic** clauses (ISA mix ban).
3. Wait **after** the clause (`s_waitcnt vmcnt` / HIP fence), not between siblings you wanted claused ([hip-craft.md](hip-craft.md) § waitcnt — `vmcnt` vs `vscnt` on RDNA).
4. Confirm with disassembly: `s_clause 0xN` then N+1 same-class ops. Missing `s_clause` on an obviously adjacent run → check for `s_waitcnt`, SALU, or type mix breaking the pass.
5. Occupancy still sized with `llvm-calc-occupancy` + `hipOccupancyMaxActiveBlocksPerMultiprocessor` — clauses do not appear in either.

## 5. Sources

1. RDNA 2 ISA Reference Guide, Document ID **70648** — §2.4 Device Memory (load clause overlap cache); §4.1.1 Instruction Clauses; §7.4 Scalar Memory Clauses and Groups; SOPP `S_CLAUSE` (length 2–63, mix bans, illegal types). Local text: `/workspace/rdna2-src/rdna2-isa.txt`.
2. LLVM `SIInsertHardClauses.cpp` (main) — soft→hard history; gfx10 VMEM/FLAT/SMEM kinds; insert after schedule.
3. LLVM `AMDGPU.td` — `FeatureGFX10` includes `FeatureMaxHardClauseLength63`; `FeatureGFX11` uses `FeatureMaxHardClauseLength32`.
4. LLVM `GCNSubtarget.h` — `hasHardClauses()` / `maxHardClauseLength()`.
5. LLVM `SIFormMemoryClauses.cpp` — XNACK restartable groups (distinct); RP / occupancy preservation while bundling.
6. GPUOpen Occupancy explained — reservation limiters only; latency-hiding definition — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26).

## 6. Cross-links

| Page | Role vs this lock |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold page; hard clause is *(out of min)* |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Issue / refill / lack of work — arbiter cousin |
| [hip-craft.md](hip-craft.md) | waitcnt / scopes that **break** clauses if placed mid-burst |
| [kcache-occupancy.md](kcache-occupancy.md) | SMEM / kernarg path; SMEM hard clauses are the scalar cousin |
| [l0-gl1-occupancy.md](l0-gl1-occupancy.md) / [l2-occupancy.md](l2-occupancy.md) / [infinity-cache-occupancy.md](infinity-cache-occupancy.md) | Cache thrash from *too many* waves — orthogonal; clauses change *who issues* on a pipe, not MaxWaves |
| [vgpr-occupancy.md](vgpr-occupancy.md) | Scheduler may keep more live addresses to form longer clusters pre-RA — measure VGPR, don’t invent a clause limiter |
