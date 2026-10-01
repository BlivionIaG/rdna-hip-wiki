# EXEC / lane divergence vs effective occupancy (gfx1030)

Lock: **`EXEC` mask / lane divergence is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** Control flow on RDNA is **scalar** (whole-wave); lane-varying predicates update **`EXEC`**, and inactive lanes treat vector ops as **NOP**. The wave still **occupies** its VGPR∩LDS slot while doing little or no useful SIMD work. Completes the occupancy-adjacent set next to [waitcnt-occupancy.md](waitcnt-occupancy.md) (parked while waiting) and [spi-ace-occupancy.md](spi-ace-occupancy.md) (empty slots): here the slot is **filled** but **lane-idle**.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **`EXEC`** | Per-wave bit mask (ISA §3.3): 1 = lane executes vector op, 0 = dormant / NOP. Wave32 uses the **low 32** bits; upper half ignored. |
| **`EXECZ` / `EXECNZ`** | Status helpers: all-active-bits zero / not. Used by `S_CBRANCH_EXECZ` / `_EXECNZ`. |
| **Lane divergence** | Lanes disagree on a predicate → compiler saves/restores `EXEC` (`s_and_saveexec` / `s_or_b64 exec, …`) and may emit `s_cbranch_execz` to skip a block when no lane is active. |
| **Uniform control flow** | Whole-wave branch (`S_CBRANCH_*` on SCC / VCCZ / EXECZ) — PC change, not a per-lane branch unit. |

Date: **2026-10-01** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [wave-size-occupancy.md](wave-size-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [hip-craft.md](hip-craft.md) §2.1 / DPP, [valu.md](valu.md), [architecture.md](architecture.md), [fa-occupancy.md](fa-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat divergence as **effective SIMD utilization**, not MaxWaves: PIX / `llvm-calc-occupancy` book wave slots; a wave with sparse `EXEC` still holds those slots while inactive lanes burn issue width as NOPs (ISA §2 / §3.3). |
| **Take** | Prefer **uniform** predicates (same outcome for all lanes in the wave) or **predicated** selects (`v_cndmask` / compare→mask) over divergent `if` trees in hot DOT / K / KV loops. When a mask is lane-varying by construction (causal window, expert routing), keep the hot path as **mask arithmetic** so both sides are not serialized under alternating `EXEC`. |
| **Take** | On HIP default **wave32**, remember ISA’s EXEC=0 skip rules: **VALU** may be skipped (unless it writes SGPR/VCC); **wave32 memory instructions are not skipped** when `EXEC` is zero — use `S_CBRANCH_EXECZ` / compiler `s_cbranch_execz` to jump over dead VMEM blocks (ISA §3.3). Wave64 can skip a half when that half’s `EXEC` is 0 ([wave-size-occupancy.md](wave-size-occupancy.md)). |
| **Take** | When dumping ISA, expect `s_and_saveexec_*` / `s_or_b* exec` around divergent regions and optional `s_cbranch_execz`. Spurious execz on **always-partially-active** predicates (e.g. even/odd tid) is a compiler peephole smell (LLVM `SIAnnotateControlFlow` + `SIPreEmitPeephole`; varying-weight work after PR #117567 → #123749) — not a PIX occupancy miss. |
| **Leave** | Do **not** put active-lane count, `EXEC` density, or “divergence occupancy %” into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. |
| **Leave** | Do **not** invent cycles-per-divergent-branch, lane-utilization tables, or a gfx1030 “branch unit occupancy” number. Do not open UNC / retip extras for this page. |
| **Leave** | Do **not** copy CDNA/MFMA “matrix wave” divergence lore onto packed DOT — gfx1030 DOT is per-lane (`fdot2` / `sdot4`), still under `EXEC`. |

## 1. What ISA says (RDNA 2, Document 70648)

### 1.1 Scalar control flow + EXEC (§2)

> All kernel control flow is handled using **scalar ALU** instructions. … every wavefront has an **EXECute mask** that determines which work-items are active … Active work-items execute the vector instruction, and dormant ones treat the instruction as a **NOP**.

`EXEC` is changed by SALU / `V_CMPX`. It **affects** VALU, VMEM, LDS, GDS, export. It does **not** affect scalar ALU/memory or the branch instructions themselves (§3.3).

### 1.2 EXECute mask (§3.3)

| Rule | Detail |
|---|---|
| Bits | 1 = execute, 0 = do not |
| Wave32 | Upper 32 bits of `EXEC` / `VCC` **ignored**; `EXECZ` reflects the low half only |
| Skip when EXEC=0 | VALU skippable unless writes SGPR; **wave32 VMEM not skipped**; wave64 VMEM may skip one half, not both |
| Fast path | Use **`CBRANCH` / `S_CBRANCH_EXECZ`** when the mask is likely all-zero |

### 1.3 Branching (§4.2)

Whole-wave PC changes: `S_BRANCH`, `S_CBRANCH_{VCCZ,VCCNZ,EXECZ,EXECNZ,SCCZ,SCCNZ}`, `S_SETPC` / `S_SWAPPC` / `S_CALL_B64`. Vector compares set **VCC**; `VCCZ`/`VCCNZ` decide wave-level branches. Lane-varying `if` is **mask update + optional execz skip**, not a per-lane branch pipe.

Subvector loops (`S_SUBVECTOR_LOOP_*`) force half-`EXEC` passes for wave64 — HIP ships wave32; leave subvector as ISA awareness only.

## 2. What LLVM / HIP do on gfx1030

| Piece | Role |
|---|---|
| `SIAnnotateControlFlow` | Lowers divergent CFG with `amdgcn_if` / `amdgcn_else`; later machine code uses `s_and_saveexec` / restore |
| `s_cbranch_execz` | Skip a region when no lane remains active — good when the wave is often fully inactive; wasteful when every wave is **partially** active |
| `SIPreEmitPeephole` | May delete execz when branch weights / heuristics say both sides likely run |
| Varying-branch work | LLVM PR #117567 (closed) → follow-on #123749 (intrinsic argument) — **Watch** on tip hipcc; not a dest craft pin |
| Predication | Uniform or mask-select paths avoid saveexec churn; DPP / `permlane` / `__shfl_xor` for intra-wave reduce without LDS ([hip-craft.md](hip-craft.md) §5.2) |

hipcc for `--offload-arch=gfx1030` owns emission. Craft = keep hot loops **uniform or predicated**; inspect `.s` when a decode/prefill inner loop is branchy.

## 3. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Waitcnt / caches / clauses / SPI | **No** | Park / thrash / interleave / refill (siblings) |
| **`EXEC` / lane divergence** | **No** | Slots stay reserved; inactive lanes NOP; wave32 still **issues** VMEM under empty `EXEC` unless branched over — weak **effective** SIMD work while theory looks full |

GPUOpen Occupancy explained: occupancy = assigned waves / 16 slots per SIMD (RDNA2); purpose is **hiding memory latency** by switching waves. It lists VGPR / LDS / threadgroup / barriers as reservation limiters — **not** active-lane density. Divergence is a **utilization** problem on an already-assigned wave, orthogonal to whether SPI filled the slot.

Mental model (no invented cycles):

```text
high reserved occupancy (VGPR∩LDS look full)
    + divergent if/else or sparse causal/expert masks
    → many lanes EXEC=0 on VALU/VMEM (NOP)
    → wave32 may still issue VMEM under EXEC=0 unless s_cbranch_execz
    → measured useful FLOP/byte low while PIX occupancy looks fine
```

That is **not** “raise MaxWaves.” Fix predicate shape (uniformize, predication, re-tile so masks are wave-aligned), or accept the mask cost as algorithm — do not chase another occupancy rung.

## 4. Craft checklist (HIP → ISA)

1. Hot K / DOT / KV loop: avoid lane-varying C++ `if` that wraps large VMEM/VALU blocks; prefer compare → `v_cndmask` / masked store, or structure the grid so the predicate is **uniform per wave**.
2. When a divergent region is unavoidable and often **all-inactive**, keep the compiler’s `s_cbranch_execz` (or an equivalent early-out); when it is **always partially active**, expect execz to be useless — don’t hand-add more.
3. Wave32 produce: do not assume empty-`EXEC` VMEM is free — ISA does not skip it; branch over or don’t emit.
4. Intra-wave reductions: DPP / `permlane` / `__shfl_xor` under full `EXEC` ([hip-craft.md](hip-craft.md) §5.2) — divergence mid-reduce needs an explicit mask story.
5. Occupancy still sized with `llvm-calc-occupancy` + `hipOccupancyMaxActiveBlocksPerMultiprocessor` — `EXEC` density does not appear in either.

## 5. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](architecture.md) / [hip-craft.md](hip-craft.md) | Wave32 native; EXEC half-skip for wave64 VALU/VMEM | Explicit **not a PIX row**; wave32 **VMEM not skipped** under EXEC=0; effective vs theoretical framing |
| [wave-size-occupancy.md](wave-size-occupancy.md) | Wave32 vs wave64 rescales VGPR/N | Divergence / EXEC skip asymmetry as occupancy-adjacent |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Scoreboard park | Dual: waitcnt = **time-idle** slot; divergence = **lane-idle** slot |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Empty / under-filled slots | Orthogonal: SPI fills; EXEC decides how many lanes in a filled slot do work |
| [occupancy-composite.md](occupancy-composite.md) | Leave caches / clauses / waitcnt / SPI out of min | EXEC / divergence as named *(out of min)* sibling |
| [fa-occupancy.md](fa-occupancy.md) | FA LDS/VGPR/`__launch_bounds__` | Do not restate FA pin; causal/window masks are the classic divergence class — craft only |

## 6. Sources

1. RDNA 2 ISA Reference Guide, Document ID **70648** — §2 program organization (scalar CF + EXEC); §2.1 wave32/64 half-skip; §3.3 EXECute mask + EXEC=0 skip limitations; §4.2 branching / subvector. Local text: `/workspace/rdna2-src/rdna2-isa.txt`.
2. GPUOpen Occupancy explained — reservation limiters only; latency hiding via wave switch — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
3. LLVM AMDGPUUsage — gfx1030 / wavefrontsize / cumode — https://llvm.org/docs/AMDGPUUsage.html
4. LLVM control-flow / execz — `SIAnnotateControlFlow`; PR #117567 (closed) / #123749 (follow-on) on spurious `s_cbranch_execz` when predicates are likely varying
5. Wiki companions: [occupancy-composite.md](occupancy-composite.md), [wave-size-occupancy.md](wave-size-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [hip-craft.md](hip-craft.md)

## 7. Cross-links

| Page | Role vs this lock |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold page; EXEC/divergence is *(out of min)* |
| [wave-size-occupancy.md](wave-size-occupancy.md) | Wave32 vs wave64 EXEC half-skip rules |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Time-idle park vs lane-idle NOP |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Branches / SALU break `S_CLAUSE` — divergent CF also breaks clause adjacency |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | Slot fill — not lane activity inside a slot |
| [hip-craft.md](hip-craft.md) | Wave32 contract; DPP / predication craft |
| [valu.md](valu.md) | Per-lane DOT still under EXEC |
| [vgpr-occupancy.md](vgpr-occupancy.md) | Divergent live ranges can raise VGPR — measure the cliff; don’t invent an EXEC MaxWaves row |
