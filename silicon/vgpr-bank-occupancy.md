# VGPR / SGPR operand-read banks vs effective occupancy (gfx1030)

Lock: **GFX10 register-file operand banks (VGPR 4 banks, round-robin by register index; SGPR 8 banks of aligned pairs) are not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** An instruction that reads two or more source operands from the same bank needs an extra operand-read cycle per collision; hardware *input operand gathering* tries to pre-load and hide it. The wave keeps its VGPR∩LDS slot either way. Since **LLVM 13** the compiler does **no** bank-aware reassignment (pass dropped: no measured win, real compile-time cost), so this is a **know-it, do-not-chase-it** row — never trade VGPRs or an occupancy rung for bank spreading.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **VGPR bank** | `bank = vN mod 4` — `v0, v4, v8, …` bank 0; `v1, v5, v9, …` bank 1; etc. (LLVM `GCNRegBankReassign.cpp` header, GFX10). Tuples start at their `sub0` bank and walk round-robin. |
| **SGPR bank** | 8 banks, allocated **in pairs**: `s[0:1], s[16:17], s[32:33]` bank 0; `s[2:3], s[18:19], …` bank 1; … (same source). |
| **Operand-read cycle** | “The shader can read one dword from each of these banks once per cycle. If an instruction has to read more register operands from the same bank an additional cycle is needed.” Example from the source: `V_FMA_F32 V111 = V0 + V4 * V8` → **3** operand-read cycles, **up to 2** stall cycles. |
| **Input operand gathering** | Same source: “HW attempts to pre-load registers through input operand gathering, but a stall cycle may occur if that fails.” The old pass weighted a collision lower when both operands were defined well before the use (“old enough to be pre-loaded”). |
| **DPP `BANK_MASK`** | **Different thing.** ISA 70648 §13.3 DPP16 `BANK_MASK[59:56]` is a 4-lane-group **write mask**; the ISA itself notes “the term ‘bank’ here is not the same as was used for the VGPR bank.” |
| **LDS bank** | **Different thing.** 64 × dword LDS banks per WGP ([lds-bank-occupancy.md](lds-bank-occupancy.md)). |
| **VOPD bank rules** | gfx11+ dual-issue legality (`GCNVOPDUtils.cpp` bank masks). **Dead** on gfx1030 — no VOPD ([valu.md](valu.md)). |

Date: **2026-10-05** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [vgpr-occupancy.md](vgpr-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md), [valu.md](valu.md), [hip-craft.md](hip-craft.md) §7.5, [trans-occupancy.md](trans-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [lds-bank-occupancy.md](lds-bank-occupancy.md), [fa-occupancy.md](fa-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat operand-bank collisions as **issue efficiency inside a filled slot**, not MaxWaves: PIX / `llvm-calc-occupancy` book VGPR∩LDS; a bank collision costs operand-read cycles on that instruction and never frees or reserves a wave slot. |
| **Take** | Size occupancy as usual ([vgpr-occupancy.md](vgpr-occupancy.md)): even the old reassign pass refused to pick a register above the **occupancy VGPR/SGPR cap** (`scavengeReg` stops at `getMaxNumVGPRs(Occupancy)`). Bank craft is strictly subordinate to the rung. |
| **Take** | When reading `.s` for a DOT/FMA inner loop, it is fine to *notice* three-source ops whose sources share `vN mod 4` (e.g. `v_fma_f32 vD, v0, v4, v8`). Treat it as a footnote; the first suspects remain VGPR rung, waitcnt parks, LDS conflicts, TRANS mix ([occupancy-composite.md](occupancy-composite.md)). |
| **Take** | Packed DOT (`fdot2` / `sdot4` builtins) reads two packed sources **plus** the accumulator — three operand reads, same bank rule as `v_fma`; input gathering + sibling waves are the intended hiding path. |
| **Leave** | Do **not** put VGPR/SGPR bank conflicts into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. |
| **Leave** | Do **not** hand-assign registers in inline asm, add padding VGPRs, or restructure a kernel to spread banks. LLVM measured **no** performance case for it on GFX10 and deleted the pass (D101313). No current hipcc pass models it for gfx1030. |
| **Leave** | Do **not** invent per-opcode stall counts, gather-window depth, or a rocprof counter for operand-bank stalls — none is documented in ISA 70648 or GPUOpen; the only numbers are the LLVM source comment above. |
| **Leave** | Do **not** confuse with DPP `BANK_MASK`, LDS banks, or gfx11 VOPD bank legality. |
| **Leave** | Do **not** open UNC / retip extras / restate the FA pin. |

## 1. What the sources say

### 1.1 LLVM `GCNRegBankReassign` (primary — only concrete numbers)

File header (pass present through LLVM 12; quoted from `llvmorg-12.0.0`):

> On GFX10 registers are organized in banks. VGPRs have 4 banks assigned in a round-robin fashion: v0, v4, v8... belong to bank 0. v1, v5, v9... to bank 1, etc. SGPRs have 8 banks and allocated in pairs, so that s0:s1, s16:s17, s32:s33 are at bank 0. s2:s3, s18:s19, s34:s35 are at bank 1 etc.
>
> The shader can read one dword from each of these banks once per cycle. If an instruction has to read more register operands from the same bank an additional cycle is needed. HW attempts to pre-load registers through input operand gathering, but a stall cycle may occur if that fails. For example V_FMA_F32 V111 = V0 + V4 * V8 will need 3 cycles to read operands, potentially incuring 2 stall cycles.

Implementation details relevant to occupancy:

| Code | Meaning |
|---|---|
| `NUM_VGPR_BANKS 4`, `NUM_SGPR_BANKS 8` | Bank counts above |
| `analyzeInst` | Stall estimate = popcount of already-used banks hit by each new **explicit use**; AGPRs skipped; a sub-register spanning ≥ 4 VGPRs (≥ 8 SGPR pairs) is skipped as “covers all banks” |
| `getOperandGatherWeight` | Collision weighted up only if an operand was defined within the last *StallCycles* instructions (not pre-loadable) |
| `scavengeReg` | **Stops at the occupancy register cap** — never raised VGPR/SGPR count to fix a bank |
| `ST->hasRegisterBanking()` | Gated on `FeatureRegisterBanking`, listed in the **`FeatureGFX10`** generation (so gfx1010 *and* gfx1030) in `llvmorg-12.0.0` `AMDGPU.td` |

Source: https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-12.0.0/llvm/lib/Target/AMDGPU/GCNRegBankReassign.cpp ; original review https://reviews.llvm.org/D61344 (gfx1010).

### 1.2 Pass removed (D101313, April 2021)

> Experiments show that the GCNRegBankReassign pass significantly impacts the compilation time and there is no case for which we see any improvement in performance. This patch removes this pass and its associated test cases from the tree.

Checked: `GCNRegBankReassign.cpp` exists at `llvmorg-12.0.1`, **404 at `llvmorg-13.0.0`**. `FeatureRegisterBanking` is still declared in `llvmorg-13.0.0` `AMDGPU.td` but has **no** match in current `main` `AMDGPU.td` / `GCNSubtarget.h` (checked 2026-10-05). The current `FeatureISAVersion10_3_0` list (LDS 1024 granule, Dot1/2/5/6/7/10, `FeatureMaxWavesPerEU16`, …) carries no banking feature.

Source: https://reviews.llvm.org/D101313

### 1.3 ISA 70648 / GPUOpen

- ISA 70648 does **not** define VGPR operand banks or a stall rule in the text we hold (`/workspace/rdna2-src/rdna2-isa.txt`). Its only mention is the DPP16 `BANK_MASK` note distinguishing DPP banks from “the VGPR bank.”
- GPUOpen Occupancy explained / GPC24: PIX limiters are VGPR / LDS / Thread Group Size / Barriers only — no register-bank row ([occupancy-composite.md](occupancy-composite.md)).
- RDNA deck: 1 VALU instruction / cycle / SIMD32; dependency stalls filled by other waves. Operand banks are not quantified there.

**Wiki is silent beyond §1.1** — do not extend it with invented gather depth or per-op tables.

## 2. Relation to occupancy — out of the PIX min

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Waitcnt / caches / clauses / SPI / EXEC / TRANS | **No** | Park / thrash / interleave / refill / lane-idle / SFU-starve (siblings) |
| **VGPR / SGPR operand banks** | **No** | Extra operand-read cycle(s) on a colliding instruction if gathering misses; slot unchanged; compiler does not model it (LLVM ≥ 13) |

Mental model (no invented cycles):

```text
three-source VALU / DOT with two+ sources in the same vN mod 4 bank
    → +1 operand-read cycle per extra same-bank read (LLVM comment)
    → hidden if operands were produced early (input operand gathering)
    → slot stays reserved; other waves may fill issue
    → LLVM measured no case where reassigning banks helped → footnote, not a lever
```

## 3. Craft checklist (HIP → ISA)

1. Size the rung first: `llvm-calc-occupancy` + metadata dump ([occupancy-dump.md](occupancy-dump.md)); `__launch_bounds__` as usual. Bank layout never justifies extra VGPRs.
2. Hot loop shape matters more than register numbers: keep ≥ 5 independent accumulators / unroll so DOT dest latency hides ([hip-craft.md](hip-craft.md) §7.5) — that also gives operand gathering time (sources defined early).
3. Do not hand-pin registers in inline asm for banks. If inline asm already pins registers for another reason, prefer not to put all three sources of an FMA/DOT in one `mod 4` bank — free choice only.
4. Profiling: there is no documented gfx1030 counter for operand-bank stalls; do not attribute a VALU-busy gap to banks without an A/B that changes only register assignment.

## 4. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [vgpr-occupancy.md](vgpr-occupancy.md) | 1024 file, granule 16, waves/EU ladder | Bank layout inside the file is **not** a rung; reassign never crossed the rung |
| [sgpr-occupancy.md](sgpr-occupancy.md) | SGPR not a limiter on GFX10+ | SGPR 8 paired banks — also out of min |
| [valu.md](valu.md) / [hip-craft.md](hip-craft.md) §7.5 | 1 VALU/cycle/SIMD32; no VOPD | Operand-read bank footnote; VOPD bank rules Dead |
| [lds-bank-occupancy.md](lds-bank-occupancy.md) | LDS 64-bank conflicts = latency | Different “bank”; register banks are per-instruction operand reads |
| [occupancy-composite.md](occupancy-composite.md) | Out-of-min siblings | Register operand banks as named *(out of min)* row |

## 5. Sources

1. LLVM `GCNRegBankReassign.cpp` @ `llvmorg-12.0.0` — bank layout, read rule, gather, occupancy-capped scavenging — https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-12.0.0/llvm/lib/Target/AMDGPU/GCNRegBankReassign.cpp
2. LLVM D61344 — gfx1010 GCNRegBankReassign pass — https://reviews.llvm.org/D61344
3. LLVM D101313 — drop the pass (no perf win, compile-time cost) — https://reviews.llvm.org/D101313
4. LLVM `AMDGPU.td` @ `llvmorg-12.0.0` (`FeatureRegisterBanking` in `FeatureGFX10`) and current `main` (absent; `FeatureISAVersion10_3_0`) — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPU.td
5. RDNA 2 ISA Reference Guide, Document **70648** — §13.3 DPP16 `BANK_MASK` note. Local `/workspace/rdna2-src/rdna2-isa.txt`
6. GPUOpen Occupancy explained — PIX limiters — https://gpuopen.com/learn/occupancy-explained/
7. LLVM `GCNVOPDUtils.cpp` — gfx11+ VOPD bank masks (Dead on gfx1030) — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNVOPDUtils.cpp

## 6. Cross-links

| Page | Role vs this lock |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold page; register banks are *(out of min)* |
| [vgpr-occupancy.md](vgpr-occupancy.md) | The real VGPR rung |
| [sgpr-occupancy.md](sgpr-occupancy.md) | SGPR not a limiter; paired banks footnote |
| [valu.md](valu.md) | Issue table; no VOPD |
| [hip-craft.md](hip-craft.md) | §7.5 issue model; accumulator unroll |
| [lds-bank-occupancy.md](lds-bank-occupancy.md) | The *other* bank conflict (LDS) |
| [trans-occupancy.md](trans-occupancy.md) | Sibling: SFU mix inside a filled slot |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Sibling: time-idle park |
