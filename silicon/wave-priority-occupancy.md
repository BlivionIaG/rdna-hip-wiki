# Wave priority (`S_SETPRIO`) / `S_SLEEP` vs effective occupancy (gfx1030)

Lock: **Wave priority and sleep are arbitration / idle knobs inside already-reserved slots — not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** `S_SETPRIO` reorders *which resident wave* the arbiter picks; `S_SLEEP` parks a resident wave for a coarse clock window. Neither adds or frees VGPR∩LDS slots. Completes the "out of min" set next to [waitcnt-occupancy.md](waitcnt-occupancy.md) (park on a counter) and [hard-clause-occupancy.md](hard-clause-occupancy.md) (arbiter locked for a burst): here the arbiter is **biased** (priority) or the wave **opts out** for a while (sleep).

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

Date: **2026-10-06** Europe/Paris.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **SPI_PRIO** | STATUS[2:1]. Priority set by the SPI at wave create (0 lowest, 3 highest). For compute, comes from `COMPUTE_PGM_RSRC1.PRIORITY` (bits 11:10) which the code object must leave **0**; CP fills it (AMDGPUUsage). |
| **USER_PRIO** | STATUS[4:3]. Set by the shader with `S_SETPRIO SIMM16[1:0]`. |
| **Overall priority** | ISA: `{SPIPrio[1:0] + UserPrio[1:0], WaveAge[3:0]}` — priority sum is the high field, wave age is the low-order tiebreak. The ISA does not spell out the age direction or saturation of the sum; do not invent it. |
| **`S_SLEEP N`** | Sleep for `64*(N-1) .. 64*N` clocks, `N = SIMM16[6:0]`, approximate; `N=0` no sleep. (ISA §4.1 summary table says "64 – 960 clocks"; the opcode text allows `N` up to 127. Treat as approximate; the opcode text is the encoding contract.) |
| **`S_WAKEUP`** | Pings all waves of the **same threadgroup** out of `S_SLEEP` early; no-op if they are not sleeping or the wave is not in a threadgroup. Race-safe by construction (a missed ping just finishes its sleep). |
| **Stream priority** | `hipStreamCreateWithPriority` is a queue-level knob, a different layer from wave `S_SETPRIO`. Do not conflate. |

Companions: [occupancy-composite.md](occupancy-composite.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [hard-clause-occupancy.md](hard-clause-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md), [trans-occupancy.md](trans-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [rdna-allreduce.md](rdna-allreduce.md), [hip-craft.md](hip-craft.md), [architecture.md](architecture.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat `S_SETPRIO` / `S_SLEEP` as **effective-occupancy arbitration**, not MaxWaves: a sleeping or deprioritised wave still holds its VGPR∩LDS slot. Size with `llvm-calc-occupancy` + `hipOccupancyMaxActiveBlocksPerMultiprocessor` as before. |
| **Take** | Spin / poll loops on device flags: keep a `__builtin_amdgcn_s_sleep(N)` backoff between polls so a polling wave yields issue to siblings on the same SIMD. Extras custom AR already does this (`__builtin_amdgcn_s_sleep(8)` ≈ 448–512 clocks per back-off; [rdna-allreduce.md](rdna-allreduce.md) item 5) — this page only names why it is cheap for neighbours. |
| **Take** | Intra-WG producer/consumer waits on a memory flag: `S_SLEEP` in the waiter + `S_WAKEUP` from the writer is ISA-blessed and race-safe (missed ping ⇒ sleep just expires). No HIP builtin for `s_wakeup` on gfx10 (LLVM has only `int_amdgcn_s_wakeup_barrier`, gfx12) — inline `asm volatile("s_wakeup")` if ever needed. **Later** craft only; prefer `__syncthreads()` when a plain barrier fits ([barrier-occupancy.md](barrier-occupancy.md)). |
| **Take** | Treat `__builtin_amdgcn_s_setprio(p)` as a **measured A/B**, never default craft. If tried, bracket a compute burst (raise, then **restore 0**) so a long VALU stretch in one wave does not starve younger waves' VMEM issue — the same intent as LLVM's pass (§2). |
| **Take** | LLVM pass `-mllvm -amdgpu-set-wave-priority` (off by default) is a zero-code A/B: raises to prio 3 at entry, lowers to 0 after the last VMEM load that precedes ≥ threshold VALU (default 100; per-kernel `"amdgpu-wave-priority-threshold"` attribute). Confirm it exists in the pinned ROCm LLVM (see [toolchain/matrix.md](../toolchain/matrix.md)) with `--save-temps` + grep `s_setprio` before relying on it — **Watch**, not verified on the pin here. |
| **Leave** | Do **not** put priority, sleep time, or wake-up latency into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. |
| **Leave** | Do **not** copy radiance-vllm-mxfp4's `s_setprio(1)` around a WMMA run: it is a gfx11/gfx12 WMMA kernel and they measured it as a loss on every prefill shape they tried (`autoround-tests/RESULTS.md`, "Measured and rejected"). Data point only; WMMA is absent on gfx1030. |
| **Leave** | Do **not** use CDNA `s_setprio` around-MFMA recipes as gfx1030 craft — no MFMA here ([hip-craft.md](hip-craft.md) §7.5). |
| **Leave** | Do **not** use `s_sleep_var` (`__builtin_amdgcn_s_sleep_var`, SOP1 gfx12/gfx13 only) or `s_setprio_inc_wg` (GFX1250/GFX13) on gfx1030. |
| **Leave** | Do **not** set `COMPUTE_PGM_RSRC1.PRIORITY` in a hand-written code object; AMDGPUUsage says it must be 0. |
| **Leave** | Do **not** open UNC / retip extras / restate the FA pin. |

## 1. What the ISA says (RDNA 2, Document 70648)

- §3.4 Status registers: `SPI_PRIO` 2:1 (set at create), `USER_PRIO` 4:3 (set by `S_SETPRIO`); 0 lowest, 3 highest.
- §4.1 Table 7 Control Instructions: `S_SETPRIO` "Modifies the priority of this wavefront"; `S_SLEEP` "sleep for 64 – 960 clock cycles"; `S_NOP` repeatable up to eight times in hardware (contrast: NOP burns issue, sleep yields).
- SOPP opcode 15 `S_SETPRIO`: user prio = `SIMM16[1:0]`; overall = `{SPIPrio + UserPrio, WaveAge[3:0]}`.
- SOPP opcode 14 `S_SLEEP`: `64*(SIMM16[6:0]-1) .. 64*SIMM16[6:0]` clocks, approximate; 0 = no sleep.
- SOPP opcode 3 `S_WAKEUP`: wakes all waves with the same threadgroup ID from `S_SLEEP`; ignored when not sleeping; NOP outside a threadgroup; designed for efficient memory polling / fBarrier.

Local text: `/workspace/rdna2-src/rdna2-isa.txt` (lines ~989–994, ~1687–1707, ~7363–7375, ~7475–7486).

## 2. What LLVM / HIP do

| Piece | Role on gfx1030 |
|---|---|
| `__builtin_amdgcn_s_setprio(imm16)` → `int_amdgcn_s_setprio` → `S_SETPRIO` | Available (SOPP gfx10 encoding). Immediate only. |
| `__builtin_amdgcn_s_sleep(imm)` → `S_SLEEP` | Available; immediate only. |
| `s_wakeup` | Instruction exists for GFX8–GFX12 (`S_WAKEUP`, gfx10 opcode 0x003) but **no** clang builtin; `int_amdgcn_s_wakeup_barrier` is gfx12 named-barrier only. |
| `__builtin_amdgcn_s_sleep_var`, `__builtin_amdgcn_s_setprio_inc_wg` | gfx12+/GFX1250+ only — **Dead** here. |
| `AMDGPUSetWavePriority` pass | Hidden flag `-amdgpu-set-wave-priority` (default **false**). Entry functions only. Finds VMEM loads followed by ≥ threshold VALU on some path (backedges ignored); inserts `s_setprio 3` at entry (before first VALU) and `s_setprio 0` after the last such VMEM load / on exit edges. Intent (file header): "allow younger waves to issue their VMEM instructions as well." |
| `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` | `min(WG+LDS, SGPR, VGPR)` — **no** priority / sleep term. |

## 3. Relation to occupancy — out of the PIX min

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)    // unchanged
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Waitcnt / caches / clauses / EXEC / TRANS / banks / SPI | **No** | Siblings |
| **`S_SETPRIO`** | **No** | Reorders arbiter choice among resident waves; can starve low-prio siblings if left raised |
| **`S_SLEEP` / `S_WAKEUP`** | **No** | Resident wave voluntarily idle for a coarse window; slot stays reserved; siblings get its issue cycles |

Mental model:

```text
poll loop without backoff → wave re-issues loads/SALU every cycle it wins → steals issue from siblings on the SIMD
poll loop with s_sleep(N) → wave out of arbitration ~64·N clocks → siblings issue; slot still held
s_setprio raised forever → age tiebreak overridden → younger waves' VMEM issue can lag (the thing LLVM's pass lowers prio to avoid)
```

## 4. Craft checklist (HIP → ISA)

1. Device-side spin waits: always back off with `__builtin_amdgcn_s_sleep(N)`; pick N for the expected wait (each unit ≈ 64 clocks). Keep spin caps / abort paths as extras already does.
2. Never leave `s_setprio` raised at kernel exit paths; pair every raise with a restore to 0.
3. A/B `-mllvm -amdgpu-set-wave-priority` only on memory-then-long-VALU kernels (dequant-then-DOT shape), and only after VGPR∩LDS sizing is settled. Report as Watch until a measured win exists.
4. Inspect `.s` for stray `s_setprio` / `s_sleep` from third-party headers when porting RDNA3/4 or CDNA kernels.

## 5. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [rdna-allreduce.md](rdna-allreduce.md) | Poll backoff `s_sleep(8)` | ISA clock window for `N`, why it yields issue, `S_WAKEUP` scope (same TG only — not cross-rank) |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | Counter park | Voluntary park (`S_SLEEP`) vs hardware wait |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | Arbiter locked to one wave for a clause | Arbiter **biased** by priority |
| [barrier-occupancy.md](barrier-occupancy.md) | Barrier slots | `S_SLEEP`+`S_WAKEUP` as intra-TG alternative (Later) |
| [occupancy-composite.md](occupancy-composite.md) | Out-of-min siblings | Priority / sleep as named *(out of min)* row |

## 6. Sources

1. RDNA 2 ISA Reference Guide, Document ID **70648** — §3.4 STATUS `SPI_PRIO`/`USER_PRIO`; §4.1 Table 7; SOPP `S_WAKEUP` (3), `S_SLEEP` (14), `S_SETPRIO` (15). Local: `/workspace/rdna2-src/rdna2-isa.txt`.
2. LLVM `llvm/lib/Target/AMDGPU/AMDGPUSetWavePriority.cpp` (main) — pass logic, `amdgpu-set-wave-priority-valu-insts-threshold` default 100 — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUSetWavePriority.cpp
3. LLVM `AMDGPUTargetMachine.cpp` — `-amdgpu-set-wave-priority` hidden, `cl::init(false)`.
4. LLVM `IntrinsicsAMDGPU.td` / `SOPInstructions.td` — `int_amdgcn_s_setprio`, `int_amdgcn_s_sleep`, `S_WAKEUP` gfx8–12, `S_SLEEP_VAR` gfx12/13, `S_SETPRIO_INC_WG` GFX1250/13.
5. LLVM AMDGPUUsage — `"amdgpu-wave-priority-threshold"` attribute; `COMPUTE_PGM_RSRC1.PRIORITY` "Must be 0", CP fills — https://llvm.org/docs/AMDGPUUsage.html
6. radiance-vllm-mxfp4 (codeberg ggz14) @ `22c69cdf` — `autoround-tests/RESULTS.md` "Measured and rejected: `s_setprio(1)` around the WMMA run"; `ar_prefill_opt.h` `SETPRIO` template flag. Leave (WMMA arch).
