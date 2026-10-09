# Software hazard waitstates (s_nop / s_waitcnt_depctr / vscnt null) vs effective occupancy (gfx1030)

Lock: **Software hazard waitstates are not a PIX / `llvm-calc-occupancy` MaxWaves row, and on gfx1030 the compiler inserts none for ordinary compute code.** GFX10 has hardware data-dependency interlocks (`FeatureGFX10` carries `no-data-dep-hazard`), and gfx1030 (`FeatureISAVersion10_3_0`) carries **none** of the gfx101x hazard/bug features (`vmem-to-scalar-write-hazard`, `smem-to-vector-write-hazard`, `lds-branch-vmem-war-hazard`, `vcmpx-exec-war-hazard`, `vcmpx-permlane-hazard`, `inst-fwd-prefetch-bug`, `nsa-*`, `offset-3f-bug`, `flat-segment-offset-bug`, `lds-misaligned-bug`, `negative-unaligned-scratch-offset-bug`). ISA 70648 §4.5 is one sentence: "Inserting S_NOP is not required to achieve correct operation." Same instruction stream, zero padding. The cost only reappears when you compile the **same source for gfx101x** (Navi10/12/14, BC-250 `gfx1013`), where every VMEM→SGPR rewrite, LDS↔VMEM branch switch and SMEM→VALU write gets an `s_waitcnt_depctr` / `s_waitcnt_vscnt null, 0` park inside the slot the wave already holds.

Sibling of [waitcnt-occupancy.md](waitcnt-occupancy.md) (memory-counter parks), [exec-divergence-occupancy.md](exec-divergence-occupancy.md) (`v_cmpx` vs `s_and_saveexec`), [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md) (DPP/permlane padding), [hard-clause-occupancy.md](hard-clause-occupancy.md).

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-10-09** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [exec-divergence-occupancy.md](exec-divergence-occupancy.md), [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md), [icache-occupancy.md](icache-occupancy.md) (`S_INST_PREFETCH`), [hip-craft.md](hip-craft.md) §7, [codegen-stack.md](codegen-stack.md).

## Take / Leave

| | |
|---|---|
| **Take** | On gfx1030 write HIP without hazard padding. No `s_nop`, `v_nop`, `__builtin_amdgcn_s_nop`, or hand `s_waitcnt_depctr` for VALU→VALU, VALU→DPP, VMEM→SGPR, SMEM→`v_cmp`, or LDS↔VMEM switches. Probe §3: gfx1030 emits **0** of each on a tile loop + divergent tail + SGPR-base rewrite loop. |
| **Take** | `v_cmpx` is the gfx1030 branch idiom (writes EXEC directly, no `s_and_saveexec`). gfx101x avoids it (`vcmpx-exec-war-hazard`) and emits `v_cmp` + `s_and_saveexec_b32`. That is a gfx101x-only extra SALU per branch, not a gfx1030 cost. |
| **Take** | Fatbin rule, backed by the compiler: gfx1030 objects **lack** the gfx101x waits (probe: gfx1013 build of the same no-DOT source = 8× `s_waitcnt_depctr depctr_vm_vsrc(0)` + 4× `s_waitcnt_vscnt null, 0x0`; gfx1030 = 0). So shipping gfx1030 code onto gfx1013 / gfx101x is a **correctness** hazard (VMEM-then-SGPR-write "leads to incorrect execution", SMEM-then-`v_cmp` "page faults", per LLVM feature text), not just a missing-DOT issue. Matches the existing "never ship gfx1030 DOT objects onto gfx1013" scope. |
| **Take** | For BC-250 (`gfx1013`) or Navi10 fatbins, budget these parks into *their* effective occupancy: one `depctr_vm_vsrc(0)` after each VMEM whose address SGPRs are rewritten next (unrolled pointer-chase / per-iter base update), one at LDS↔VMEM branch joins, `vscnt null, 0` before VMEM/LDS after a divergent store. Hoist base updates out of the unrolled body or keep SGPR bases constant per tile. **Later**, gfx101x lane only. |
| **Leave** | Do **not** copy gfx9 / gfx101x inline asm or Navi10 HIP with `s_nop N`, `v_nop`, or `s_waitcnt_depctr` into gfx1030 kernels "for safety". Dead issue slots in the hot loop, never needed for correctness on gfx1030 (ISA §4.5). |
| **Leave** | Do **not** put hazard padding into the theoretical min or `--lds=`. It is per-instruction issue time inside a held slot, like waitcnt parks. |
| **Leave** | Do **not** read gfx1030 `.s` diffs vs gfx1011/gfx1013 builds as "gfx1030 codegen is worse/better". The delta is hazard mitigation + `v_cmpx` policy, everything else matched in the probe. Forcing the six gfx101x features onto gfx1030 (`-Xclang -target-feature -Xclang +…`) adds only the waits (7 depctr + 2 vscnt null); the rest of the stream is byte-identical. |
| **Leave** | Do **not** use `S_VERSION` as a wait state (ISA: "may issue in the same cycle… use S_NOP instead"). Moot on gfx1030 since no NOPs are needed. |
| **Leave** | Do **not** invent hazard-cycle counts. Neither ISA 70648 nor the LLVM feature text gives a stall cost for the gfx101x workarounds. |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. |

## 1. Sources

1. AMD "RDNA 2" ISA, Document 70648 (local `/workspace/rdna2-src/rdna2-isa.txt`): §4.5 "Manually Inserted Wait States (NOPs)" is the single line quoted above. §8.2 puts VMEM data hazards on the developer only via VMCNT/VSCNT (see [waitcnt-occupancy.md](waitcnt-occupancy.md)). `S_WAITCNT_DEPCTR` (SOPP 35) exists on RDNA2.
2. LLVM `llvm/lib/Target/AMDGPU/AMDGPU.td`, `release/22.x` (fetched 2026-10-09): `FeatureGFX10` includes `FeatureNoDataDepHazard` ("Does not need SW waitstates"). `FeatureISAVersion10_1_Common` adds the "gfx101x bugs" block listed in the lock. `FeatureISAVersion10_3_0` = `FeatureISAVersion10_Common` + encodings + Dot1/2/5/6/7/10 + shader-cycles. **No** hazard features. `FeatureISAVersion10_1_3` (gfx1013) = 10_1_Common + GFX10_A encoding, so it inherits every gfx101x hazard and has no Dot2 (`fdot2` needs `dot10-insts`).
3. clang/LLVM 22.1.8 (Debian, `/usr/lib/llvm-22`). Probe sources and `.s` in `/workspace/hazard-probe/`. The ROCm 7.14 amd-llvm pin was not re-checked; feature sets for gfx1030 have been stable across upstream releases, but confirm with `-mattr=help` / a probe on the pin before quoting it as pin-verified.

## 2. What GFX10 still makes the compiler pad (and gfx1030 does not)

| Feature (gfx101x only) | LLVM text | Mitigation on gfx101x | gfx1030 |
|---|---|---|---|
| `vmem-to-scalar-write-hazard` | VMEM followed by scalar write to EXEC/M0/SGPR → incorrect execution | `s_waitcnt_depctr depctr_vm_vsrc(0)` | none |
| `smem-to-vector-write-hazard` | `s_load_dword` followed by `v_cmp` page faults | `s_mov_b32 null, 0` / wait | none |
| `lds-branch-vmem-war-hazard` | switching LDS↔VMEM without `VM_VSRC=0` | `s_waitcnt_vscnt null, 0x0` at the switch | none |
| `vcmpx-exec-war-hazard` | `v_cmpx` WAR on EXEC | avoid `v_cmpx`; `v_cmp` + `s_and_saveexec` | uses `v_cmpx` |
| `vcmpx-permlane-hazard` | `v_permlane` right after `v_cmpx` | dummy `v_mov_b32 vN, vN` | none (ISA §6.10 still states the rule; see [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md)) |
| `inst-fwd-prefetch-bug` | `S_INST_PREFETCH` hangs | compiler never emits it | allowed ([icache-occupancy.md](icache-occupancy.md)) |
| `lds-misaligned-bug` | multi-dword LDS/flat not naturally aligned in WGP mode | split accesses | not present; alignment still matters for `ds_read_b128` selection ([hip-craft.md](hip-craft.md) `alignas(16)`) |

gfx11 has its own set (`valu-trans-use-hazard`, `vcmpx-permlane-hazard` in `FeatureISAVersion11_Common`, partial-forwarding/mask-write hazards). Not gfx1030, and not counted here.

## 3. Probe (2026-10-09, clang 22.1.8, `-O3 --cuda-device-only -nogpulib`)

Kernels (`/workspace/hazard-probe/k.hip`): `tile` = loop with in-loop `s_load` param, global→LDS store, barrier, 8× `fdot2` from LDS, VALU compare on an SGPR-derived value + break; divergent tail with `v_permlanex16` in one arm and an LDS store in the other; final LDS read → global store. `rfl` = 16× unrolled loop where each global load's SGPR base is rewritten right after by `readfirstlane`. `k_nodot.hip` replaces `fdot2` with scalar FMAs so gfx1013 compiles.

| Target | `s_nop` | `s_waitcnt_depctr` | `s_waitcnt_vscnt null, 0x0` | `v_cmpx` |
|---|---|---|---|---|
| gfx1011 (`k.hip`) | 0 | 8 | 4 | 0 |
| gfx1013 (`k_nodot.hip`) | 0 | 8 | 4 | 0 |
| **gfx1030** (`k.hip` and `k_nodot.hip`) | **0** | **0** | **0** | 2 |
| gfx1030 + six gfx101x features forced | 0 | 7 | 2 | 2 |
| gfx1100 (`k.hip`) | 0 | 0 | 0 | 2 |

In `rfl` on gfx1011, 6 of the 8 `depctr_vm_vsrc(0)` parks sit one after each `global_load_dword` whose `s[4:5]` base is rewritten on the next SALU. That is the shape an unrolled per-expert / per-page pointer walk takes. On gfx1030 the same loop has no park.

Not audited: GitHub code search (default-branch index only) found no `s_nop` / `s_waitcnt_depctr` in `opengfx1030/vllm-rdna`. `rdna_extras` itself was not grepped this pass, so no extras claim is made.

## 4. Where it sits in the occupancy composite

Out of the theoretical min, like waitcnt and divergence. On gfx1030 the row is **empty** for compiler-generated code. The only ways to pay it are hand-inserted NOP/depctr asm, or a gfx101x fatbin. If a gfx1030 kernel profile shows `s_nop` / `s_waitcnt_depctr` in the hot loop, it came from inline asm or a copied gfx9/gfx101x helper. Remove it there, don't model it.
