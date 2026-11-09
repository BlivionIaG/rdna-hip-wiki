---
title: "VALU constant bus and VOP3 literals vs VGPR occupancy (gfx1030)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/GCNSubtarget.cpp
  - https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/GCNFeatures.td
  - https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp
  - https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/test/MC/AMDGPU/gfx10-constant-bus.s
---

# VALU constant bus and VOP3 literals vs VGPR occupancy (gfx1030)

Lock: **The constant bus is not a PIX / `llvm-calc-occupancy` MaxWaves row.** It is a per-instruction operand rule: how many scalar values (SGPRs, VCC, EXEC, literals) one VALU instruction may read. It touches occupancy only indirectly. When an instruction needs more scalar operands than the rule allows, the compiler copies the extra one into a VGPR, and a loop-invariant copy stays live in a VGPR. GFX10 (gfx1030 included) relaxed the rule compared with GFX9, so fewer of those copies are needed.

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-11-03** Europe/Paris.

Companions: [sgpr-occupancy.md](sgpr-occupancy.md) (SGPRs are not a limiter on GFX10+), [vgpr-occupancy.md](vgpr-occupancy.md), [vgpr-bank-occupancy.md](vgpr-bank-occupancy.md) (operand-read cycles, a different effect), [icache-occupancy.md](icache-occupancy.md) (instruction size), [w4a16.md](../../kernels/quant/w4a16.md) (`0x64006400` dequant).

## Take / Leave

| | |
|---|---|
| **Take** | On gfx1030 a VALU instruction may read **two** scalar values, and SGPRs, VCC, EXEC used as data, and literal constants all count. At most **one** of them may be a 32-bit literal. Inline constants are free and don't count.[^isa-inputs] |
| **Take** | The exceptions cost one slot: add/sub with carry-in, FMAS and CNDMASK read an SGPR (VCC) implicitly, and that read counts toward the two. 64-bit shifts may use only **one** scalar value.[^isa-inputs] LLVM encodes the same rule: `getConstantBusLimit` returns 1 before GFX10, 1 for the 64-bit shift opcodes, and 2 otherwise.[^llvm-cbl] The MC tests show `v_add_f32 v0, s0, s1` assembling for gfx10, while `v_ashrrev_i64 v[0:1], s2, s[0:1]` and `v_div_fmas_f32 v5, s3, 0x123, v3` are rejected.[^mc-cb] |
| **Take** | GFX10 allows a literal in the **VOP3** encoding (`FeatureVOP3Literal` is part of `FeatureGFX10`).[^llvm-gfx10] The machine verifier rejects a VOP3 literal only on targets without that feature.[^llvm-verify] So 3-source ops like `v_and_or_b32`, `v_fma_f32`, `v_perm_b32` can take one literal directly, with no `v_mov` into a VGPR first. |
| **Take (occupancy)** | Keep loop-invariant, wave-uniform constants and scales in SGPRs or as literals, not in VGPRs. SGPR count does not limit waves on gfx1030 ([sgpr-occupancy.md](sgpr-occupancy.md)), and VGPR count does. When an instruction is over the limit, LLVM's VOP3 legalizer moves the extra operand into a VGPR (`legalizeOpWithMove`); with `hasVOP3Literal` it first allows one literal as long as a constant-bus slot remains.[^llvm-legalize] That VGPR copy is the occupancy cost. Inference from the code above, not a measured count. |
| **Take (craft)** | Two **different** non-inline literals cannot share one instruction.[^isa-inputs] The fp16 nibble dequant `(q & 0x000F000F) \| 0x64006400` fits a single `v_and_or_b32` only if one mask is a literal and the other sits in an SGPR (or a VGPR). Neither value is an inline constant: integer inline constants are -16..64, and the float ones are ±0.5, ±1, ±2, ±4 and 1/(2π).[^isa-inline] |
| **Take (craft)** | VOP3 plus a literal is a **96-bit** instruction, and the ISA warns that heavy use "may reduce performance".[^isa-vop3lit] In hot unrolled loops, an SGPR operand (64-bit VOP3, no literal dword) is the smaller encoding; see [icache-occupancy.md](icache-occupancy.md) for why instruction bytes matter. |
| **Take Later (fork call, probe only)** | Dump the ISA of the extras W4 dequant kernels (for example `moe_w4a8_gemm_kernel_v2` at tile 8, repo-reported 121 VGPR, close to the 128 line) and count hoisted `v_mov_b32 vN, 0x…` constants that stay live across the K loop. If any do, moving them to SGPRs could free VGPRs. No compiler on the box today (clang-22 gone), so this is unverified. Routed as a fork note, not a ticket. |
| **Leave** | Do not port GFX9-era advice ("only one SGPR per VALU op, materialise the second in a VGPR") to gfx1030. That was the pre-GFX10 limit of 1.[^llvm-cbl] |
| **Leave** | Not a bank-conflict fix. Constant-bus slots and VGPR operand banks are separate rules; see [vgpr-bank-occupancy.md](vgpr-bank-occupancy.md). |

## 1. The rule in one table

| Operand kind | Counts toward the 2 scalar slots? | Notes |
|---|---|---|
| SGPR (`s`, VCC, EXEC as data) | yes, once per distinct register | the same SGPR used twice counts once[^isa-inputs][^llvm-verify] |
| 32-bit literal | yes; at most one literal per instruction | VOP3 + literal = 96-bit instruction[^isa-inputs][^isa-vop3lit] |
| Inline constant (-16..64, ±0.5, ±1, ±2, ±4, 1/(2π)) | no | free[^isa-inputs][^isa-inline] |
| Implicit VCC read (addc/subb, `v_div_fmas`, `v_cndmask`) | yes | leaves one slot[^isa-inputs][^mc-cb] |
| 64-bit shifts (`v_lshlrev_b64`, `v_lshrrev_b64`, `v_ashrrev_i64`) | limit is 1, not 2 | [^isa-inputs][^llvm-cbl] |

## Sources

[^isa-inputs]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30, §6.2.1 "Instruction Inputs", Limitations list. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (local text `/workspace/rdna2-src/rdna2-isa.txt`, checked 2026-11-03)
[^isa-inline]: Same guide, §6.2 operand table: values 129-192 = int 1..64, 193-208 = int -1..-16, 240-248 = 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2π). https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[^isa-vop3lit]: Same guide, chapter 6 VALU microcode-format text just before §6.2: "VOP3 plus a literal makes a 96-bit instruction and excessive use of this combination may reduce performance." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[^llvm-cbl]: LLVM `GCNSubtarget::getConstantBusLimit` @ `fb51739533ea17d53587c7955f0705bbf84c01a3`: returns 1 if generation < GFX10, 1 for the 64-bit shift opcodes, else 2. https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/GCNSubtarget.cpp#L273-L297
[^llvm-gfx10]: LLVM `def FeatureGFX10` @ `fb51739533ea17d53587c7955f0705bbf84c01a3` lists `FeatureVOP3Literal`. https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/GCNFeatures.td#L1811-L1827
[^llvm-verify]: LLVM `SIInstrInfo::verifyInstruction` @ `fb51739533ea17d53587c7955f0705bbf84c01a3`: counts distinct SGPRs and one literal against `getConstantBusLimit`; "VOP2/VOP3 instruction uses more than one literal"; "VOP3 instruction uses literal" only when `!hasVOP3Literal()`. https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L5820-L5884
[^llvm-legalize]: LLVM `SIInstrInfo::legalizeOperandsVOP3` @ `fb51739533ea17d53587c7955f0705bbf84c01a3`: `LiteralLimit = hasVOP3Literal() ? 1 : 0`; operands over the constant-bus or literal budget go through `legalizeOpWithMove`; comment "We can use one SGPR in each VOP3 instruction prior to GFX10 and two starting from GFX10." https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L7136-L7175
[^mc-cb]: LLVM MC test `gfx10-constant-bus.s` @ `fb51739533ea17d53587c7955f0705bbf84c01a3`. https://github.com/llvm/llvm-project/blob/fb51739533ea17d53587c7955f0705bbf84c01a3/llvm/test/MC/AMDGPU/gfx10-constant-bus.s
