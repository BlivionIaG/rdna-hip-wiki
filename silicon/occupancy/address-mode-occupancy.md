---
title: "VMEM address mode (64-bit vaddr vs global SADDR vs buffer V#) vs effective occupancy (gfx1030)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/FLATInstructions.td
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/BUFInstructions.td
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp
  - https://godbolt.org/
---

# VMEM address mode vs effective occupancy (gfx1030)

Lock: **how a load names its address changes VGPR count and VALU work, not a PIX MaxWaves row.** On gfx1030 a GLOBAL load either takes a full 64-bit address in a VGPR pair (`SADDR` off) or a 64-bit SGPR base plus a **32-bit unsigned per-lane VGPR offset** (`SADDR` on).[^isa-flat-fields][^isa-legal-addr] A MUBUF load takes a 128-bit buffer descriptor (V#) in SGPRs, one 32-bit VGPR offset (`offen`), and a 32-bit SGPR `soffset`.[^isa-mubuf-fields][^llvm-buf-offen] So a wave holding *k* addresses live pays 2*k* VGPRs in 64-bit mode, *k* in SADDR mode, and 1 shared VGPR in buffer mode when the per-load step is uniform. Those VGPRs feed the term on [vgpr-occupancy.md](vgpr-occupancy.md); they only change waves/SIMD when they push `.vgpr_count` across a 16-VGPR allocation rung.[^llvm-flat-vaddr][^probe-addr]

Sibling of [vgpr-occupancy.md](vgpr-occupancy.md) (the term it moves), the VMEM-to-LDS page from wiki PR #29 (`lds-dma-occupancy.md`, the other in-flight-VGPR lever), the SALU page from wiki PR #22 (buffer mode moves address math to SALU), and [cache-policy.md](../arch-and-memory/cache-policy.md) §3.4 (V# bits).

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-10-26** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [hip-craft.md](../hip-craft/hip-craft.md), [hipkittens-patterns.md](../external-projects/hipkittens-patterns.md) ("cheap address generation" Take).

## Take / Leave

| | |
|---|---|
| **Take** | Write tile loads as **uniform base pointer + 32-bit unsigned element offset** (`(const char*)base + (unsigned)off`) so clang can pick `global_load … vOFF, s[base]`. Probe: 16 dword gathers drop from 23 VGPRs (clang 22) or 40 (clang 19) to 17, and the 32 64-bit add halves become 15 single 32-bit adds.[^probe-addr][^isa-legal-addr] |
| **Take** (Later) | For strided K/V or weight tiles where every load steps by a uniform amount, a **buffer resource** keeps one `voffset` VGPR for all loads and puts the step in `soffset` (SALU `s_mul`/`s_lshl`). Probe: 0 VALU address instructions in the load body for both widths.[^probe-addr][^llvm-buf-offen] Costs: 4 SGPRs for the V#, SALU work per distinct `soffset`, and you own the descriptor bits ([cache-policy.md](../arch-and-memory/cache-policy.md) §3.4). |
| **Take** | Fold small constant steps into the **immediate offset**: 12 bits on both paths on GFX10 (GLOBAL/SCRATCH signed, MUBUF unsigned, max 4095). Steps past that need an add per load.[^isa-flat-fields][^isa-mubuf-fields][^llvm-flatoff][^llvm-mubufoff] |
| **Take** | Expect **no VGPR difference for `dwordx4` loads** when the address dies at issue: LLVM reuses the address pair as the low half of the 4-dword destination (`global_load_dwordx4 v[1:4], v[1:2], off`). All three dwordx4 variants in the probe sit at 33. The saving shows up with dword-or-narrower loads and with addresses that stay live across a compute phase.[^probe-addr] |
| **Must** | `SADDR` needs the 32-bit offset to be provably unsigned and in range. A signed `int` index or a 64-bit `long` stride keeps clang on the 64-bit `vaddr` path (probe `kflat64` vs `ksaddr`). Check the `.s` for `, off` vs `, s[..]` before trusting a rewrite.[^probe-addr] |
| **Must** | Re-probe on the **ROCm 7.14.0** amd-clang before any extras change. The probe used upstream clang 19.1.0 and 22.1.0 on Compiler Explorer, and the two already differ (in the dword case clang 19 builds each 64-bit address with its own `v_mad_u64_u32`, 23 in the body, and keeps the pairs live, so it lands at 40 VGPRs to clang 22's 23).[^probe-addr] |
| **Leave** | Do **not** treat address mode as a waves/SIMD limiter or add it to `llvm-calc-occupancy`. In the probe every variant stays at or under 64 VGPRs, which is the 16-wave rung; no rung was crossed. It matters only for a kernel already within one granule of a rung (for example 64 vs 80 is 16 vs 12 waves; table on [vgpr-occupancy.md](vgpr-occupancy.md)).[^probe-addr] |
| **Leave** | Do **not** port buffer-descriptor tricks that assume gfx11/gfx12 fields or the gfx950 LDS-DMA width. Keep this page to the GFX10 encodings cited here. |
| **Leave** | Do **not** invent latency or bandwidth differences between the three paths. ISA 70648 and the LLVM sources cited here give no cycle cost per address mode. |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. Which extras kernels use which mode was not audited in this pass. |

## 1. What the ISA allows (GFX10.3)

| Path | Address operands | VGPRs per live address | Immediate |
|---|---|---|---|
| GLOBAL, `SADDR` = NULL | 64-bit address in `ADDR`, `ADDR+1` | 2 | 12-bit signed |
| GLOBAL, `SADDR` set | SGPR pair base + 32-bit unsigned VGPR offset | 1 | 12-bit signed |
| MUBUF `offen` | V# in 4 SGPRs + 32-bit VGPR offset + SGPR/inline `soffset` | 1 (shared if only `soffset` varies) | 12-bit unsigned |

Sources: ISA Table 51 and §9.3.1 for GLOBAL, the MUBUF field table for buffer.[^isa-flat-fields][^isa-legal-addr][^isa-mubuf-fields] LLVM encodes the same split: the GLOBAL pseudo picks `VGPROp_32` for `vaddr` when `EnableSaddr` is set and `VGPROp_64` otherwise, and the `_OFFEN` buffer pattern takes `VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset`.[^llvm-flat-vaddr][^llvm-buf-offen]

## 2. Probe (2026-10-26, upstream clang 19.1.0 and 22.1.0, `--target=amdgcn-amd-amdhsa -mcpu=gfx1030 -O3 -nogpulib`)

`k.c` (8 × `dwordx4`) and `k32.c` (16 × `dword`): each lane gathers one row index, then issues N independent loads at a uniform stride from that row and sums them. Three kernels: `kflat64` (64-bit `long` arithmetic on a global pointer), `ksaddr` (uniform base + `unsigned` byte offset), `kbuf` (`__builtin_amdgcn_raw_buffer_load_b32/b128` with the stride in `soffset`).[^probe-addr]

| Kernel | Width × N | `.amdhsa_next_free_vgpr` clang 19 / 22 | 64-bit address adds (`v_add_co*`) clang 22 | 32-bit VALU adds with SGPR | SALU `s_mul`/`s_lshl` |
|---|---|---|---|---|---|
| `kflat64` | dword × 16 | **40** / **23** | 32 | 0 | 1 |
| `ksaddr` | dword × 16 | 17 / 17 | 0 | 15 | 1 |
| `kbuf` | dword × 16 | 17 / 17 | 0 | 0 | 15 |
| `kflat64` | dwordx4 × 8 | 33 / 33 | 16 | 0 | 1 |
| `ksaddr` | dwordx4 × 8 | 33 / 33 | 0 | 7 | 1 |
| `kbuf` | dwordx4 × 8 | 33 / 33 | 0 | 0 | 7 |

Counts are from the clang 22 `.s` bodies except the clang 19 VGPR column. Every row rounds to ≤ 64, so all six sit at 16 waves/SIMD. The probe measures register and instruction counts only, not time.

## 3. Where it sits in the occupancy composite

Out of the theoretical min, like the VMEM-to-LDS page (wiki PR #29): it is an input that lowers `vgpr_count` and VALU issue, not a separate limiter. VALU address math competes for the same issue slot as `fdot2`/`sdot4` (dot-issue page, wiki PR #25); buffer mode moves that work to the scalar pipe instead (SALU page, wiki PR #22).

## Sources

1. AMD "RDNA 2" ISA Reference Guide, doc 70648 (local text `/workspace/rdna2-src/rdna2-isa-amd.txt`): MUBUF field table (p.57), Table 51 FLAT/GLOBAL/SCRATCH fields (pp.83–84), §9.3.1 legal addressing combinations (p.86).
2. LLVM `FLATInstructions.td`, `BUFInstructions.td`, `AMDGPU.td`, `AMDGPUBaseInfo.cpp`, `SIInstrInfo.cpp` at eb61ef62df4fd5151c4c379302f636ac430bc32d.
3. Compiler Explorer probe, compilers `cclang1910` and `cclang2210` with an amdgcn target; sources and `.s` saved at `/workspace/addr-probe/` on the box.

[^isa-flat-fields]: AMD "RDNA 2" ISA, doc 70648, Table 51 (pp.83–84): `ADDR` "For 64-bit addresses, ADDR has the LSBs, and ADDR+1 has the MSBs. As an offset a single VGPR has a 32 bit unsigned offset"; `OFFSET` "Scratch, Global: 12-bit signed byte offset"; `SADDR` "Global: Use the SGPR to provide a base address; the VGPR provides a 32-bit offset per lane." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-26)
[^isa-legal-addr]: Same guide, §9.3.1 (p.86): Global "a. VGPR (32 or 64 bit) supplies the address. Indicated by: SADDR == NULL. b. SGPR (64 bit) supplies an address, and a VGPR (32 bit) supplies an offset"; "Every mode above can also add the 'instruction immediate offset' to the address." https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-26)
[^isa-mubuf-fields]: Same guide, MUBUF field table (p.57): `SOFFSET` "SGPR to supply unsigned byte offset. SGPR, M0, or inline constant"; `OFFSET` 12 bits "Unsigned byte offset". https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-26)
[^llvm-flat-vaddr]: LLVM `FLATInstructions.td` at eb61ef62df4fd5151c4c379302f636ac430bc32d, `FLAT_Load_Pseudo`: `RegisterOperand VaddrOp = !if(EnableSaddr, VGPROp_32, VGPROp_64)`. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/FLATInstructions.td#L229-L250 (checked 2026-10-26)
[^llvm-buf-offen]: LLVM `BUFInstructions.td` at eb61ef62df4fd5151c4c379302f636ac430bc32d: `_OFFEN` load pattern `(… VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset, timm:$offset …)`. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/BUFInstructions.td#L1467 (checked 2026-10-26)
[^llvm-flatoff]: LLVM `AMDGPU.td` at eb61ef62df4fd5151c4c379302f636ac430bc32d, `def FeatureGFX10` lists `FeatureFlatOffsetBits12` (L1833), and `AMDGPUBaseInfo.cpp` `getNumFlatOffsetBits` returns 12 for that feature. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L1804-L1838 ; https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L3331-L3337 (checked 2026-10-26)
[^llvm-mubufoff]: LLVM `SIInstrInfo.cpp` at eb61ef62df4fd5151c4c379302f636ac430bc32d, `getMaxMUBUFImmOffset`: 12 offset bits before GFX12, returns `(1 << 12) - 1`. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L10630-L10635 (checked 2026-10-26)
[^probe-addr]: Compiler Explorer API, compilers `cclang1910` (clang 19.1.0) and `cclang2210` (clang 22.1.0), flags `--target=amdgcn-amd-amdhsa -mcpu=gfx1030 -O3 -nogpulib`; sources `k.c`, `k32.c` and outputs `k_*.s`, `k32_*.s` saved at box path `/workspace/addr-probe/` (not a public URL; re-run to reproduce). Upstream LLVM, not the ROCm 7.14.0 pin. https://godbolt.org/ (run 2026-10-26)
