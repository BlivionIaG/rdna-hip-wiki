---
title: "LDS bank conflicts vs occupancy (gfx1030)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
  - https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
  - https://gpuopen.com/learn/occupancy-explained/
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
---

# LDS bank conflicts vs occupancy (gfx1030)

Lock: **bank conflicts do not change `MaxWGsLDS` or the PIX “Limited by LDS” counter — they change LDS *op latency*, which changes how many wave slots you need for latency hiding.** On gfx1030 HIP default (**wave32 + WGP**), LDS is **64 banks × 4 B**; same-bank different-address accesses serialize (ISA §10.4.3: conflict-free indexed ≈ **1 cycle**, worst **64 cycles**). Pad that grows the tile can *lower* concurrent WGs; XOR / swizzle fixes conflicts at **constant** LDS bytes. Distinct from [lds-occupancy.md](lds-occupancy.md) (byte pool → WG count) and [lds-tiles.md](lds-tiles.md) (bank formula + layout recipes).[^isa70648]

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [lds-occupancy.md](lds-occupancy.md), [lds-tiles.md](lds-tiles.md), [occupancy-composite.md](../occupancy/occupancy-composite.md), [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md), [barrier-occupancy.md](../occupancy/barrier-occupancy.md), [hip-craft.md](../hip-craft/hip-craft.md), [architecture.md](../arch-and-memory/architecture.md) § LDS.

## Take / Leave

| | |
|---|---|
| **Take** | Treat bank conflicts as a **latency** term, not a fifth PIX resource. `llvm-calc-occupancy` / PIX LDS limiter only see **bytes**. A conflict-free and a 16-way tile with the same aligned LDS report the **same** `MaxWGsLDS`. [^llvm-calc-occupancy] |
| **Take** | Prefer **XOR / index swizzle** over row padding when you are near an LDS occupancy cliff (`alignTo(LDS) ≤ 32 KB` for 4 WGs/WGP, or ≤ 64 KB for 2). CK: pad costs **12.5–25%** extra LDS; XOR is **zero** storage ([CK LDS bank conflicts](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html); [ROCm blog 2025-07-25](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)). [^rocm-hardware-lds-bank-conflicts-html][^rocm-lds-bank-conflict-readme-html] |
| **Take (craft)** | Rough latency model (**theory**, not measured on V620): an *N*-way same-bank conflict on one indexed op ≈ **N** serial bank beats (bounded by ISA worst **64**). If the inner loop is LDS-stall bound, hiding that stall needs roughly **N×** more eligible waves than the conflict-free case — same VGPR/LDS *slot* budget, worse *effective* latency hiding ([GPUOpen Occupancy explained](https://gpuopen.com/learn/occupancy-explained/)). [^gpuopen-occupancy] |
| **Take** | HIP: same-address broadcast is **not** a conflict; different addresses in one bank **are** ([HIP hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)). WGP bank index: `bank = (byte_addr / 4) % 64` ([lds-tiles.md](lds-tiles.md) §1.2). [^hip-hw-impl] |
| **Leave** | Do not “buy occupancy” by padding into the next LDS ladder step (e.g. 30 KB → 36 KB after pad) — you may drop **4→2** or **2→1** WGs/WGP while only fixing banks ([lds-occupancy.md](lds-occupancy.md) ladder). |
| **Leave** | Do not import CK CDNA phase groups (`ds_*_b128` 8-lane phases, wave64, 32 banks) as gfx1030 cycle counts. Transform (XOR) is portable; phase timing is not ([lds-tiles.md](lds-tiles.md) §2.1). [^isa70648] |
| **Leave** | Do not open a ticket / retip extras for bank-vs-occupancy. Measure with `rocprof` LDS-bank counters on the live object if a tile looks LDS-latency bound. |

## 1. Two different “LDS limits”

| Axis | What it limits | Where it lives | Bank conflicts? |
|---|---|---|---|
| **Bytes / pool** | Concurrent WGs on the WGP | `MaxWGsLDS = floor(LocalMemorySize / alignTo(LDS))` — [lds-occupancy.md](lds-occupancy.md) | **No** — layout-invariant |
| **Banks / cycle** | Cycles per `ds_*` indexed op | ISA §10.4.3 serialization; HIP conflict resolution | **Yes** — layout-dependent [^isa70648] |

PIX `WaveOccupancyLimiters` “Limited by LDS” (GPUOpen Occupancy explained) is the **byte** axis: SPI cannot assign another WG because the LDS *allocation* does not fit. It does **not** increment because waves are stalled on bank serialization. Occupancy as **latency-hiding capacity** (same GPUOpen post) *does* care about bank stalls: more waves help only if there is ALU (or other-wave work) to issue while the conflicting LDS op drains.[^gpuopen-occupancy]

So: fixing banks without growing LDS → same theoretical waves/EU, better *usable* hiding. Growing LDS to fix banks → may **cut** theoretical waves/EU on the byte ladder.

## 2. gfx1030 constants (WGP, wave32)

| Knob | Value | Source |
|---|---|---|
| Banks × width | **64 × 4 B** (256 B/cycle aggregate wording in HIP) | ISA §2.3.1 / §10.1; HIP hardware implementation (RDNA 2/3/4) [^isa70648][^hip-hw-impl] |
| Bank affiliation | Two sets of 32, one per CU-half of the WGP | ISA §2.3.1; [lds-tiles.md](lds-tiles.md) [^isa70648] |
| Conflict-free indexed (wave32) | **as little as 1 cycle** | ISA §10.4.3 [^isa70648] |
| Worst same-bank serialization | **64 cycles** | ISA §10.4.3 [^isa70648] |
| Per-WG LDS cap / WGP pool | 64 KB / 128 KB | ISA §2.3.1; LLVM `getLocalMemorySize` [^isa70648][^llvm-baseinfo] |
| Allocation granule (design) | Prefer **1 KB** (ISA §3.6.6); LLVM occupancy align may use 512 B | [lds-occupancy.md](lds-occupancy.md) §3 [^isa70648] |

HIP also states SIMDs connect to LDS in pairs with a **64-byte bidirectional port** per pair — peak wording vs ISA “32 × 32-bit” is **not reconciled**; use as bounds ([lds-tiles.md](lds-tiles.md) §1.1).

## 3. Conflict depth vs wave-slot need (**theory**)

GPUOpen: occupancy is the SIMD’s capacity to hide stalls by switching waves; one assigned wave ⇒ a stall cannot be hidden at all.

```text
# Theory sketch only — not a V620 measurement
cycles_LDS_op ≈ conflict_ways          # 1 if conflict-free; N if N-way; ≤ 64 ISA bound
waves_to_hide ≳ ceil(cycles_LDS_op / cycles_ALU_between_LDS)
```

Implications for gfx1030 wave32 tiles:

| Access (see [lds-tiles.md](lds-tiles.md) §1.5) | Ways (theory) | Occupancy note |
|---|---|---|
| Sequential `float` / `half2` | 1 (conflict-free) | Byte ladder dominates; do not pad |
| Stride-32 dwords column walk | **16-way** | Needs ~16× more hiding depth *or* XOR/pad; bytes unchanged until you pad |
| Stride-64 dwords | **32-way** | Near worst-case; fix layout before chasing `waves_per_eu` |
| Scalar `half` walk | **2-way** | Pack to `half2` — free occupancy win |

Pinning `amdgpu_waves_per_eu(4,8)` does **not** remove bank serialization; it only budgets VGPR so SPI can *assign* that many waves. If every wave hits the same 16-way LDS pattern in lockstep, you still serialize — measure, then XOR.

## 4. Pad vs XOR at the occupancy cliff (**theory + CK numbers**)

Worked sketch for a 256-thread WG, WGP mode, ISA 1 KB round ([lds-occupancy.md](lds-occupancy.md) ladder):

| Tile LDS (aligned) | Fix | New LDS | `MaxWGsLDS` | Waves/WGP at WG=256 if LDS binds |
|---|---|---|---|---|
| 28 KB | none / XOR | 28 KB | **4** | 32 (= 8/SIMD) |
| 28 KB | +12.5% pad → ~31.5 KB → align 32 KB | 32 KB | **4** | 32 — still OK [^pcisig-pcie4] |
| 30 KB | +25% pad → ~37.5 KB → align 38 KB | 38 KB | **2** | **16** (= 4/SIMD) — **half** the WGs |
| 30 KB | XOR | 30 KB | **4** | 32 — conflicts gone, slots kept |
| 48 KB | any pad into >32 KB region | still 2 | **2** | already at 2-WG ceiling |
| 60 KB | +12.5% pad → ~67.5 KB | **illegal** (>64 KB WG cap) or forced shrink | — | pad can kill the tile |

CK published pad overhead **12.5–25%** and “4-way conflicts can reduce effective LDS bandwidth by **75%**” (CK bank-conflict page — CDNA-oriented; treat % as order-of-magnitude, verify on gfx1030). ROCm blog (2025-07-25): pad is “tricky to tune, consumes extra LDS”; XOR solves read/write conflicts with **no extra buffer**.[^rocm-hardware-lds-bank-conflicts-html]

**Craft rule:** if `alignTo(raw_LDS × 1.25, 1024)` crosses a ladder boundary (32 KB or 64 KB), **do not pad** — XOR or change the access pattern (pack `half2`, avoid stride-32/64 column walks).[^rocm-gpu-arch-specs][^llvm-baseinfo]

## 5. Relation to the composite fold

[occupancy-composite.md](../occupancy/occupancy-composite.md):

```text
waves/EU = min(VGPR, SGPR≡16, WG+LDS+barrier)
```

Bank conflicts are **outside** that min. They show up as:

1. Longer `ds_*` latency in RGP / `rocprof` (instruction timing), not as PIX “Limited by LDS”.
2. A false urge to raise `amdgpu_waves_per_eu` / lower VGPR — useful only if slots were empty *and* ALU exists to fill the stall; useless if the layout still 16-ways every wave.

Order of work when LDS looks “slow”:

1. Dump aligned LDS bytes → [lds-occupancy.md](lds-occupancy.md) / [occupancy-dump.md](../occupancy/occupancy-dump.md).
2. Check bank map / pad vs XOR → this page + [lds-tiles.md](lds-tiles.md).
3. Only then retune `waves_per_eu` / `__launch_bounds__` → [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md).

## 6. Measured vs theory

| Claim | Status |
|---|---|
| 64 banks × 4 B; serialize different addresses same bank | **ISA / HIP** [^isa70648] |
| Conflict-free ≈ 1 cycle (wave32); worst 64 cycles | **ISA §10.4.3** [^isa70648] |
| *N*-way ≈ *N* serial beats | **Theory** (consistent with ISA bound; not a V620 bench) |
| Pad 12.5–25% LDS; XOR zero extra | **CK docs / ROCm blog** (CDNA examples; overhead % portable) |
| Pad crossing 32 KB drops 4→2 WGs/WGP | **Theory** from LLVM/ISA byte math ([lds-occupancy.md](lds-occupancy.md)) |
| gfx1030 `ds_read_b128` phase groups / exact cycles | **Unknown** — do not invent |

## Sources

1. AMD RDNA 2 ISA 70648 — https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (§2.3.1, §3.6.6, §10.1, §10.4.3)
2. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/
3. HIP Hardware implementation (LDS banks / conflict resolution) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
4. CK Tile — Understanding AMD GPU LDS and Bank Conflicts — https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
5. ROCm blog — Avoiding LDS Bank Conflicts (2025-07-25) — https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
6. Wiki siblings — [lds-occupancy.md](lds-occupancy.md), [lds-tiles.md](lds-tiles.md), [occupancy-composite.md](../occupancy/occupancy-composite.md)

Footnotes:

[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)
[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-10-11)
[^rocm-hardware-lds-bank-conflicts-html]: Understanding AMD GPU LDS and Bank Conflicts — Composable Kernel 1.3.0 Documentation. https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html (checked 2026-10-11)
[^rocm-lds-bank-conflict-readme-html]: Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile Framework — ROCm Blogs. https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html (checked 2026-10-11)
[^gpuopen-occupancy]: GPUOpen, "Occupancy explained". https://gpuopen.com/learn/occupancy-explained/ (checked 2026-10-11)
[^hip-hw-impl]: ROCm HIP docs, "Hardware implementation". https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (checked 2026-10-11)
[^llvm-baseinfo]: LLVM AMDGPUBaseInfo.cpp (IsaInfo), main snapshot eb61ef62df4f. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp (checked 2026-10-11)
[^pcisig-pcie4]: PCI-SIG FAQ, PCI Express 4.0 (16 GT/s per lane). https://pcisig.com/faq?field_category_value%5B%5D=pci_express_4.0 (checked 2026-10-11)
[^rocm-gpu-arch-specs]: ROCm docs 7.2.4, "GPU architecture specifications" (Radeon PRO V620 row: gfx1030, 32 GiB, 72 CU, wave 32/64, LDS 128 KiB, Infinity Cache 128 MiB, L2 4 MiB, GL1 128 KiB, L0 vector 16 KiB, L0 scalar 16 KiB, L0 instruction 32 KiB, VGPR file 512 KiB). https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11)
