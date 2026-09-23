# Vector L0 / GL1 vs occupancy (gfx1030)

Lock: **Vector L0 (TCP) and GL1 are not PIX / `llvm-calc-occupancy` MaxWaves limiters.** They are **shared read-path capacity** (L0 ~16 KB/CU; GL1 ~128 KB/SA). Thrash raises *memory-wait* idle and can **lower effective latency hiding** even when theoretical waves/EU look healthy. Completes the occupancy-adjacent set next to [icache-occupancy.md](icache-occupancy.md) (fetch stalls) and [infinity-cache.md](infinity-cache.md) (GPU-wide MALL). Not a fifth PIX row.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:** **L0** = vector / TCP data cache per CU (ROCm “L0 Vector Cache”). **GL1** = Graphics L1 / “GL1 Data Cache per SA” (ROCm “Graphics L1”). Do **not** confuse L0 with I$ ([icache-occupancy.md](icache-occupancy.md)) or with Infinity Cache / IC ([infinity-cache.md](infinity-cache.md)).

Date: **2026-09-22** Europe/Paris.

Companions: [architecture.md](architecture.md) §4.3, [cache-policy.md](cache-policy.md) §1 / §3, [infinity-cache.md](infinity-cache.md), [l2-occupancy.md](l2-occupancy.md), [occupancy-composite.md](occupancy-composite.md), [icache-occupancy.md](icache-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [fa-occupancy.md](fa-occupancy.md), [hip-craft.md](hip-craft.md) §1.3 / §6.

## Take / Leave

| | |
|---|---|
| **Take** | Treat L0/GL1 as **working-set / coalescing / wave-count** pressure on the *vector* read path: 16 KB/CU → GL1 128 KB/SA → L2 → IC → GDDR6. Extra waves help only while they hide real miss latency **without** multiplying distinct cold lines through the same tiny L0/GL1. |
| **Take** | Skinny decode GEMV: stage **A** (activation) in LDS/VGPR; stream **B** (weights) with coalesced wave32 loads; use `__launch_bounds__` / `amdgpu_waves_per_eu` to **cap** occupancy when the weight stream is already memory-bound ([cache-policy.md](cache-policy.md) §0). |
| **Take** | Coalesce to **128 B** lines (wave32 × 4 B). Structure-of-arrays, consecutive `threadIdx.x`. That is the native happy path for L0/GL1/L2 ([cache-policy.md](cache-policy.md) §3.2; HIP Wave32 × 4 B). |
| **Take** | When measured occupancy is high but the kernel is memory-wait heavy, check **TCP / GL1 hit rates** (rocprofiler-compute Memory Chart) *before* chasing another VGPR wave — same spirit as I$ SQC counters on the fetch side. |
| **Leave** | Do **not** put L0 / GL1 / TCP capacity into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters` (VGPR / LDS / Thread Group Size / Barriers only). |
| **Leave** | Do **not** invent L0/GL1 miss cycle counts, associativity, hit rates, or TB/s — ISA / GPUOpen occupancy / `kfd_crat` do not publish miss penalties. Profile. |
| **Leave** | Do **not** invent persist / bypass bits to “keep decode in L0.” HIP default loads already hit-LRU; GLC/DLC are **coherence / fence** bypass, not an occupancy lever ([cache-policy.md](cache-policy.md)). |
| **Leave** | Do not open UNC / retip extras for this page. |

## 1. What gfx1030 L0 / GL1 actually are

| Property | L0 (vector / TCP) | GL1 (Graphics L1) | Source |
|---|---|---|---|
| Size | **16 KiB per CU** | **128 KiB per SA** | ROCm gpu-arch-specs V620; `kfd_crat` Sienna “TCP L1 Cache per CU” / “GL1 Data Cache per SA” |
| Line size | **128 B** | **128 B** | RDNA deck; architecture §4.3; HIP Wave32×4 B note |
| Sharing | **1 CU** (`num_cu_shared = 1`) | **~10 CU** on full Navi 21 SA (`num_cu_shared = 10`) | `kfd_crat` Sienna Cichlid |
| Role on read path | First vector filter after VMEM | SA-shared filter before L2 | ISA §2.4; HIP Hardware implementation |
| Write policy | Write-through to L2 | Write-through / stores bypass L1 (ISA) | HIP + ISA §8.1.10 |
| Coherence | Software; CUs do **not** snoop each other’s L0 | Same hierarchy; L2 is coherence point | HIP Hardware implementation |

Read path (ISA §2.4, restated):

```text
VGPR / LDS → L0 (per CU, TCP) → GL1 (per SA) → L2 → Infinity Cache → GDDR6
```

HIP: vector L0 is write-through, “typical size of 16 KB per CU”; L2 is the coherence point; atomics execute at L2.

ROCm gpu-arch-specs (docs-7.2.4 opened this pass) **Radeon PRO V620** row:

| Field | V620 |
|---|---|
| LLVM target | gfx1030 |
| CU | 72 |
| Infinity Cache | 128 MiB |
| L2 | 4 MiB |
| Graphics L1 | 128 KiB |
| L0 Vector | 16 KiB |
| L0 Scalar | 16 KiB |
| L0 Instruction | 32 KiB |

Source: https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html

**SA count on V620:** `kfd_crat` Sienna packs GL1 at **10 CU/SA** on the full die. V620 is a **72 CU** harvest — do **not** invent `72/10` as a live SA map. Prefer topology / CRAT dump on the box; occupancy craft only needs “GL1 is shared across the CUs of one SA,” not a fixed divisor.

**Associativity:** **unknown** (same lock as [architecture.md](architecture.md) / [cache-policy.md](cache-policy.md)). Do not invent 16-way.

## 2. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Scratch / private | **No** | Latency; rare ROCr `waves_per_cu` cut |
| I$ (SQC) | **No** | Instruction-fetch idle ([icache-occupancy.md](icache-occupancy.md)) |
| **L0 / GL1** | **No** | Memory-wait / cache thrash; measured occupancy can look “full” while effective latency hiding collapses |
| L2 | **No** | GPU-wide mid-cache thrash ([l2-occupancy.md](l2-occupancy.md)) |
| Infinity Cache | **No** | GPU-wide thrash ([infinity-cache.md](infinity-cache.md)) |

GPUOpen Occupancy explained: compile-time reservation set is **VGPR, SGPR (fixed on RDNA), LDS, threadgroup size, barriers**. Cache hierarchy is the **other** side of the story — “Peak occupancy does not always mean peak performance”: more waves can thrash scarce shared caches outside the WGP booking model. Occupancy is capacity to hide memory latency by switching waves; L0/GL1 thrash **increases** that latency and can make extra waves harmful.

Contrast with I$: I$ misses stall **instruction fetch** (sibling waves help only if their PCs are warm). L0/GL1 misses stall **vector memory**; sibling waves help only if they have ready ALU or hits in the **same** small filter — cold multi-wave weight streams often just multiply fills.

## 3. How misses hurt effective occupancy (no invented cycles)

Mental model only — **no published gfx1030 L0/GL1 miss-penalty table**:

1. Wave issues a vector load → L0 tag check.
2. L0 miss → GL1; GL1 miss → L2 → IC → GDDR6.
3. Wave waits on `s_waitcnt` / VMEM dependency → scheduler switches to another resident wave (**this** is why occupancy exists).
4. If every resident wave is also missing distinct lines through the same 16 KB L0 (or the SA’s 128 KB GL1), switching finds **no** useful work → SIMD idle / memory-bound collapse.
5. PIX / `llvm-calc-occupancy` still report the VGPR∩LDS reservation — they never saw the thrash.

GLC/DLC note ([cache-policy.md](cache-policy.md)): GLC=1 forces L0 miss (no L0 persistence across waves); on gfx10 DLC bypasses L1. Default HIP WGP workgroup-scope acquires use GLC. That is **correctness**, not a way to “free L0 for occupancy.” Do not sprinkle GLC to tune MaxWaves.

## 4. Profiling (cite counters; do not invent hit rates)

| Signal | How | Use |
|---|---|---|
| TCP / GL0 hit rate | rocprofiler-compute Memory Chart / GL0 (TCP) panel | First-level vector filter health. Published RDNA3.5 formula shape: `100 * (TCP_REQ − TCP_REQ_MISS) / TCP_REQ` when denom ≠ 0 — confirm gfx1030 analysis config exposes the same `TCP_*` names before treating as gospel |
| GL1 / GL1C hit rate | Same tool, GL1 panel | SA-shared filter. Published shape: `100 * (GL1C_REQ − GL1C_REQ_MISS) / GL1C_REQ` |
| Wave memory wait / Idle | RGP instruction timing; rocprofiler-SDK thread trace | Correlate miss spikes with wait; green “hidden” latency only counts if other waves ran ALU |
| Working-set estimate | Bytes touched per CU / SA this step vs 16 KB / 128 KB | Rough craft check — not a hardware counter |

Do **not** treat a non-100% TCP hit rate alone as a dest flip. Streaming W4 that misses L0 by design can still be correct if IC/L2 absorb reuse; thrash matters when **extra waves** turn a tolerable miss stream into SA-wide thrash.

## 5. Practical shapes (gfx1030 extras craft)

| Shape | Care about L0/GL1? | Why |
|---|---|---|
| Skinny decode GEMV, A in LDS, B streamed coalesced, modest waves/EU | **Yes — cap waves** | Classic GPUOpen memory-bound thrash: max waves multiply cold weight lines through 16 KB L0 |
| Activation `x` / small scales already in LDS or VGPR | **Usually ignore L0** | Fits; limiter is DOT/VALU or weight path |
| FA / attention with LDS tiles holding QK tiles | **Medium** | Tile should live in LDS; global gather still hits L0/GL1 — FA resource pin stays on [fa-occupancy.md](fa-occupancy.md) (not restated) |
| Fat unsharded weight GEMM that already blows IC | **L0/GL1 secondary** | Misses fall through to GDDR6; nontemporal + fit IC first ([infinity-cache.md](infinity-cache.md)) |
| Concurrent HIP streams on one V620 | **Yes at GL1/IC** | Two streams share SA GL1 and GPU IC — occupancy on *each* stream stacks thrash |

Rule of thumb (capacity, not a measured hit rate): one CU’s L0 is **16 KB** — a 128×128 FP16 tile is already **32 KB**, so L0 is a **streaming filter**, not a GEMM panel store ([architecture.md](architecture.md) § craft). Prefer LDS for reuse tiles; keep L0 for coalesced streaming.

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](architecture.md) §4.3 | L0 16 KB/CU, GL1 128 KB/SA, 128 B lines | Occupancy framing + explicit **not a PIX row** |
| [cache-policy.md](cache-policy.md) | Topology, GLC/DLC, thrash via over-occupy | Ties miss → effective latency hiding; Take/Leave for decode GEMV wave caps |
| [infinity-cache.md](infinity-cache.md) | GPU-wide MALL thrash from too many waves | Separates **CU/SA filters** (L0/GL1) from **128 MB IC** |
| [l2-occupancy.md](l2-occupancy.md) | GPU-wide mid L2 thrash | Separates **4 MiB L2** from L0/GL1 and from IC |
| [icache-occupancy.md](icache-occupancy.md) | I$ fetch stalls out of min | Sibling for **vector data** path vs instruction path |
| [occupancy-composite.md](occupancy-composite.md) | PIX four + scratch/I$ out | L0/GL1 as another out-of-min effective-occupancy sibling |

## Sources (opened this pass)

1. GPUOpen Occupancy explained — reservation limiters VGPR/LDS/threadgroup/barriers; “Peak occupancy does not always mean peak performance” (cache thrash from extra waves); latency-hiding definition — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
2. HIP Hardware implementation — vector L1/L0 write-through, ~16 KB/CU, L2 coherence point, software coherence, RDNA WGP / Wave32 / 128 B lines — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
3. ROCm GPU hardware specifications docs-7.2.4 — V620 gfx1030: Graphics L1 128 KiB, L0 Vector 16, L0 Scalar 16, L0 Instruction 32, L2 4 MiB, IC 128 MiB, 72 CU — https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
4. `kfd_crat` Sienna Cichlid patch — TCP L1 16 KB/CU (`num_cu_shared=1`); GL1 128 KB/SA (`num_cu_shared=10`); L2 4096 KB; L3 128×1024 KB — https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html
5. RDNA 2 ISA Reference Guide 70648 — §2.4 read path L2→L1→L0; §8.1 GLC L0 persistence / miss; §8.1.10 stores Miss-Evict L0, L1 bypassed on stores — https://gpuopen.com/news/rdna2-isa-available/
6. GPUOpen RDNA Architecture presentation — L0/L1/L2 table, 128 B lines (sizes cross-checked with gpu-arch-specs / kfd) — https://gpuopen.com/download/RDNA_Architecture_public.pdf
7. rocprofiler-compute RDNA GL0 (TCP) / GL1 conceptual pages — TCP = vector L0; hit-rate formula shapes; **documented panels are RDNA3.5/gfx115x-oriented — verify gfx1030 config before locking counter names** — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/gl0-cache.html and `…/gl1-cache.html`
8. Wiki companions already locking topology / policy: [architecture.md](architecture.md), [cache-policy.md](cache-policy.md), [infinity-cache.md](infinity-cache.md), [icache-occupancy.md](icache-occupancy.md), [occupancy-composite.md](occupancy-composite.md)
