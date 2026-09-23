# L2 vs effective occupancy (gfx1030)

Lock: **L2 is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** It is the **GPU-wide mid-level** data cache (**4 MiB** on Navi 21 / V620), above GL1/TCP and **below Infinity Cache / MALL**. Thrash raises *memory-wait* / longer miss latency → weaker effective latency hiding even when VGPR∩LDS occupancy looks full. Same occupancy-adjacent class as [l0-gl1-occupancy.md](l0-gl1-occupancy.md) and [icache-occupancy.md](icache-occupancy.md). Not a fifth PIX row.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:** this page is **L2** (GPU-wide, ~4 MiB, 128 B lines). Do **not** confuse with **Infinity Cache / IC / MALL / L3** ([infinity-cache.md](infinity-cache.md), 128 MiB, 64 B lines) or with **L0/GL1** ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)). Rocprofiler RDNA docs often call L2 **GL2 / GL2C** (vs Instinct **TCC**).

Date: **2026-09-23** Europe/Paris.

Companions: [architecture.md](architecture.md) §4.3 / §4.4, [cache-policy.md](cache-policy.md) §1 / §3, [infinity-cache.md](infinity-cache.md), [occupancy-composite.md](occupancy-composite.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [icache-occupancy.md](icache-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [fa-occupancy.md](fa-occupancy.md), [hip-craft.md](hip-craft.md) §1.3 / §6.

## Take / Leave

| | |
|---|---|
| **Take** | Treat L2 as **GPU-wide reuse / coalescing / wave-count** pressure on the mid data path: L0 → GL1 → **L2 (4 MiB)** → IC (128 MiB) → GDDR6. Extra waves help only while they hide real miss latency **without** multiplying distinct cold lines through the same 4 MiB. |
| **Take** | Coalesce / tile so the working set fits reuse windows; stage tiles in **LDS** when **A** is reused; stream **B** coalesced (128 B lines). Cap occupancy with `__launch_bounds__` / `amdgpu_waves_per_eu` when the stream is already memory-bound ([cache-policy.md](cache-policy.md) §0). |
| **Take** | When measured occupancy is high but memory-wait dominates, check **L2 hit / miss** (rocprofiler-compute Memory Chart / GL2 panel when exposed) **before** chasing another VGPR wave — same spirit as TCP/GL1 and SQC I$ counters. |
| **Take** | Distinguish **L2 thrash** (4 MiB mid filter) from **IC thrash** (128 MiB last-level on V620). A layer shard that misses L2 but fits IC is still an IC win ([infinity-cache.md](infinity-cache.md)); over-filling waves can thrash **both**. |
| **Leave** | Do **not** put L2 capacity into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters` (VGPR / LDS / Thread Group Size / Barriers only). LLVM `computeOccupancy` mins WG+LDS, VGPR, SGPR — no cache size term. |
| **Leave** | Do **not** invent L2 miss-cycle tables, associativity, channel count, or V620-specific GB/s. Channel / interleave / assoc = **unknown** ([architecture.md](architecture.md) §4.4). Profile. |
| **Leave** | Do **not** confuse L2 with Infinity Cache, or copy rocprofiler gfx115x “GL2 = final on-chip cache before DRAM” wording onto V620 — on RDNA2, **IC sits after L2**. |
| **Leave** | Do not invent a gfx1030-locked rocprofiler counter list; cite documented GL2C / TCC **shapes** only with a verify-on-box note. Do not open UNC / retip extras for this page. |

## 1. What gfx1030 L2 actually is

| Property | Value | Source |
|---|---|---|
| Size (Navi 21 / V620) | **4 MiB** (4096 KB) GPU-wide | ROCm gpu-arch-specs V620 “L2 Cache (MiB) = 4”; `kfd_crat` Sienna Cichlid L2 Data Cache 4096 KB |
| Line size | **128 B** | `kfd_crat`; RDNA deck; architecture §4.4 (IC is **64 B**) |
| Sharing | **Whole GPU** (one monolithic L2 on RDNA2 discrete) | architecture § (no multi-GCD); ISA §2.4 “multiple channels of L2” |
| Role on read path | Mid filter after GL1; before Infinity Cache | ISA §2.4; architecture §4.3–4.5 |
| Coherence | **L2 is the coherence point**; atomics execute at L2 | HIP Hardware implementation |
| Hit-on-miss | **Yes** (HIP) | HIP Hardware implementation; contrast I$/K$ not hit-on-miss |
| Write path | L0/GL1 write-through → L2 | HIP + ISA §8.1 |
| Associativity | **unknown** | Same lock as architecture / cache-policy |
| Channel count / interleave | **unknown** on Navi 21 (HIP “32 channels / 256 B” is **CDNA** text — do not reuse) | architecture §4.4 |

Read path (ISA §2.4 + RDNA2 IC, restated):

```text
VGPR / LDS → L0 (per CU, TCP) → GL1 (per SA) → L2 (4 MiB, GPU) → Infinity Cache (128 MiB) → GDDR6
```

ROCm gpu-arch-specs (docs-7.2.4) **Radeon PRO V620** row (L2 / IC excerpt):

| Field | V620 |
|---|---|
| LLVM target | gfx1030 |
| CU | 72 |
| Infinity Cache | 128 MiB |
| L2 | 4 MiB |
| Graphics L1 | 128 KiB |
| L0 Vector | 16 KiB |

Source: https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html

**vs Infinity Cache:** a decode/weight working set that **misses 4 MiB L2** but **fits 128 MiB IC** is still an IC reuse win — not “L2 failed so GDDR6.” L2 thrash and IC thrash are **different** capacity cliffs ([infinity-cache.md](infinity-cache.md); namu/architecture: L2 stayed 2–4 MB while SE/WGP grew).

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
| L0 / GL1 | **No** | CU/SA vector-filter thrash ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)) |
| **L2** | **No** | GPU-wide mid-cache thrash; measured occupancy can look “full” while miss latency grows |
| Infinity Cache | **No** | GPU-wide last-level thrash ([infinity-cache.md](infinity-cache.md)) |

GPUOpen Occupancy explained: compile-time reservation set is **VGPR, SGPR (fixed on RDNA), LDS, threadgroup size, barriers**. Cache hierarchy is the **other** side — “Peak occupancy does not always mean peak performance”: more waves can thrash scarce shared caches **outside** the WGP booking model. Occupancy is capacity to hide memory latency by switching waves; L2 thrash **increases** that latency and can make extra waves harmful. [occupancy-composite.md](occupancy-composite.md) already Leaves over-filling memory-bound kernels that thrash **IC/L2**.

LLVM confirm (read-only): `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` combine **WG+LDS, VGPR, SGPR** only — no L2 (or any cache) argument.

Contrast with L0/GL1: those are **tiny per-CU / per-SA** filters (16 KB / 128 KB). L2 is **4 MiB GPU-wide** — larger reuse window, but still far below a layer shard or unsharded weight tensor; concurrent streams and over-occupied skinny GEMV still thrash it.

## 3. How misses hurt effective occupancy (no invented cycles)

Mental model only — **no published gfx1030 L2 miss-penalty table**:

1. Wave issues a vector load → L0 → GL1 → L2 tag check.
2. L2 miss → Infinity Cache; IC miss → GDDR6.
3. Wave waits on `s_waitcnt` / VMEM dependency → scheduler switches to another resident wave (**this** is why occupancy exists).
4. If every resident wave is also missing distinct lines through the same 4 MiB L2 (and/or the 128 MiB IC), switching finds **no** useful work → SIMD idle / memory-bound collapse.
5. PIX / `llvm-calc-occupancy` still report the VGPR∩LDS reservation — they never saw the thrash.

SLC / nontemporal ([cache-policy.md](cache-policy.md)): marks streaming at L2 (hit leaves age / Hit-No-Allocate with DLC). Correct for **cold** weight shards you are **not** trying to keep; wrong for a tile you want L2/IC to retain. Not an occupancy MaxWaves lever.

## 4. Profiling (cite counters; do not invent hit rates)

| Signal | How | Use |
|---|---|---|
| GL2 / GL2C hit rate | rocprofiler-compute Memory Chart / GL2 panel when the gfx10 analysis config exposes it | Mid-level filter health. Published RDNA3.5 formula shape: `100 * GL2C_HIT_sum / (GL2C_HIT_sum + GL2C_MISS_sum)` when denom ≠ 0 — **documented panels are gfx115x-oriented; verify gfx1030 config** before treating names as gospel |
| Instinct-style TCC | CDNA / MI docs use `TCC_HIT` / `TCC_MISS` | **Not** the RDNA naming; do not paste MI counter lists onto V620 |
| Wave memory wait / Idle | RGP instruction timing; rocprofiler-SDK thread trace | Correlate miss spikes with wait; green “hidden” latency only counts if other waves ran ALU |
| Working-set estimate | Hot bytes this step vs **4 MiB** (L2) and **128 MiB** (IC) | Rough craft check — not a hardware counter |

Do **not** treat a non-100% L2 hit rate alone as a dest flip. Streaming W4 that misses L2 by design can still be correct if **IC** absorbs reuse; thrash matters when **extra waves** turn a tolerable miss stream into GPU-wide L2/IC thrash.

## 5. Practical shapes (gfx1030 extras craft)

| Shape | Care about L2? | Why |
|---|---|---|
| Skinny decode GEMV, A in LDS, B streamed coalesced, modest waves/EU | **Yes — cap waves** | Classic GPUOpen memory-bound thrash: max waves multiply cold weight lines through 4 MiB L2 (and then IC) |
| Activation `x` / small scales already in LDS or VGPR | **Usually ignore L2** | Fits filters; limiter is DOT/VALU or weight path |
| Hot tile / small reused panel that fits in a few MiB | **Yes — keep default loads** | L2 is the first GPU-wide reusable store; nontemporal would defeat reuse |
| Fat unsharded weight GEMM that already blows IC | **L2 secondary** | Misses fall through IC to GDDR6; nontemporal + fit IC first ([infinity-cache.md](infinity-cache.md)) |
| Concurrent HIP streams on one V620 | **Yes at L2/IC** | Two streams share one GPU L2 and one IC — occupancy on *each* stream stacks thrash |

Rule of thumb (capacity, not a measured hit rate): L2 is **4 MiB** — architecture craft note: a 4096-d FP16 weight row is 8 KB → ~512 such rows fit in L2; a full layer does **not**. Prefer LDS for A-reuse tiles; keep L2/IC for coalesced streaming + reread windows; do not chase more waves when L2/IC miss + memory-wait already dominate.

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](architecture.md) §4.4 | L2 4 MB, 128 B, coherence point, channel unknown | Occupancy framing + explicit **not a PIX row** |
| [cache-policy.md](cache-policy.md) | Topology, SLC/nontemporal, thrash via over-occupy | Ties L2 miss → effective latency hiding; Take/Leave for decode wave caps |
| [infinity-cache.md](infinity-cache.md) | GPU-wide MALL thrash; L2 mid vs IC last | Separates **4 MiB L2** thrash from **128 MiB IC** thrash |
| [l0-gl1-occupancy.md](l0-gl1-occupancy.md) | CU/SA vector filters out of min | Sibling for **GPU-wide mid** data path |
| [icache-occupancy.md](icache-occupancy.md) | I$ fetch stalls out of min | Sibling for **data** L2 vs instruction path |
| [occupancy-composite.md](occupancy-composite.md) | Leave thrash IC/L2 when over-filling | L2 as named out-of-min effective-occupancy sibling |

## Sources (opened this pass)

1. GPUOpen Occupancy explained — reservation limiters VGPR/LDS/threadgroup/barriers; “Peak occupancy does not always mean peak performance” (cache thrash from extra waves); latency-hiding definition — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
2. HIP Hardware implementation — L2 coherence point, atomics at L2, hit-on-miss, write-through L0/L1 — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
3. ROCm GPU hardware specifications docs-7.2.4 — V620 gfx1030: L2 4 MiB, IC 128 MiB, GL1 128 KiB, L0 Vector 16, 72 CU — https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
4. `kfd_crat` Sienna Cichlid patch — L2 4096 KB, 128 B lines; L3 128×1024 KB, 64 B — https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html
5. RDNA 2 ISA Reference Guide 70648 — §2.4 read path (channels of L2 → L1 → L0); §8.1 GLC/SLC — https://gpuopen.com/news/rdna2-isa-available/
6. GPUOpen RDNA Architecture presentation — L0/L1/L2 table, 128 B lines — https://gpuopen.com/download/RDNA_Architecture_public.pdf
7. LLVM `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy.cpp` — occupancy min of WG+LDS, VGPR, SGPR only (no L2) — https://github.com/llvm/llvm-project
8. rocprofiler-compute RDNA GL2 cache — `GL2C_HIT_sum` / `GL2C_MISS_sum` formula shapes; **documented panels are RDNA3.5/gfx115x-oriented — verify gfx1030; on RDNA2 IC sits after L2 so “final on-chip before DRAM” wording does not map 1:1** — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.14.1/conceptual/rdna/gl2-cache.html
9. Wiki companions already locking topology / policy: [architecture.md](architecture.md), [cache-policy.md](cache-policy.md), [infinity-cache.md](infinity-cache.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [icache-occupancy.md](icache-occupancy.md), [occupancy-composite.md](occupancy-composite.md)
