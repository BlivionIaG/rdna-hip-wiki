# Infinity Cache vs effective occupancy (gfx1030)

Lock: **Infinity Cache (IC / MALL / L3) is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** It is the **GPU-wide last-level** on-die data cache (**128 MiB** on Navi 21 / V620), **after L2** and **before GDDR6**. Thrash raises *memory-wait* / GDDR6 miss latency → weaker effective latency hiding even when VGPR∩LDS occupancy looks full. Same occupancy-adjacent class as [l2-occupancy.md](l2-occupancy.md) and [l0-gl1-occupancy.md](l0-gl1-occupancy.md). Not a fifth PIX row.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:** this page is **Infinity Cache / IC / MALL / L3** (GPU-wide, **128 MiB** on V620, **64 B** lines). Do **not** confuse with:

- **SQC I$** ([icache-occupancy.md](icache-occupancy.md)) — instruction cache, 32 KiB/WGP
- **L2 / GL2** ([l2-occupancy.md](l2-occupancy.md)) — mid-level, **4 MiB**, **128 B** lines
- **L0 / GL1** ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)) — CU/SA vector filters

SKU fit tables and “no HIP persist bit” stay in [infinity-cache.md](infinity-cache.md). This page is the **occupancy framing** sibling.

Date: **2026-09-28** Europe/Paris.

Companions: [infinity-cache.md](infinity-cache.md), [architecture.md](architecture.md) §4.5, [cache-policy.md](cache-policy.md) §0 / §3 / §4, [l2-occupancy.md](l2-occupancy.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [occupancy-composite.md](occupancy-composite.md), [icache-occupancy.md](icache-occupancy.md), [kcache-occupancy.md](kcache-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [fa-occupancy.md](fa-occupancy.md), [hip-craft.md](hip-craft.md) §1.3 / §6, [../engine/cache-aware.md](../engine/cache-aware.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat IC as the **last GPU-wide reuse cliff** on the data path: L0 → GL1 → L2 (4 MiB) → **IC (128 MiB)** → GDDR6. Extra waves help only while they hide real miss latency **without** multiplying distinct cold lines through the same 128 MiB. |
| **Take** | Quantize / TP-shard until the **hot layer shard + working KV pages this step touches** fit IC; keep **default** loads on that set ([infinity-cache.md](infinity-cache.md)). Cap occupancy with `__launch_bounds__` / `amdgpu_waves_per_eu` when the stream is already memory-bound ([cache-policy.md](cache-policy.md) §0). |
| **Take** | When measured occupancy is high but memory-wait dominates **and** the working set already misses L2, check **hot bytes vs 128 MiB** and concurrent streams **before** chasing another VGPR wave — IC thrash is the last-level cliff, not a PIX limiter. |
| **Take** | Distinguish **L2 thrash** (4 MiB mid filter) from **IC thrash** (128 MiB last-level). A shard that misses L2 but fits IC is still an IC win ([l2-occupancy.md](l2-occupancy.md)); over-filling waves can thrash **both**. |
| **Leave** | Do **not** put IC capacity into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters` (VGPR / LDS / Thread Group Size / Barriers only). LLVM `computeOccupancy` mins WG+LDS, VGPR, SGPR — no cache size term. |
| **Leave** | Do **not** invent IC miss-cycle tables, TB/s rooflines, associativity, or V620-specific hit rates. AMD’s “2.4× / 2.5×” marketing is relative, not a lab number ([infinity-cache.md](infinity-cache.md)). Profile. |
| **Leave** | Do **not** invent a HIP persist / IC bypass / IC prefetch intrinsic — gfx1030 has none. Do not copy rocprofiler gfx115x “GL2 = final on-chip cache before DRAM” wording onto V620 — on RDNA2, **IC sits after L2**. |
| **Leave** | Do not invent a gfx1030-locked MALL/IC counter list; leave undocumented IC hit/miss names. Do not open UNC / retip extras for this page. |

## 1. What gfx1030 Infinity Cache actually is

| Property | Value | Source |
|---|---|---|
| Size (Navi 21 / V620) | **128 MiB** GPU-wide | ROCm gpu-arch-specs V620 “Infinity Cache (MiB) = 128”; `kfd_crat` Sienna Cichlid L3 128×1024 KB |
| Line size | **64 B** | `kfd_crat` L3; architecture / infinity-cache (L0/L1/L2 are **128 B**) |
| Sharing | **Whole GPU** (one MALL per V620; TP=4 does **not** share IC across cards) | infinity-cache; architecture §4.5 |
| Role on read path | Last-level after L2; miss → GDDR6 | ISA hierarchy + RDNA2 IC intro |
| HIP control | **None** — no persist, no bypass, no IC prefetch | infinity-cache; cache-policy |
| Slice sketch (Navi 21) | 16 × 8 MB (recap) | namu RDNA §2.2 / infinity-cache cite — capacity framing only |
| Introduced | **RDNA 2** (GFX10.3); absent on RDNA1 / Vega / GCN | AMD 2020 press; infinity-cache |

Read path (restated):

```text
VGPR / LDS → L0 (per CU, TCP) → GL1 (per SA) → L2 (4 MiB, GPU) → Infinity Cache (128 MiB) → GDDR6
```

ROCm gpu-arch-specs (docs-7.2.4) **Radeon PRO V620** row (IC / L2 excerpt):

| Field | V620 |
|---|---|
| LLVM target | gfx1030 |
| CU | 72 |
| Infinity Cache | 128 MiB |
| L2 | 4 MiB |
| Graphics L1 | 128 KiB |
| L0 Vector | 16 KiB |

Source: https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html

**vs L2:** L2 stayed **2–4 MiB** while SE/WGP grew — a layer does **not** live in L2. IC is the capacity that makes TP-sharded W4/mxfp4 decode reuse plausible on V620 ([infinity-cache.md](infinity-cache.md) fit table).

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
| K$ (SQC DCache) | **No** | Scalar-load wait ([kcache-occupancy.md](kcache-occupancy.md)) |
| L0 / GL1 | **No** | CU/SA vector-filter thrash ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)) |
| L2 | **No** | GPU-wide mid-cache thrash ([l2-occupancy.md](l2-occupancy.md)) |
| **Infinity Cache** | **No** | GPU-wide last-level thrash; measured occupancy can look “full” while miss latency collapses to GDDR6 |
| SPI / ACE / grid fill | **No** | Lack of work / launch-rate ([spi-ace-occupancy.md](spi-ace-occupancy.md)) |

GPUOpen Occupancy explained: compile-time reservation set is **VGPR, SGPR (fixed on RDNA), LDS, threadgroup size, barriers**. Cache hierarchy is the **other** side — “Peak occupancy does not always mean peak performance”: more waves can thrash scarce shared caches **outside** the WGP booking model. Occupancy is capacity to hide memory latency by switching waves; IC thrash **increases** that latency (GDDR6) and can make extra waves harmful. [occupancy-composite.md](occupancy-composite.md) already Leaves over-filling memory-bound kernels that thrash **IC/L2**.

LLVM confirm (read-only): `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` combine **WG+LDS, VGPR, SGPR** only — no IC (or any cache) argument.

## 3. How misses hurt effective occupancy (no invented cycles)

Mental model only — **no published gfx1030 IC miss-penalty table**:

1. Wave issues a vector load → L0 → GL1 → L2 → IC tag check.
2. IC miss → GDDR6 (V620 peak **512 GB/s** product page — not a measured kernel BW).
3. Wave waits on `s_waitcnt` / VMEM dependency → scheduler switches to another resident wave (**this** is why occupancy exists).
4. If every resident wave is also missing distinct lines through the same **128 MiB** IC (or stacking a second HIP stream / fat prefill on the same V620), switching finds **no** useful work → SIMD idle / memory-bound collapse.
5. PIX / `llvm-calc-occupancy` still report the VGPR∩LDS reservation — they never saw the thrash.

Two occupancy bugs on IC-bound decode ([infinity-cache.md](infinity-cache.md)):

- **Too few waves** (`waves_per_eu(1,1)` trap) — wastes CU before IC matters.
- **Too many waves** on a miss stream — multiplies GDDR6 traffic through the same MALL.

SLC / nontemporal ([cache-policy.md](cache-policy.md)): correct for **cold** weight shards you are **not** trying to keep in IC; wrong for a TP-sharded W4 panel you want reread. Not an occupancy MaxWaves lever.

## 4. Profiling (cite shapes; do not invent IC hit rates)

| Signal | How | Use |
|---|---|---|
| Hot-set estimate | Hot bytes this step vs **128 MiB** (IC) and **4 MiB** (L2) | Rough craft check — capacity, not a hardware counter ([infinity-cache.md](infinity-cache.md) fit table) |
| Wave memory wait / Idle | RGP instruction timing; rocprofiler-SDK thread trace | Correlate GDDR6-bound waits with over-occupy; green “hidden” latency only counts if other waves ran ALU |
| L2 / GL2C panels | rocprofiler-compute Memory Chart when exposed | Mid filter only; **documented panels are gfx115x-oriented** and often call GL2 “final on-chip before DRAM” — **false on RDNA2** (IC after L2). Verify gfx1030; do not treat GL2C as IC |
| MALL / IC counters | — | **Leave inventing names.** No gfx1030-locked IC hit/miss list in this wiki pass |

Do **not** treat a non-100% L2 hit rate alone as “IC failed.” Streaming W4 that misses L2 by design can still be correct if **IC** absorbs reuse; thrash matters when **extra waves** or a second stream turn a tolerable IC-resident set into GPU-wide MALL thrash → GDDR6.

## 5. Practical shapes (gfx1030 extras craft)

| Shape | Care about IC? | Why |
|---|---|---|
| Skinny decode GEMV / FA gathers, W4 shard ≤ IC, modest waves/EU | **Yes — fit + cap waves** | Classic GPUOpen memory-bound thrash: max waves multiply cold lines through 128 MiB IC |
| 7B/27B W4A16 or mxfp4 **TP=4** layer shard | **Yes — keep default loads** | Fit tables in [infinity-cache.md](infinity-cache.md); leftover IC for short GQA KV |
| Fat unsharded / FP16 layer that already blows 128 MiB | **Stream nontemporal** | IC cannot hold it; use IC for `x` + working KV pages only |
| Concurrent HIP streams / SDMA on one V620 | **Yes** | One GPU-wide MALL — occupancy on *each* stream stacks thrash |
| Prefill GEMM whose activation working set already misses 128 MiB | **IC secondary** | No IC win expected; do not over-occupy hoping for one |
| `(1,1)` waves_per_eu on decode | **First fix occupancy floor** | CU waste before IC matters |

Rule of thumb (capacity, not a measured hit rate): keep the **reread set** inside **128 MiB** on a quiet GPU; prefer LDS for A-reuse tiles; do not chase more waves when IC miss + memory-wait already dominate.

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [infinity-cache.md](infinity-cache.md) | SKU table, fit, no HIP persist, over-occupy thrash | Explicit **not a PIX MaxWaves row** + effective-occupancy mental model |
| [l2-occupancy.md](l2-occupancy.md) | Mid 4 MiB thrash; points IC to infinity-cache | Named **IC occupancy sibling** after L2 |
| [cache-policy.md](cache-policy.md) | Topology, SLC, thrash via over-occupy | Ties IC miss → effective latency hiding; Take/Leave for decode wave caps |
| [architecture.md](architecture.md) §4.5 | IC 128 MB, 64 B, not XGMI | Occupancy framing |
| [occupancy-composite.md](occupancy-composite.md) | Leave thrash IC/L2 when over-filling | IC as named out-of-min effective-occupancy sibling |
| [icache-occupancy.md](icache-occupancy.md) | SQC I$ fetch stalls | Different resource — instruction vs last-level data |

## Sources (opened this pass)

1. GPUOpen Occupancy explained — reservation limiters VGPR/LDS/threadgroup/barriers; “Peak occupancy does not always mean peak performance” (cache thrash from extra waves); latency-hiding definition — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
2. ROCm GPU hardware specifications docs-7.2.4 — V620 gfx1030: Infinity Cache 128 MiB, L2 4 MiB, GL1 128 KiB, L0 Vector 16, 72 CU — https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
3. `kfd_crat` Sienna Cichlid patch — L2 4096 KB, 128 B lines; L3 128×1024 KB, 64 B — https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html
4. AMD 2020-10-28 RX 6000 press (IC as last-level on-die cache) — https://www.amd.com/en/newsroom/press-releases/2020-10-28-amd-unveils-next-generation-pc-gaming-with-amd-rad.html
5. V620 product page — 128 MB IC, 72 CU, 512 GB/s — https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
6. RDNA 2 ISA Reference Guide 70648 — hierarchy / memory path framing — https://gpuopen.com/news/rdna2-isa-available/
7. LLVM `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy.cpp` — occupancy min of WG+LDS, VGPR, SGPR only (no IC) — https://github.com/llvm/llvm-project
8. rocprofiler-compute RDNA GL2 cache — `GL2C_*` formula shapes; **documented panels are RDNA3.5/gfx115x-oriented and call GL2 “final on-chip before DRAM” — does not map 1:1 to RDNA2 (IC after L2); no gfx1030 IC counter invent** — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.14.1/conceptual/rdna/gl2-cache.html
9. Wiki companions: [infinity-cache.md](infinity-cache.md), [l2-occupancy.md](l2-occupancy.md), [cache-policy.md](cache-policy.md), [architecture.md](architecture.md), [occupancy-composite.md](occupancy-composite.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md)
