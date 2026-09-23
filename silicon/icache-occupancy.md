# Instruction cache (I$) vs occupancy (gfx1030 / WGP)

Lock: **SQC instruction cache (I$) is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** It is a **shared 32 KB/WGP fetch resource**. Misses raise *instruction-fetch* idle (effective latency hiding), not a fifth row next to VGPR/LDS/WG/barrier. Completes the occupancy-adjacent set after [scratch-occupancy.md](scratch-occupancy.md) (also out of the theoretical min).

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:** this page is **I$ / SQC instruction cache**. Wiki shorthand **IC** elsewhere often means **Infinity Cache** ([infinity-cache.md](infinity-cache.md)) — do not conflate.

Date: **2026-09-21** Europe/Paris.

Companions: [architecture.md](architecture.md) §4.3, [cache-policy.md](cache-policy.md) §2.6, [occupancy-composite.md](occupancy-composite.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [l2-occupancy.md](l2-occupancy.md), [hip-craft.md](hip-craft.md) §4.2, [fa-occupancy.md](fa-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat I$ as **code-footprint / fetch-stall** pressure: 32 KB/WGP shared by **4 SIMD32**, 64 B lines. Size hot loops to fit; share one dequant/header path across call sites. |
| **Take** | Unroll the **inner K** (DOT2 / sdot4) for the ~5-cycle VALU dest latency; **do not** unroll whole M×N epilogues or paste four giant helpers into one TU ([hip-craft.md](hip-craft.md) §4.2). |
| **Take** | When a fat unrolled / EXL3-style kernel underperforms despite healthy VGPR∩LDS occupancy, collect **SQC_ICACHE_HITS / MISSES / REQ** (rocprofiler-compute Icache hit rate) before shaving another VGPR. |
| **Take** | Extra waves help I$-miss latency **only if** other resident waves still have ready instructions already in I$. Same-kernel thrash of one hot path is **not** fixed by more occupancy (unlike GDDR6 VMEM). |
| **Leave** | Do **not** put I$ / code size into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters` (VGPR / LDS / Thread Group Size / Barriers only). |
| **Leave** | Vector L0/GL1 thrash is a **separate** out-of-min sibling — see [l0-gl1-occupancy.md](l0-gl1-occupancy.md). |
| **Leave** | L2 thrash is a **separate** out-of-min sibling — see [l2-occupancy.md](l2-occupancy.md). |
| **Leave** | Do **not** invent a gfx1030 I$-miss cycle count — the ISA / GPUOpen occupancy page do not publish one. Profile hit rate + wave idle instead. |
| **Leave** | Do **not** rely on `.amdhsa_inst_pref_size` / COMPUTE_PGM_RSRC3 initial prefetch as a gfx1030 lever — that encoding is **gfx11+** (LLVM). `S_INST_PREFETCH` exists on GFX10 but is a **short ahead-of-PC** hint (1–3 × 64 B), not a cure for a 64 KB+ working set. |
| **Leave** | Do not open UNC / retip extras for this page. |

## 1. What gfx1030 I$ actually is

| Property | Value | Source |
|---|---|---|
| Size | **32 KiB per WGP** (~2 CU) | ROCm gpu-arch-specs (V620 / W6800: “L0 Instruction Cache (KiB) = 32”); RDNA architecture deck; `kfd_crat` Navi 21 “Scalar L1 Instruction Cache per SQC” |
| Line size | **64 B** | RDNA deck; `kfd_crat` |
| R/W | Read-only | RDNA deck |
| Sharing scope | **One WGP** — all **4 SIMD32** share one I$ | RDNA whitepaper front-end §; deck “32KB per WGP (~2CUs)” |
| Banks / assoc (RDNA whitepaper) | **4 banks × 128 lines × 64 B**; **4-way** set-associative | AMD RDNA Architecture whitepaper (dual-CU front-end). ISA / GPUOpen deck / `kfd_crat` omit associativity — cite whitepaper only for that cell; [architecture.md](architecture.md) §4.3 previously left assoc as unknown from deck/ISA/kfd |
| Fetch width | **32 B / cycle / SIMD** (typically 2–4 instructions) | RDNA whitepaper (≈4× GCN I$ bandwidth claim) |
| Hit-on-miss | **No** — duplicate pending fills count as misses | HIP + architecture §4.3 (same as scalar K$) |
| Scalar sibling | K$ **16 KiB/WGP**, 64 B lines (kernarg / `__constant__`) | Same sources; not this page’s limiter |

GCN contrast (deck): Vega-class I$ was **32 KB per 4 CUs** with **32 B** lines. RDNA moved to **32 KB per WGP** with **64 B** lines — denser per dual-CU front-end, not a larger absolute budget for one fat shader.

V620: 72 CU → **36 WGP** → 36 independent I$ instances at 32 KiB each. Waves on the same WGP **contend**; waves on different WGPs do not share that 32 KiB.

## 2. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Scratch / private | **No** | Latency; rare ROCr `waves_per_cu` cut |
| **I$** | **No** | Instruction-fetch stalls / wave **idle**; measured occupancy can still look “full” while IPC collapses |

GPUOpen Occupancy explained lists the compile-time reservation set as **VGPR, SGPR (fixed on RDNA), LDS, threadgroup size, barriers**. Instruction cache is **not** in that reservation story. Occupancy is framed as capacity to **hide memory latency** by switching waves — I$ misses are a different stall class: the wave cannot issue until the fetch returns, and sibling waves help only if their PCs are already warm in the same 32 KB.

rocprofiler-SDK thread-trace (gfx10+): wave **Idle** includes gaps caused by **instruction cache miss** (alongside arbiter loss and register dependency). That is the right mental model: I$ pressure shows up as **idle / issue wait**, not as a PIX limiter percentage.

## 3. Profiling (no invented miss latency)

| Signal | How | Use |
|---|---|---|
| `SQC_ICACHE_HITS` / `SQC_ICACHE_MISSES` / `SQC_ICACHE_REQ` | rocprofv3 PMC / rocprofiler-compute Memory Chart | Hit rate = `100 * HITS / (HITS + MISSES)` when denom ≠ 0 |
| Instruction Cache BW | `SQC_TC_INST_REQ × 128 B / time` (rocprofiler-compute SoL / Memory Chart) | Sanity that fetch traffic is real |
| Wave Idle / Issue Wait | Thread trace / SoL wave wait metrics | Correlate with I$ miss spikes |
| Code size | `llvm-objdump` / RGA ISA; `.text` size of the hot kernel | Rough footprint vs 32 KiB — not a hardware counter |

Do **not** treat a non-100% I$ hit rate alone as a dest flip. Decode TUs that share one small DOT loop usually sit fine; regress when epilogues + helpers exceed the WGP working set.

## 4. Prefetch on gfx1030 (narrow lever)

| Mechanism | gfx1030? | Role |
|---|---|---|
| `S_INST_PREFETCH` | ISA / LLVM **yes** (GFX10+ SOPP) | Prefetch **1–3 × 64 B** lines ahead of PC ([cache-policy.md](cache-policy.md) §2.6) |
| `.amdhsa_inst_pref_size` / RSRC3 INST_PREF_SIZE | **gfx11+** only | Initial entry-point prefetch window (up to ~8 KiB on gfx11, larger on gfx12) — **Leave** as a V620 lever |
| `llvm.prefetch` | ignored before gfx1250 | Not I$ |

Craft: keep the **steady-state loop** inside I$; do not plan a “prefetch the next layer into IC” opcode strategy for decode ([cache-policy.md](cache-policy.md)).

## 5. Practical shapes (fa_rdna2 / skinny / EXL3-class)

| Shape | Care about I$? | Why |
|---|---|---|
| Skinny decode DOT inner (small K-unroll, shared dequant) | **Usually ignore** | Hot path ≪ 32 KiB; limiter is VGPR/LDS or memory |
| `fa_rdna2`-class attention with bounded helpers | **Low** unless you paste many specializations into one CU object | Prefer one header path; FA resource pin stays on [fa-occupancy.md](fa-occupancy.md) (not restated here) |
| Giant fully-unrolled M×N + multiple dequant variants in one TU | **Yes** | Classic “mind the I$ size” foot-gun (RDNA deck ILP slide) |
| EXL3-style unpack → half2 → `fdot2` with fat codebook tables in `.text` | **Yes if tables/helpers land in instruction stream** | Prefer data in LDS/VGPR/kernarg; keep ISA loop compact (`3inst` class) |

Rule of thumb (GCN-era GPUOpen GDC still useful for magnitude): 32 KiB / ~4–8 B per instruction ≈ **4K–8K instructions** if one shader owns the cache; shared / multi-kernel WGP use wants smaller average footprints. Not a hard ISA limit — use it to reject “unroll everything” patches.

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](architecture.md) §4.3 | 32 KB/WGP, 64 B, RO, not hit-on-miss | Occupancy framing + whitepaper fetch/banks + explicit **not a PIX row** |
| [cache-policy.md](cache-policy.md) | Prefetch table; IC fit | Ties miss → wave idle / SQC counters |
| [hip-craft.md](hip-craft.md) §4.2 | Unroll foot-gun | Places I$ next to scratch as **out-of-min** occupancy sibling |
| [occupancy-composite.md](occupancy-composite.md) | PIX four + scratch out | I$ as effective-occupancy / IPC pressure, not fold input |

## Sources (opened this pass)

1. AMD RDNA Architecture whitepaper — L0 I$ 32 KB/WGP, 4-way, 4×128×64 B banks, 32 B/cycle/SIMD fetch — https://www.techpowerup.com/gpu-specs/docs/amd-rdna-whitepaper.pdf (same text as AMD public RDNA whitepaper mirrors)
2. GPUOpen RDNA Architecture presentation — I$ 32 KB/WGP, 64 B lines; “mind the I$ size” ILP slide — https://gpuopen.com/download/RDNA_Architecture_public.pdf
3. ROCm GPU hardware specifications — V620 gfx1030 “L0 Instruction Cache (KiB) = 32” — https://rocmdocs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
4. `kfd_crat` Navi 21 patch — Scalar L1 Instruction Cache per SQC = 32 KB / 2 CU — https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html
5. GPUOpen Occupancy explained — reservation limiters VGPR/LDS/threadgroup/barriers; latency-hiding definition; no I$ row — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
6. rocprofiler-compute RDNA System Speed-of-Light — Instruction Cache Hit Rate from `SQC_ICACHE_HITS` / `MISSES` — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/system-speed-of-light.html
7. rocprofiler-SDK thread trace — Idle includes instruction cache miss — https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.2.4/how-to/using-thread-trace.html
8. LLVM AMDGPU — `S_INST_PREFETCH` GFX10+ (`SOPInstructions.td`); `.amdhsa_inst_pref_size` gfx11+ only (PR #126981 / #192306) — https://github.com/llvm/llvm-project
9. GPUOpen GDC2017 Advanced Shader Programming on GCN — I$ size → instruction-count magnitude (historical, still useful for footprint intuition) — https://gpuopen.com/download/GDC2017-Advanced-Shader-Programming-On-GCN.pdf
10. RDNA 2 ISA Reference Guide 70648 — execution model / SQ front-end context (sizes from whitepaper/deck/`kfd_crat`, not a published miss-penalty table) — https://gpuopen.com/news/rdna2-isa-available/
