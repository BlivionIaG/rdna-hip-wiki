---
title: "Scalar data cache (K$) vs occupancy (gfx1030 / WGP)"
sources:
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html
  - https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/drm/amd/amdkfd/kfd_crat.c#L556
  - https://gpuopen.com/download/RDNA_Architecture_public.pdf
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
  - https://gpuopen.com/learn/occupancy-explained/
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/system-speed-of-light.html
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/tcp-cache.html
  - https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html
  - https://rocm.docs.amd.com/en/docs-7.2.3/conceptual/gpu-arch/mi300-mi200-performance-counters.html
---

# Scalar data cache (K$) vs occupancy (gfx1030 / WGP)

Lock: **SQC scalar data cache (K$ / sL1D) is not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** It is a **shared 16 KB/WGP** read path for **kernarg / `__constant__` / `s_load_*` / SMEM**. Misses raise *scalar-memory* wait (effective latency hiding / SALU stall), not a fifth row next to VGPR/LDS/WG/barrier. Completes the SQC occupancy-adjacent pair with [icache-occupancy.md](icache-occupancy.md) (I$ = instruction; K$ = scalar data).[^llvm-calc-occupancy]

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:** this page is **K$ / SQC DCache / L0 Scalar Cache / sL1D**. Do **not** conflate with **I$** ([icache-occupancy.md](icache-occupancy.md)), **vector L0/TCP** ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)), or wiki shorthand **IC** = Infinity Cache ([infinity-cache.md](../arch-and-memory/infinity-cache.md)). ROCm Radeon PRO tables label it **L0 Scalar Cache (KiB)**; Instinct tables use **L1 Scalar Cache**.

Date: **2026-09-24** Europe/Paris.

Companions: [architecture.md](../arch-and-memory/architecture.md) §4.3, [cache-policy.md](../arch-and-memory/cache-policy.md) §0 / `__constant__`, [hip-craft.md](../hip-craft/hip-craft.md) §4 (kernarg/SMEM), [icache-occupancy.md](icache-occupancy.md), [occupancy-composite.md](occupancy-composite.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [l2-occupancy.md](l2-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md), [fa-occupancy.md](../attention/fa-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat K$ as **uniform scalar / kernarg / constant** pressure: **16 KiB/WGP**, 64 B lines, shared by the WGP front-end. Keep kernarg + `__constant__` small and wave-uniform; put hot scales/strides in SGPRs once, not as per-iteration vector reloads. [^rocm-gpu-arch-specs][^kfd-crat] |
| **Take** | Prefer **device kernarg** (`HIP_FORCE_DEV_KERNARG=1`, HIP 7.14 default) so launch args hit the scalar path without a host-visible detour ([hip-craft.md](../hip-craft/hip-craft.md) §4). [^rocm-reference-env-variables-html] |
| **Take** | When decode launches look “full” on VGPR∩LDS but SALU / `s_waitcnt lgkmcnt` dominate, collect **SQC_DCACHE_HITS / MISSES / REQ** (rocprofiler-compute Scalar Data Cache Hit Rate) **before** chasing another VGPR wave. |
| **Take** | Extra waves help K$-miss latency **only if** sibling waves have useful work while this wave waits on `s_load`. Same-kernel thrash of a fat constant working set is **not** fixed by more occupancy. |
| **Leave** | Do **not** put K$ / `__constant__` size into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters` (VGPR / LDS / Thread Group Size / Barriers only). [^llvm-calc-occupancy] |
| **Leave** | Do **not** put a **per-lane** nibble→fp16 LUT in `__constant__` / K$ and index it per thread — serializes the inner loop (llama.cpp #24438). W4A16 uses the `0x64006400` bit-trick so you never do that ([cache-policy.md](../arch-and-memory/cache-policy.md)). |
| **Leave** | Do **not** invent a gfx1030 K$-miss cycle table. Profile hit rate + `lgkmcnt` wait / wave idle. |
| **Leave** | Do **not** use `S_ATC_PROBE` / `S_ATC_PROBE_BUFFER` as a weight-tensor strategy — probes SQC data cache only; useless for large tensors ([cache-policy.md](../arch-and-memory/cache-policy.md)). |
| **Leave** | Do not open UNC / retip extras for this page. |

## 1. What gfx1030 K$ actually is

| Property | Value | Source |
|---|---|---|
| Size | **16 KiB per WGP** (~2 CU) | ROCm gpu-arch-specs V620 / W6800 “L0 Scalar Cache (KiB) = 16”; RDNA architecture deck; `kfd_crat` Navi 21 “Scalar L1 Data Cache per SQC” [^kfd-crat][^rdna-arch] |
| Line size | **64 B** | RDNA deck; `kfd_crat` [^kfd-crat][^rdna-arch] |
| R/W | Read-oriented scalar path (invalidate via `S_DCACHE_INV`) | ISA / HIP; [cache-policy.md](../arch-and-memory/cache-policy.md) |
| Sharing scope | **One WGP** — SQC shared by the dual-CU front-end | RDNA deck; architecture §4.3 [^rdna-arch] |
| Hit-on-miss | **No** — duplicate pending fills count as misses | HIP + architecture §4.3 (same as I$) |
| Traffic | `s_load_*`, kernarg (AS4), `__constant__`, scalar buffer loads | hip-craft §4; LLVM AMDGPUUsage [^llvm-amdgpuusage] |
| Instruction sibling | I$ **32 KiB/WGP**, 64 B lines | [icache-occupancy.md](icache-occupancy.md) [^rocm-gpu-arch-specs][^kfd-crat] |

ROCm gpu-arch-specs (docs-7.2.4) **Radeon PRO V620** row (scalar / instruction excerpt):

| Field | V620 |
|---|---|
| LLVM target | gfx1030 |
| CU | 72 |
| L0 Vector Cache | 16 KiB [^rocm-gpu-arch-specs][^kfd-crat] |
| **L0 Scalar Cache** | **16 KiB** [^rocm-gpu-arch-specs][^kfd-crat] |
| **L0 Instruction Cache** | **32 KiB** [^rocm-gpu-arch-specs][^kfd-crat] |
| L2 | 4 MiB [^rocm-gpu-arch-specs][^kfd-crat] |
| Infinity Cache | 128 MiB [^rocm-gpu-arch-specs] |

Source: https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html[^rocm-reference-gpu-arch-specs-html]

V620: 72 CU → **36 WGP** → 36 independent K$ instances at 16 KiB each. Waves on the same WGP **contend** for that 16 KiB; waves on different WGPs do not share it.[^rocm-gpu-arch-specs][^kfd-crat][^v620-product]

**vs SGPR occupancy:** GFX10+ always allocates the full SGPR file and SGPR is **not** a MaxWaves limiter ([sgpr-occupancy.md](sgpr-occupancy.md)). K$ is the **cache** that feeds those SGPRs from kernarg/constant — a miss stalls the scalar pipe; it does not change the PIX SGPR column.

## 2. Relation to occupancy — out of the PIX min

Theoretical waves/EU ([occupancy-composite.md](occupancy-composite.md)):

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect when stressed [^llvm-calc-occupancy] |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Caps reserved wave slots |
| Scratch / private | **No** | Latency; rare ROCr `waves_per_cu` cut |
| I$ (SQC) | **No** | Instruction-fetch idle ([icache-occupancy.md](icache-occupancy.md)) |
| **K$ (SQC DCache)** | **No** | Scalar-load wait / `lgkmcnt`; measured occupancy can look “full” while SALU stalls |
| L0 / GL1 / L2 / IC | **No** | Vector / GPU-wide data thrash (siblings) |

GPUOpen Occupancy explained: compile-time reservation set is **VGPR, SGPR (fixed on RDNA), LDS, threadgroup size, barriers**. Scalar data cache is **not** in that reservation story. Occupancy hides **memory** latency by switching waves — K$ misses are scalar-memory stalls: the wave waits on `s_waitcnt lgkmcnt` until the fill returns; sibling waves help only if they are not also thrashing the same 16 KiB.[^gpuopen-occupancy]

LLVM confirm (read-only): `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` combine **WG+LDS, VGPR, SGPR** only — no K$ (or any cache) argument.[^llvm-calc-occupancy]

## 3. How misses hurt effective occupancy (no invented cycles)

Mental model only — **no published gfx1030 K$-miss penalty table**:

1. Wave issues `s_load` / kernarg / `__constant__` read → SQC DCache tag check.
2. Miss → fill (through the scalar miss path toward GL1/L2 — profile, do not invent hop counts).
3. Wave parks on `s_waitcnt lgkmcnt` → scheduler may run another resident wave.
4. If every resident wave is also missing distinct constant/kernarg lines through the same 16 KiB K$, switching finds **no** useful scalar work → SALU / issue idle.[^rocm-gpu-arch-specs][^kfd-crat]
5. PIX / `llvm-calc-occupancy` still report the VGPR∩LDS reservation — they never saw the thrash.[^llvm-calc-occupancy]

**Coherence note:** scalar buffer loads are a **separate** cache from vector L0; Clang marks `__builtin_amdgcn_s_buffer_load_*` as generally avoidable unless the read-only contract is intentional ([hip-craft.md](../hip-craft/hip-craft.md)). Do not expect vector stores to update K$ without an invalidate.

## 4. Profiling (cite counters; do not invent hit rates)

| Signal | How | Use |
|---|---|---|
| `SQC_DCACHE_HITS` / `SQC_DCACHE_MISSES` / `SQC_DCACHE_REQ` | rocprofv3 PMC / rocprofiler-compute Memory Chart / SoL | Hit rate = `100 * HITS / (HITS + MISSES)` when denom ≠ 0 |
| Scalar Data Cache BW | `SQC_TC_DATA_READ_REQ × 128 B / time` (rocprofiler-compute SoL) | Sanity that scalar traffic is real [^rocm-rdna-system-speed-of-light-html] |
| `SQC_DCACHE_INPUT_VALID_READYB` | PMC (when exposed) | Input stall while DCache not ready |
| Wave Issue Wait / SALU wait | Thread trace / SoL | Correlate with K$ miss spikes |
| `.amdhsa_kernarg_size` / constant footprint | `.s` / NT_AMDGPU_METADATA | Rough working set vs 16 KiB — not a hardware counter [^llvm-amdgpuusage] |

**Caveat:** shipped rocprofiler-compute RDNA SoL panels are often **gfx115x-oriented**; counter **names** (`SQC_DCACHE_*`) match Instinct SQC docs and RDNA SoL formulas — **verify gfx1030 analysis config** on box before treating panel availability as gospel.[^rocm-rdna-tcp-cache-html]

Do **not** treat a non-100% K$ hit rate alone as a dest flip. Healthy decode with tiny kernarg + a few uniform scales should sit fine; regress when constant tables or per-lane LUTs blow the 16 KiB window.

## 5. Practical shapes (gfx1030 extras craft)

| Shape | Care about K$? | Why |
|---|---|---|
| Skinny decode: tiny kernarg, scales in SGPR, A in LDS, B streamed | **Usually ignore** | Working set ≪ 16 KiB; limiter is VGPR/LDS or vector memory [^rocm-gpu-arch-specs][^kfd-crat] |
| Wave-uniform group scales / zp in `__constant__` or kernarg | **Yes — keep small** | Exactly what K$ is for; fits when tables are compact |
| Per-lane nibble LUT in `__constant__` | **Yes — Leave** | Per-lane K$ index serializes the loop (llama.cpp #24438); use bit-trick / VALU instead |
| Fat weight / codebook tables stuffed into `__constant__` | **Yes — Leave** | Exceeds 16 KiB; belongs in global + LDS/VGPR, not K$ [^rocm-gpu-arch-specs][^kfd-crat] |
| Giant kernarg blob / many descriptors reloaded every tile | **Yes** | Misses + `lgkmcnt` wait; hoist to SGPR once |
| Concurrent kernels on one WGP with large constant sets | **Yes** | Share one 16 KiB K$ [^rocm-gpu-arch-specs][^kfd-crat] |

Rule of thumb (capacity, not a measured hit rate): K$ is **16 KiB** — a few dozen uniform dwords and a small scale table fit; a layer shard or dequant LUT does **not**. Prefer LDS/VGPR for per-lane data; keep K$ for true wave-uniform traffic.[^rocm-gpu-arch-specs][^kfd-crat]

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](../arch-and-memory/architecture.md) §4.3 | K$ 16 KB/WGP, 64 B, RO, not hit-on-miss | Occupancy framing + explicit **not a PIX row** [^rocm-gpu-arch-specs][^kfd-crat] |
| [cache-policy.md](../arch-and-memory/cache-policy.md) | `__constant__` → K$; LUT Leave; `S_ATC_PROBE` | Ties miss → effective latency hiding / SQC_DCACHE counters |
| [hip-craft.md](../hip-craft/hip-craft.md) §4 | Kernarg / SMEM / `HIP_FORCE_DEV_KERNARG` | Places K$ next to I$ as **out-of-min** occupancy sibling [^rocm-reference-env-variables-html] |
| [icache-occupancy.md](icache-occupancy.md) | Named K$ as scalar sibling | Dedicated Take/Leave + profiling for DCache |
| [sgpr-occupancy.md](sgpr-occupancy.md) | SGPR non-limiter on GFX10+ | Separates **file** (always allocated) from **cache** that fills it |
| [occupancy-composite.md](occupancy-composite.md) | Leave I$/L0/L2 out of min | K$ as named out-of-min effective-occupancy sibling |

## Sources

_Sources (opened this pass)_


1. ROCm GPU hardware specifications docs-7.2.4 — V620 gfx1030: L0 Scalar Cache 16 KiB, L0 Instruction Cache 32 KiB — https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
2. GPUOpen RDNA Architecture presentation — I$/K$ 32/16 KB per WGP, 64 B lines — https://gpuopen.com/download/RDNA_Architecture_public.pdf
3. `kfd_crat` Navi 21 patch — Scalar L1 Data Cache per SQC = 16 KB / 2 CU — https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html
4. GPUOpen Occupancy explained — reservation limiters VGPR/LDS/threadgroup/barriers; no K$ row; peak occupancy ≠ peak perf — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
5. rocprofiler-compute RDNA System Speed-of-Light — Scalar Data Cache Hit Rate from `SQC_DCACHE_HITS` / `MISSES`; BW from `SQC_TC_DATA_READ_REQ` — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/system-speed-of-light.html
6. rocprofiler-compute RDNA GL0 / Memory Chart — Dcache Hit Rate / Utilization formulas (`SQC_DCACHE_*`) — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/tcp-cache.html
7. AMD Instinct performance counters (SQC block naming) — `SQC_DCACHE_HITS` / `MISSES` / `REQ` / `MISSES_DUPLICATE` — https://rocm.docs.amd.com/en/docs-7.2.3/conceptual/gpu-arch/mi300-mi200-performance-counters.html
8. HIP env vars — `HIP_FORCE_DEV_KERNARG` default 1, 2–3 µs — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html
9. llama.cpp #24438 — per-lane K$ / `__constant__` LUT serialization
10. Wiki companions: [architecture.md](../arch-and-memory/architecture.md), [cache-policy.md](../arch-and-memory/cache-policy.md), [hip-craft.md](../hip-craft/hip-craft.md), [icache-occupancy.md](icache-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md), [occupancy-composite.md](occupancy-composite.md)

Footnotes:

[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-10-11)
[^rocm-reference-env-variables-html]: HIP environment variables — HIP 7.16.0 Documentation. https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html (checked 2026-10-11)
[^kfd-crat]: Linux v6.12 amdkfd kfd_crat.c, sienna_cichlid_cache_info (Navi 21: TCP 16 KB/CU, I$ 32 KB/2 CU, K$ 16 KB/2 CU, GL1 128 KB/10 CU, L2 4096 KB, L3 128 MB). https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/drm/amd/amdkfd/kfd_crat.c#L556 (checked 2026-10-11)
[^rdna-arch]: AMD, "RDNA Architecture" (GPUOpen, 2019). https://gpuopen.com/download/RDNA_Architecture_public.pdf (checked 2026-10-11)
[^llvm-amdgpuusage]: LLVM User Guide for AMDGPU Backend (AMDGPUUsage). https://llvm.org/docs/AMDGPUUsage.html (checked 2026-10-11)
[^rocm-reference-gpu-arch-specs-html]: GPU hardware specifications — ROCm Documentation. https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11)
[^gpuopen-occupancy]: GPUOpen, "Occupancy explained". https://gpuopen.com/learn/occupancy-explained/ (checked 2026-10-11)
[^rocm-rdna-system-speed-of-light-html]: System Speed-of-Light — ROCm Compute Profiler 3.6.0 documentation. https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/system-speed-of-light.html (checked 2026-10-11)
[^rocm-rdna-tcp-cache-html]: GL0 — ROCm Compute Profiler 3.6.0 documentation. https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/tcp-cache.html (checked 2026-10-11)
[^lists-2021-march-061392-html]: (PATCH) drm/amdkfd: Update L1 and add L2/3 cache information. https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html (checked 2026-10-11)
[^rocm-gpu-arch-mi300-mi200-performance-counters-h]: MI300 and MI200 Series performance counters and metrics — ROCm Documentation. https://rocm.docs.amd.com/en/docs-7.2.3/conceptual/gpu-arch/mi300-mi200-performance-counters.html (checked 2026-10-11)
[^rocm-gpu-arch-specs]: ROCm docs 7.2.4, "GPU architecture specifications" (Radeon PRO V620 row: gfx1030, 32 GiB, 72 CU, wave 32/64, LDS 128 KiB, Infinity Cache 128 MiB, L2 4 MiB, GL1 128 KiB, L0 vector 16 KiB, L0 scalar 16 KiB, L0 instruction 32 KiB, VGPR file 512 KiB). https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11)
[^v620-product]: AMD Radeon PRO V620 product page (72 CU, 4608 stream processors, 32 GB GDDR6, 256-bit, 512 GB/s, 300 W TBP, 2.2 GHz peak engine clock, 20.28 TFLOPs FP32 / 40.55 TFLOPs FP16 vector). https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html (checked 2026-10-11)
