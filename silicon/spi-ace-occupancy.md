# SPI / ACE vs measured occupancy (gfx1030 / V620)

Lock: **SPI (shader processor input / workgroup manager) and ACE (asynchronous compute engine) are not PIX / `llvm-calc-occupancy` MaxWaves limiters.** They explain why **measured** occupancy sits under **theoretical** when the grid cannot fill the ASIC or waves finish faster than the front-end can refill slots. Completes the “measured ≤ theory” story named on [occupancy-composite.md](occupancy-composite.md) — siblings own VGPR/LDS/WG/barrier and the cache thrash pages.

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

Date: **2026-09-25** Europe/Paris.

Companions: [architecture.md](architecture.md) § CP/SPI, [occupancy-composite.md](occupancy-composite.md), [wg-size-occupancy.md](wg-size-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [hip-craft.md](hip-craft.md) §6, [fa-occupancy.md](fa-occupancy.md), [leapdragon.md](leapdragon.md) (grid fill), [cache-policy.md](cache-policy.md) (V620 fill math).

## Take / Leave

| | |
|---|---|
| **Take** | When RGP / PIX shows **theory high** but **measured low** on a fat FA/W4 grid, first check **grid fill**, **host/device barriers between independent dispatches**, and **per-wave work vs launch cost** — not another VGPR shave. |
| **Take** | V620 fill ceiling: **72 CU → 36 WGP → 36 × 4 SIMD × 16 = 2304 wave32 slots**. A dispatch with fewer wavefronts **cannot** hit 100% measured occupancy even at full theory ([cache-policy.md](cache-policy.md)). |
| **Take** | Overlap **independent** kernels (no dependency barrier) across queues / ACEs when the profile shows empty-GPU ramp after a drain. HIP: multiple ACEs exist; each ACE dispatches **one kernel at a time**. Exact ACE count on Navi 21: **unknown** — do not invent. |
| **Take** | For skinny decode grids that under-fill 36 WGP, increase useful work per wave (tile / segment batching) so launch cost is amortized — same class as leapdragon’s segment fill ([leapdragon.md](leapdragon.md)). |
| **Leave** | Do **not** put SPI/ACE / launch-rate into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters` (VGPR / LDS / Thread Group Size / Barriers only). |
| **Leave** | Do **not** invent a gfx1030 ACE count, SPI width, or waves-per-cycle launch table. Profile measured vs theory; cite GPUOpen / HIP for the mechanism. |
| **Leave** | Do **not** assume workgroup→WGP mapping is sticky across launches — SPI placement is resource-driven and non-deterministic ([architecture.md](architecture.md); HIP hardware implementation). |
| **Leave** | Do not open UNC / retip extras for this page. |

## 1. Front-end path (host → SIMD slots)

HIP hardware implementation + wiki architecture:

```text
Host AQL / HIP launch
  → CP (CPF fetch + CPC decode)
    → ACE(s)          // break kernels into workgroups; one kernel per ACE at a time
      → SPI           // place WGs on WGP/CU; init SGPRs; reserve wave slots + VGPR + SGPR + LDS
        → SQ / SIMD   // issue + execute resident waves
```

| Block | Role for occupancy |
|---|---|
| **CP** | Accepts launches, memory copies, fences |
| **ACE** | Workgroup fan-out; concurrent kernels need **multiple ACEs** / queues and **no** false dependency |
| **SPI** | Allocates the PIX reservation set on one WGP (WGP mode) or one CU (CU mode); holds the whole WG together for LDS / `s_barrier` |
| **SIMD slots** | RDNA2: **16** wave slots / SIMD32 — that is MaxWaves, already in [vgpr-occupancy.md](vgpr-occupancy.md) |

SPI tracks the same four resources PIX names: **wave slots, VGPR, SGPR, LDS**. On GFX10+ SGPR is always enough for MaxWaves ([sgpr-occupancy.md](sgpr-occupancy.md)). When any resource is short, the WG **waits** in SPI — that wait is the theoretical limiter story. When resources are free but measured occupancy stays low, the cause is **lack of work** or **launch-rate**, not a fifth PIX row.

## 2. Measured ≤ theory — two SPI/ACE classes

GPUOpen Occupancy explained (RDNA2 MaxWaves = 16 / SIMD):

| Class | Symptom | Mechanism | Craft |
|---|---|---|---|
| **Lack of work** | Theory 16/16, measured ≪ that; short dispatch; wave count in RGP details ≪ 2304 | Grid cannot fill **36 WGP × 4 × 16** | Grow grid (segments / tiles / batch) or **overlap** an independent sibling kernel |
| **Launch-rate** | High measured occupancy at start, then drop / oscillate; waves finish faster than refill; empty-GPU ramp after a barrier | SPI/ACE setup (register init, WG place) slower than wave retire; dependency barrier drains the chip before the next launch | More useful ALU/mem work **per wave**; remove false barriers between independent launches; multi-queue overlap when legal |

Also normal: **end-of-dispatch drain** (last few waves) and **post-barrier ramp** — unavoidable when a real dependency exists; only fix is remove the dependency.

These are **not** VGPR/LDS/WG/barrier limiters. PIX `WaveOccupancyLimiters` stay binary “is this resource blocking a free slot?” — they do not encode ACE throughput.

## 3. V620 fill math (do not use 6900 XT / RDNA3 decks)

| ASIC | CU | WGP | Max wave32 slots |
|---|---|---|---|
| **V620** | **72** | **36** | **2304** (= 36 × 4 × 16) |
| RX 6900 XT (full Navi 21 class example in older wiki notes) | 80 | 40 | 2560 |
| GPUOpen RX 7900 XTX example (RDNA3) | — | 48 | 3072 |

Sources: AMD V620 product / press (72 CU); ROCm gpu-arch-specs V620 row; ISA CU = half WGP. Prefer **2304** for V620 craft. [occupancy-composite.md](occupancy-composite.md) historically said “40 WGP” for V620 — **wrong**; corrected to 36 / 2304 on this pass.

Example: 256-thread WG = 8 wave32 → **288** workgroups to fill all slots if every WG is resident at once (upper bound; real residency is still VGPR∩LDS capped).

Skinny decode with `seqs × kv_heads` ≪ 36 WGP is the leapdragon under-fill case — SPI has nothing to schedule; shaving VGPR does not create waves.

## 4. Profiling (no invented ACE counters)

| Signal | How | Use |
|---|---|---|
| Theoretical vs measured occupancy | RGP Wavefront occupancy / Pipeline tab | Theory high + measured low → this page; both low → PIX sibling |
| Wavefront count / dispatch duration | RGP details panel | Lack of work if total waves ≪ 2304 |
| Occupancy ramp after barrier | RGP Event timing + occupancy graph | Dependency drain vs launch-rate mid-dispatch |
| PIX `WaveOccupancyLimiters` | AMD PIX plugin | If all zero and measured still low → lack of work / launch-rate, not VGPR/LDS |
| Runtime `hipOccupancyMaxActiveBlocksPerMultiprocessor` | HIP | Launchability gate only — does **not** detect under-filled grids |

Do **not** invent a gfx1030 “ACE utilization” PMC name. If a counter appears in rocprofiler for CP/SPI, verify on box before locking it here.

## 5. Practical shapes (gfx1030 extras craft)

| Shape | Care about SPI/ACE? | Why |
|---|---|---|
| Fat FA / W4 prefill grid ≫ 2304 waves, theory = measured | **Usually ignore** | Front-end saturated; look at VGPR/LDS/cache thrash |
| Theory high, measured low, short kernel | **Yes** | Lack of work or launch-rate; batch work per wave / overlap |
| Skinny decode, few seqs × heads | **Yes — fill** | Under-fills 36 WGP; segments / tiles before VGPR chase |
| Graph capture with many tiny nodes + barriers | **Yes** | Barrier drain + launch-rate; prefer fused nodes / fewer false deps ([graph-capture.md](graph-capture.md)) |
| Multi-stream independent MoE / AR siblings | **Yes — overlap** | Legal concurrency across ACEs/queues without a fence between them |
| Single stream, real producer→consumer fence | **Leave as-is** | Empty-GPU ramp is the dependency tax |

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [architecture.md](architecture.md) | CP → ACE → SPI; SPI reserves slots+VGPR+SGPR+LDS; ACE count unknown | Occupancy framing: measured gap classes + V620 2304 fill |
| [occupancy-composite.md](occupancy-composite.md) | Named SPI/ACE as reason measured ≤ theory | Dedicated Take/Leave + profiling; **fixes V620 40→36 WGP** |
| [wg-size-occupancy.md](wg-size-occupancy.md) / [barrier-occupancy.md](barrier-occupancy.md) | Atomic WG lifetime / barrier slots on SPI | Separates **reservation** limiters from **front-end rate** |
| [leapdragon.md](leapdragon.md) | Softmax segments to fill 36 WGP | Ties under-fill to GPUOpen lack-of-work class |
| [hip-craft.md](hip-craft.md) | RGP measured occupancy | Points here when theory ≠ measured |

## Sources (opened this pass)

1. GPUOpen Occupancy explained — measured vs theory; lack of work; launch-rate; PIX limiters VGPR/LDS/threadgroup/barriers only; RDNA2 MaxWaves 16 — https://gpuopen.com/learn/occupancy-explained/ (2023-12-20 / 2024-06-26)
2. ROCm HIP Hardware implementation — CP/CPF/CPC, ACE (one kernel at a time), SPI workgroup manager, resource set (warp slots / VGPR / SGPR / LDS), non-deterministic WG→CU map — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
3. ROCm GPU hardware specifications docs-7.2.4 — Radeon PRO V620: gfx1030, **72 CU** — https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
4. AMD Radeon PRO V620 product / press — 72 CU — https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
5. Wiki companions: [architecture.md](architecture.md), [occupancy-composite.md](occupancy-composite.md), [cache-policy.md](cache-policy.md), [leapdragon.md](leapdragon.md), [vgpr-occupancy.md](vgpr-occupancy.md), [sgpr-occupancy.md](sgpr-occupancy.md)
