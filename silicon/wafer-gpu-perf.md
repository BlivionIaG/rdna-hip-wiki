# Silicon: wafer-ai/gpu-perf-engineering-resources

Engine digest: [../engine/notes/wafer-gpu-perf.md](../engine/notes/wafer-gpu-perf.md). This page is ISA only. List verified 2026-08-24 (`main`). No kernels in that repo.

Their **AMD** hardware block is CDNA4 / MI350 (whitepaper, ISA, counters). That is MFMA + HBM + chiplet — **not** gfx1030. Do not copy it over [architecture.md](architecture.md) or [valu.md](valu.md).

## Take (silicon)

| Their item | gfx1030 |
|---|---|
| “Read each architecture with its ISA” | Ours is **70648** (RDNA2 Shader ISA Nov 2020), not their CDNA4 PDF |
| Roofline / Volkov “measure, don’t worship occupancy” | Ridge = packed DOT vs GDDR6 512 GB/s. Measurement take. **Does not** close FA prefill `(N, 1)` |
| rocprofiler-compute | Read stack / counters. [hip-craft.md](hip-craft.md) |
| CK / HipKittens as *read* | Tile language only. [codegen-stack.md](codegen-stack.md) |

## Leave (silicon)

- CDNA4 ISA / MI350 counters as this box
- AITER / CK XDL / HipKittens **import** (WMMA/MFMA first-class; extras stays HIP + `fdot2`)
- CUDA occupancy / shared-mem / PTX — map to wave32 + 64 KB/WG LDS + IC 128 MB, don’t port
- Their hardware list has **no RDNA2 / no `V_DOT2_F32_F16`**. DOT lock stays [valu.md](valu.md)

## Contract

Occupancy leftover unchanged. GDN tip `e054854f` (pressure, not tok/s). No number from this list on coverage.
