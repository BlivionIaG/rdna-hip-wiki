---
title: "Silicon: wafer-ai/gpu-perf-engineering-resources"
sources:
  - https://github.com/wafer-ai/gpu-perf-engineering-resources
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/opengfx1030/vllm-rdna/commit/e054854f239b139b8bec5991345056b49315735d
---

# Silicon: wafer-ai/gpu-perf-engineering-resources

Engine digest: none in this repo yet (the old `engine/notes/wafer-gpu-perf.md` link never resolved). Upstream list: https://github.com/wafer-ai/gpu-perf-engineering-resources. This page is ISA only. List verified 2026-08-24 (`main`). No kernels in that repo.[^github-wafer-ai-gpu-perf-engineering-resources]

Their **AMD** hardware block is CDNA4 / MI350 (whitepaper, ISA, counters). That is MFMA + HBM + chiplet — **not** gfx1030. Do not copy it over [architecture.md](../arch-and-memory/architecture.md) or [valu.md](../arch-and-memory/valu.md).

## Take (silicon)

| Their item | gfx1030 |
|---|---|
| “Read each architecture with its ISA” | Ours is **70648** (RDNA2 Shader ISA Nov 2020), not their CDNA4 PDF [^isa70648] |
| Roofline / Volkov “measure, don’t worship occupancy” | Ridge = packed DOT vs GDDR6 512 GB/s. Measurement take. **Does not** close FA prefill `(N, 1)` |
| rocprofiler-compute | Read stack / counters. [hip-craft.md](../hip-craft/hip-craft.md) |
| CK / HipKittens as *read* | Tile language only. [codegen-stack.md](../hip-craft/codegen-stack.md) |
| CUTLASS / FA-3 **method** | Take: producer/consumer stages, epilogue, occupancy math. Remap to wave32 + 64 KB/WG LDS + `fdot2`. **Leave** TMA / WGMMA / `tcgen05` / CuTe as-is [^rocm-gpu-arch-specs][^llvm-baseinfo] |

## gfx1100 (read, not extras)

W7800 has local WMMA (`V_WMMA_*` 16×16×16). FA-3 overlap, CUTLASS pipeline, CK/HipKittens tiles are **readable** for that ISA — rewrite later in hippih, never extras import. Not CDNA4, not a V620 opcode.

## Leave (silicon)

- CDNA4 ISA / MI350 counters as this box
- AITER / CK XDL / HipKittens **import** (WMMA/MFMA first-class; extras stays HIP + `fdot2`)
- CUDA occupancy / shared-mem / PTX — map to wave32 + 64 KB/WG LDS + IC 128 MB, don’t port[^rocm-gpu-arch-specs][^llvm-baseinfo]
- Their hardware list has **no RDNA2 / no `V_DOT2_F32_F16`**. DOT lock stays [valu.md](../arch-and-memory/valu.md)

## Contract

Occupancy leftover unchanged. GDN tip `e054854f` (pressure, not tok/s). No number from this list on coverage.[^commit-e054854f]

## Sources

Primary sources for this page, cited inline as footnotes:

[^github-wafer-ai-gpu-perf-engineering-resources]: GitHub - wafer-ai/gpu-perf-engineering-resources: A curated resource list for learning AI performance engineering, from GPU fundamentals to production inference.. https://github.com/wafer-ai/gpu-perf-engineering-resources (checked 2026-10-11)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)
[^commit-e054854f]: opengfx1030/vllm-rdna commit `e054854f239b`. https://github.com/opengfx1030/vllm-rdna/commit/e054854f239b139b8bec5991345056b49315735d (exists, verified via GitHub API 2026-10-11)
[^llvm-baseinfo]: LLVM AMDGPUBaseInfo.cpp (IsaInfo), main snapshot eb61ef62df4f. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp (checked 2026-10-11)
[^rocm-gpu-arch-specs]: ROCm docs 7.2.4, "GPU architecture specifications" (Radeon PRO V620 row: gfx1030, 32 GiB, 72 CU, wave 32/64, LDS 128 KiB, Infinity Cache 128 MiB, L2 4 MiB, GL1 128 KiB, L0 vector 16 KiB, L0 scalar 16 KiB, L0 instruction 32 KiB, VGPR file 512 KiB). https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11)
