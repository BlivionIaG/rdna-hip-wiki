---
title: "Persistent kernels"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/1c1426d585c9b2653265947734225721cfc7ea6e
  - https://github.com/vllm-project/vllm/issues/60904
---
# Persistent kernels

Persistent, single-launch decode kernels (the MonoKernel class from vLLM RFC #60904[^rfc]) and how they map onto gfx1030. Back to the [kernels index](../README.md).

| Page | What it covers |
|---|---|
| [monokernel-gfx1030-dest.md](monokernel-gfx1030-dest.md) | What dest built at `1c1426d5` (PR #94): K1/K2 persistent kernel, fused GDN core, QSA decode, residency cap, plain-launch barrier, fixed-point atomics, opt-in in-kernel AR[^dest] |
| [monokernel-gfx1030-plan.md](monokernel-gfx1030-plan.md) | (historical @ `e245a469`)  Silicon plan for a gfx1030 / V620 MonoKernel port: grid sync, occupancy, per-phase map, FFN-only boundary, PCIe all-reduce, phases |

## Sources

[^dest]: dest `rdna_extras` PR #94 merge [`1c1426d585c9b2653265947734225721cfc7ea6e`](https://github.com/opengfx1030/vllm-rdna/commit/1c1426d585c9b2653265947734225721cfc7ea6e).

[^rfc]: vLLM RFC [#60904](https://github.com/vllm-project/vllm/issues/60904), "A standard for MonoKernels in vllm/models" (open, read 2026-10-13).
