---
title: "Persistent kernels"
sources:
  - https://github.com/vllm-project/vllm/issues/60904
---
# Persistent kernels

Persistent, single-launch decode kernels (the MonoKernel class from vLLM RFC #60904[^rfc]) and how they map onto gfx1030. Back to the [kernels index](../README.md).

| Page | What it covers |
|---|---|
| [monokernel-gfx1030-plan.md](monokernel-gfx1030-plan.md) | Silicon plan for a gfx1030 / V620 MonoKernel port: grid sync, occupancy, per-phase map, FFN-only boundary, PCIe all-reduce, phases |

## Sources

[^rfc]: vLLM RFC [#60904](https://github.com/vllm-project/vllm/issues/60904), "A standard for MonoKernels in vllm/models" (open, read 2026-10-13).
