---
title: "AOT glue"
sources:
  - https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb
  - https://github.com/opengfx1030/vllm-rdna/commit/71a54552ae8f6f8ad7f315c5a2291cebdb747690
---
# AOT glue

Small HIP glue that replaces Triton JIT: RMSNorm, AOT list. Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [triton-jit-aot.md](triton-jit-aot.md) | HIP AOT vs leftover Triton JIT | Take: conv1d / ApplyRotaryEmb / SwigluStep. Leave: FA/EXL3/W4/GDN. RMSNorm AOT ≠ PIECEWISE |
| [layernorm.md](layernorm.md) | scalar fp32 FMA + shfl + tiny LDS | Live AOT HIP (`83de31cf` + gated `71a54552`). Cudagraph-safe. Not FA leftover [^c-83de31cf][^c-71a54552] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-83de31cf]: BlivionIaG/vllm commit [`83de31cf`](https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb) (2026-09-01) — "fix(rocm/layernorm): close namespace braces before global-scope public entries".
[^c-71a54552]: opengfx1030/vllm-rdna commit [`71a54552`](https://github.com/opengfx1030/vllm-rdna/commit/71a54552ae8f6f8ad7f315c5a2291cebdb747690) (2026-09-11) — "perf: HIP gated_rms_norm kernel for Qwen3.x GDN RMSNormGated".
