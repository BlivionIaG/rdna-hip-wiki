---
title: "Overlays"
sources:
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/skinny_gemms.cu
  - https://github.com/BlivionIaG/vllm/commit/3e05abc9bdb92100b0fec7a91e856c147dc6849c
---
# Overlays

Reviews of other gfx1030 trees we harvest from (read-only). Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [ikantkode-gfx1030.md](ikantkode-gfx1030.md) | sourced overlay (Triton, not HIP) | Take LLMM1 gate + RMSNorm; Leave GEMV [^f-3e05abc9-skinny-gemms-cu] |
| [rdna2-extras.md](rdna2-extras.md) | HIP / silicon review of the archive overlay | `rdna2_extras` @ `3e05abc9`, read-only [^c-3e05abc9] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^f-3e05abc9-skinny-gemms-cu]: BlivionIaG/vllm `csrc/rocm/skinny_gemms.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/skinny_gemms.cu).
[^c-3e05abc9]: BlivionIaG/vllm commit [`3e05abc9`](https://github.com/BlivionIaG/vllm/commit/3e05abc9bdb92100b0fec7a91e856c147dc6849c) (2026-08-18) — "fix(rocm): disable vllm::all_reduce dispatcher on gfx1030/gfx1100".
