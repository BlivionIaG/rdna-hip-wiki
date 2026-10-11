---
title: "Attention"
sources:
  - https://github.com/vllm-project/vllm/commit/49fb2ee3481246ce88224220442c6ef30f88b209
---
# Attention

Attention-side kernels: stock Triton baseline, KV INT8, Sage QK, sparse MLA, indexer. Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [mla-sparse.md](mla-sparse.md) | scalar fp32 FMA (later `fdot2`) | Incomplete. Env-gated decode |
| [lightning-indexer.md](lightning-indexer.md) | scalar half FMA | Incomplete |
| [sage-qk.md](sage-qk.md) | QK `sdot4`, PV `fdot2` | Spec |
| [kv-int8.md](kv-int8.md) | i8 load + cvt → `fdot2` | Spec |
| [triton-attention-gfx1030.md](triton-attention-gfx1030.md) | stock Triton attention baseline + ISA map | Baseline evidence @ vLLM `49fb2ee`, not tuning. FP16 only [^c-49fb2ee] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-49fb2ee]: vllm-project/vllm commit [`49fb2ee`](https://github.com/vllm-project/vllm/commit/49fb2ee3481246ce88224220442c6ef30f88b209) (2026-08-17) — "\[ROCm\]\[CI\] add Aiter ops tests (#52208)".
