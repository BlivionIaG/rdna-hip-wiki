---
title: "Linear attention"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/7779514b449c358c14cb58c739cc33cd7efca3b4
  - https://github.com/BlivionIaG/vllm/commit/69d2efea371e52e4dca4c233e198ef627e181f0b
  - https://github.com/BlivionIaG/vllm/commit/77d6fdf801f8618a66dee12e6df14d983b530b33
---
# Linear attention

Linear-attention / recurrent kernels: GDN decode + prefill, causal conv1d, FlashKDA. Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [flashkda.md](flashkda.md) | 128×128 `S` → `fdot2` | Later. Not extras; no CUTLASS/SM90 |
| [causal-conv1d.md](causal-conv1d.md) | scalar fp32 FMA (depthwise FIR) | Live HIP @ `7779514b`. UPDATE wire order Leave [^c-7779514b] |
| [gdn-decode.md](gdn-decode.md) | fp32 FMA + shfl K-reduce | Decode HIP Live (`69d2efe`) [^c-69d2efe] |
| [gdn-prefill.md](gdn-prefill.md) | kkt scalar FMA; wy/delta_h/o `fdot2` | Prefill HIP Live (`77d6fdf8`). o BV 64 / ~56 KB. All `(2,4)` [^c-77d6fdf8] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-7779514b]: opengfx1030/vllm-rdna commit [`7779514b`](https://github.com/opengfx1030/vllm-rdna/commit/7779514b449c358c14cb58c739cc33cd7efca3b4) (2026-09-08) — "fix(rocm,fa-rdna2): torch::zeros for O_partial/M_partial on RDNA2 cudagraph replay".
[^c-69d2efe]: BlivionIaG/vllm commit [`69d2efe`](https://github.com/BlivionIaG/vllm/commit/69d2efea371e52e4dca4c233e198ef627e181f0b) (2026-08-23) — "feat: hand-port GDN packed decode to HIP RDNA2".
[^c-77d6fdf8]: BlivionIaG/vllm commit [`77d6fdf8`](https://github.com/BlivionIaG/vllm/commit/77d6fdf801f8618a66dee12e6df14d983b530b33) (2026-08-24) — "perf(gdn): o-kernel BV 32->64 halves workgroup count, +8.9% kernel / +0.5% e2e".
