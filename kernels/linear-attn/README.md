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
| [causal-conv1d.md](causal-conv1d.md) | scalar fp32 FMA (depthwise FIR) | Live HIP (still default on at `1dc3bac4`). UPDATE wire order fixed at `1dc3bac4` (HIP before Triton) [^c-7779514b] |
| [gdn-decode.md](gdn-decode.md) | fp32 FMA + shfl K-reduce | Decode HIP Live (`69d2efe`); default on at `1dc3bac4` [^c-69d2efe] |
| [gdn-prefill.md](gdn-prefill.md) | kkt scalar FMA; wy/delta_h/o `fdot2` | Prefill HIP opt-in at `1dc3bac4` (`VLLM_GDN_HIP_PREFILL=1`). o BV 32 / 45312 B and delta_h 256 thr at the tip (`77d6fdf8` BV 64 is historical). All `(2,4)` [^gdn-tip-readme] [^c-77d6fdf8] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-7779514b]: opengfx1030/vllm-rdna commit [`7779514b`](https://github.com/opengfx1030/vllm-rdna/commit/7779514b449c358c14cb58c739cc33cd7efca3b4) (2026-09-08) — "fix(rocm,fa-rdna2): torch::zeros for O_partial/M_partial on RDNA2 cudagraph replay".
[^c-69d2efe]: BlivionIaG/vllm commit [`69d2efe`](https://github.com/BlivionIaG/vllm/commit/69d2efea371e52e4dca4c233e198ef627e181f0b) (2026-08-23) — "feat: hand-port GDN packed decode to HIP RDNA2".
[^c-77d6fdf8]: BlivionIaG/vllm commit [`77d6fdf8`](https://github.com/BlivionIaG/vllm/commit/77d6fdf801f8618a66dee12e6df14d983b530b33) (2026-08-24) — "perf(gdn): o-kernel BV 32->64 halves workgroup count, +8.9% kernel / +0.5% e2e".
[^gdn-tip-readme]: See [gdn-prefill.md](gdn-prefill.md) current-state table; `csrc/rocm/gdn_prefill_o_rdna2.cu` L51-L77 and `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` L168-L175, dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/gdn_prefill_o_rdna2.cu#L51-L77 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L168-L175
