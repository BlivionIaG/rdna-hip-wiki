---
title: "Models"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc
---
# Models

Model-specific HIP glue (one page per model family). Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [qwen4exp-flash-next-hip.md](qwen4exp-flash-next-hip.md) | T46 fdot2 glue; T47–T49 HC/QSA/PLE scalar | HC compute-correct @ `8960a3bc`. At `1dc3bac4`: `VLLM_RDNA_FUSED_HC` default **on**; fused HC prefill (`hc_prefill_rdna2.cu`) default mode 2; HC_PREFILL_HIP / QSA / PLE default off [^c-8960a3bc] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-8960a3bc]: opengfx1030/vllm-rdna commit [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc) (2026-09-16) — "fix(rocm): make the Qwen4Exp HC HIP chain compute correctly".
