---
title: "Models"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc
  - https://github.com/opengfx1030/vllm-rdna/commit/ec349e9079f6d19865362010bebcd36bfbbe0f8d
  - https://github.com/opengfx1030/vllm-rdna/commit/0f596e6e0e9b2683ef7a4ec3eb6ef93eac9e1fd9
---
# Models

Model-specific HIP glue (one page per model family). Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [qwen4exp-flash-next-hip.md](qwen4exp-flash-next-hip.md) | T46 fdot2 glue; T47–T49 HC/QSA/PLE scalar | HC compute-correct @ `8960a3bc`. At `1dc3bac4`: `VLLM_RDNA_FUSED_HC` default **on**; fused HC prefill (`hc_prefill_rdna2.cu`) default mode 2; HC_PREFILL_HIP / QSA / PLE default off [^c-8960a3bc] |
| [dsv4-mhc-hip.md](dsv4-mhc-hip.md) | scalar fp32 (no DOT); 16-lane xor-shuffle Sinkhorn | DSv4 mHC pre/post HIP (`rdna/dsv4/mhc.cu`, `ec349e90` + `0f596e6e`, unchanged at `64d59b89`); decode pre GEMM split over 32 blocks; ≤16 tokens fused GEMM, else rocBLAS. Selected after AITER / TileLang for 4 streams fp16/bf16 [^c-0f596e6e] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-8960a3bc]: opengfx1030/vllm-rdna commit [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc) (2026-09-16) — "fix(rocm): make the Qwen4Exp HC HIP chain compute correctly".
[^c-0f596e6e]: opengfx1030/vllm-rdna commit [`0f596e6e`](https://github.com/opengfx1030/vllm-rdna/commit/0f596e6e0e9b2683ef7a4ec3eb6ef93eac9e1fd9) (2026-10-09) — "perf(rdna): split the decode mHC pre GEMM across blocks, parallel Sinkhorn" (file last touched before tip `64d59b89`; first added in [`ec349e90`](https://github.com/opengfx1030/vllm-rdna/commit/ec349e9079f6d19865362010bebcd36bfbbe0f8d)).
