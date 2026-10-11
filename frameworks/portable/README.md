---
title: "Portable kernel / IR frameworks (zoo)"
sources:
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/triton_attn.py
  - https://github.com/tile-ai/tilelang
  - https://iree.dev/guides/deployment-configurations/gpu-rocm/
  - https://docs.jax.dev/en/latest/pallas/gpu/reference.html
  - https://github.com/pytorch/pytorch/blob/7265531410fb689cf388a6fe083f772ec72e869a/torch/_inductor/select_algorithm.py#L950-L952
  - https://github.com/pytorch/pytorch/blob/7265531410fb689cf388a6fe083f772ec72e869a/torch/_inductor/runtime/hints.py#L304-L305
  - https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
---
# Portable kernel / IR frameworks (zoo)

Date: 2026-09-28. gfx1030 dest = HIP harvest; no framework runtime in vllm-rdna unless already contracted (Triton).

## Take / Leave

| Framework | Take | Leave |
|---|---|---|
| OpenAI Triton (ROCm) | Dest-adjacent; stock TRITON_ATTN | Blind CDNA/AITER configs [^f-49fb2ee-triton-attn-py] |
| TileLang | SIMT/schedule/HIP codegen ideas | WMMA/MFMA/MXFP4 (RDNA3+/CDNA) [^tilelang] |
| IREE + StableHLO | HIP compile zoo (`--iree-rocm-target=gfx1030`) | Serving replacement [^iree-rocm] |
| Apache TVM | Autotune / schedule ideas | Dest dependency |
| JAX Pallas | Triton backend ideas on ROCm | Mosaic GPU (NVIDIA) [^pallas-mgpu] |
| PyTorch Inductor | AMD Triton heuristics (`waves_per_eu`, `kpack`) | Shipping Inductor as engine [^inductor-sel][^inductor-hints] |
| CUTLASS / CuTeDSL | Layout algebra *ideas* | Any CuTe/CUTLASS code (NVIDIA) [^cutedsl] |

## AMD signals (short)

- TileLang: HIP + RDNA3/4/CDNA4 (2026). No gfx1030 WMMA. https://github.com/tile-ai/tilelang [^tilelang]
- IREE: HIP HAL; SKU table CDNA/RDNA3/4; arch string for gfx1030. https://iree.dev/guides/deployment-configurations/gpu-rocm/ [^iree-rocm]
- Pallas Mosaic / CuTeDSL: NVIDIA-first; not AMD dest.[^pallas-mgpu][^cutedsl]

## Related

- [engine/stock-baseline/triton-rocm.md](../../engine/stock-baseline/triton-rocm.md)
- [mojo-max/README.md](../mojo-max/README.md)

## Sources

[^f-49fb2ee-triton-attn-py]: vllm-project/vllm `vllm/v1/attention/backends/triton_attn.py` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/vllm/v1/attention/backends/triton_attn.py).
[^tilelang]: tile-ai/tilelang README (HIP backend, RDNA3/4/CDNA targets). <https://github.com/tile-ai/tilelang>
[^iree-rocm]: IREE deployment guide, GPU – ROCm (`--iree-rocm-target`, SKU table). <https://iree.dev/guides/deployment-configurations/gpu-rocm/>
[^pallas-mgpu]: JAX docs, *Mosaic GPU kernels with Pallas* (NVIDIA SM model, Hopper `wgmma`, Blackwell `tcgen05`). <https://docs.jax.dev/en/latest/pallas/gpu/reference.html>
[^inductor-sel]: pytorch/pytorch @ `72655314`, `torch/_inductor/select_algorithm.py` L950–L952 (`triton_meta["waves_per_eu"]`, `triton_meta["kpack"]`). <https://github.com/pytorch/pytorch/blob/7265531410fb689cf388a6fe083f772ec72e869a/torch/_inductor/select_algorithm.py#L950-L952>
[^inductor-hints]: pytorch/pytorch @ `72655314`, `torch/_inductor/runtime/hints.py` L304–L305 (`waves_per_eu: int`, `kpack: int`). <https://github.com/pytorch/pytorch/blob/7265531410fb689cf388a6fe083f772ec72e869a/torch/_inductor/runtime/hints.py#L304-L305>
[^cutedsl]: NVIDIA CUTLASS docs, CuTe DSL overview (compiles through the NVIDIA CUDA toolchain to CUDA kernels). <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html>
