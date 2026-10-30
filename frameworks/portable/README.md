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
  - https://github.com/triton-lang/triton/releases/tag/v3.9.0
  - https://github.com/triton-lang/triton/pull/12042
  - https://github.com/pytorch/pytorch/blob/v2.15.0/.ci/docker/triton_version.txt
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

- Triton 3.9.0 (2026-10-27) removed RDNA1/RDNA2 (`gfx101x`, `gfx103x`) from the AMD backend; the release notes say to stay on Triton 3.8 for those GPUs.[^triton-39][^triton-12042] PyTorch 2.15.0 bundles Triton 3.9.0, so a stock 2.15 upgrade carries this change.[^pt215-triton] Any gfx1030 Triton pin or upgrade is a dest decision for VLLM_FORK_Manager.

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
[^triton-39]: triton-lang/triton v3.9.0 release notes, Breaking changes ("RDNA1/RDNA2 removed ... Use Triton 3.8 for those GPUs"). <https://github.com/triton-lang/triton/releases/tag/v3.9.0>
[^triton-12042]: triton-lang/triton PR #12042, "[AMD] Remove RDNA1 and RDNA2 target support", merged 2026-09-30 (`4c39833`). <https://github.com/triton-lang/triton/pull/12042>
[^pt215-triton]: pytorch/pytorch @ `v2.15.0`, `.ci/docker/triton_version.txt` = `3.9.0`. <https://github.com/pytorch/pytorch/blob/v2.15.0/.ci/docker/triton_version.txt>
[^cutedsl]: NVIDIA CUTLASS docs, CuTe DSL overview (compiles through the NVIDIA CUDA toolchain to CUDA kernels). <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html>
