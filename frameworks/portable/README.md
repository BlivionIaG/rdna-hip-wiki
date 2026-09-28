# Portable kernel / IR frameworks (zoo)

Date: 2026-09-28. gfx1030 dest = HIP harvest; no framework runtime in vllm-rdna unless already contracted (Triton).

## Take / Leave

| Framework | Take | Leave |
|---|---|---|
| OpenAI Triton (ROCm) | Dest-adjacent; stock TRITON_ATTN | Blind CDNA/AITER configs |
| TileLang | SIMT/schedule/HIP codegen ideas | WMMA/MFMA/MXFP4 (RDNA3+/CDNA) |
| IREE + StableHLO | HIP compile zoo (`--iree-rocm-target=gfx1030`) | Serving replacement |
| Apache TVM | Autotune / schedule ideas | Dest dependency |
| JAX Pallas | Triton backend ideas on ROCm | Mosaic GPU (NVIDIA) |
| PyTorch Inductor | AMD Triton heuristics (`waves_per_eu`, `kpack`) | Shipping Inductor as engine |
| CUTLASS / CuTeDSL | Layout algebra *ideas* | Any CuTe/CUTLASS code (NVIDIA) |

## AMD signals (short)

- TileLang: HIP + RDNA3/4/CDNA4 (2026). No gfx1030 WMMA. https://github.com/tile-ai/tilelang
- IREE: HIP HAL; SKU table CDNA/RDNA3/4; arch string for gfx1030. https://iree.dev/guides/deployment-configurations/gpu-rocm/
- Pallas Mosaic / CuTeDSL: NVIDIA-first; not AMD dest.

## Related

- [engine/triton-rocm.md](../../engine/triton-rocm.md)
- [mojo-max/README.md](../mojo-max/README.md)
