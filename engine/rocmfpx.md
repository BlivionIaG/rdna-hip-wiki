# ROCmFPX — leverage, not a vLLM port

Date: 2026-08-17. Contract. Digest: [notes/rocmfpx.md](notes/rocmfpx.md). Source: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX). Occupancy still first. Do not edit `perf/rdna2_w4a16` from here.

**Verdict:** no vLLM port. It is llama.cpp GGUF + HIP/Vulkan, tuned on Strix `gfx1151`. There is a `scripts/build-rdna2.sh`. We **leverage** the codebook expand + integer-dot. Side project: [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md).

## What we steal

HIP codebook (`rocmfp4_hip_codebook.cuh`): nibble → `__builtin_amdgcn_perm` → **i8 DP4A operands** (Codebook10 `{0,±1,±2,±3,±4,±6,±8,±10}`). Scale is finite **UE4M3**, arithmetic decode (they rejected a scale LUT). Activations on the hot path are **Q8** (`vec_dot_*_q8_1`).

```
// keep
nibble → perm codebook → i8x4 → ggml_cuda_dp4a / sdot4 vs A8
C *= ue4m3

// drop
sdot8 on raw nibbles     // not {−8…7}
fdot2 E2M1 LUT           // not NVFP4
porting MMVQ/MMQ into vLLM
FlashInfer / WMMA / FP4 MMA
```

Closest existing contract: **W4A8** after W8A8 ([sdot4-explore.md](../kernels/sdot4-explore.md), [w4a4.md](w4a4.md)). Different codebook, same DOT class. @RDNA2_Researcher confirms the issued opcode (`sdot4` vs scalar FMA) on the ISA dump.

`amdgcn_perm` expand = no LDS LUT (same rule as mxfp4 / llama.cpp #24438).

## What we do not steal

| Piece | Why |
|---|---|
| Engine / scheduler | ggml graphs, not vLLM V1 |
| GGUF loader | vLLM is safetensors / AWQ / compressed-tensors |
| MTP | llama.cpp `draft-mtp`. We still need fat tile `q>1` |
| Vulkan | their Strix default. We are HIP/vLLM |
| TurboQuant KV | runtime cache type, not a vLLM dtype |
| Their tok/s | gfx1151, not V620 |

## vLLM (later, optional)

Only if someone names a **safetensors** checkpoint with this codebook. Then: new quant key + `perm`→`sdot4` GEMM **after W8A8**. Not a rewrite of `q_gemm_rdna2`.

## Sources

- `ggml/rocmfp4/rocmfp4.h`, `rocmfp4_hip_codebook.cuh`
- [notes/rocmfpx.md](notes/rocmfpx.md)
- [silicon/valu.md](../silicon/valu.md)
