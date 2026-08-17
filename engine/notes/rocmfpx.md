# ROCmFPX — source digest (not a vLLM port)

Date: 2026-08-17. Sourced from [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX). Their tok/s stay here. Occupancy still first.

**Contracts:** [../rocmfpx.md](../rocmfpx.md) (leverage), [../llamacpp-rocmfpx.md](../llamacpp-rocmfpx.md) (V620 side project), `kernels/rocmfpx.md` (silicon). Stock GGUF in vLLM is a loader for Q4_0/K/IQ*, not these types.

**Verdict:** do **not** port this into vLLM. It is a **llama.cpp GGUF family** (CPU + HIP + Vulkan), tuned on Strix Halo `gfx1151`. There is a `scripts/build-rdna2.sh` (gfx1030), not a vLLM backend, not a HIP FA/GEMM we can drop in.

Steal one silicon lesson. Ignore the rest as an engine.

## What it is

| Family | GGUF | Layout |
|---|---|---|
| ROCmFP4 | `Q4_0_ROCMFP4` | 32 W / block, 16 B nibbles + **2× UE4M3** (4.50 bpw) |
| ROCmFP4 FAST | `Q4_0_ROCMFP4_FAST` | same nibbles + **1× UE4M3** (4.25 bpw) |
| ROCmFP2 | `Q2_0_ROCMFPX` | 2.50 bpw, S40 `{−4,−1,+1,+4}`, dual UE4M3 |
| FP3/6/8 | preview | not optimized |

Agent/STRIX presets are **tensor routing** (keep embeddings / attn-KV at Q5_K/Q6_K), not a new DOT.

## Inner op (HIP)

Not an FP4 unit. Not NVFP4 E2M1. Not integer W4A4 `sdot8`.

Codebook10 (half-scale signed ints, max **10**): `{0, ±1, ±2, ±3, ±4, ±6, ±8, ±10}`. Expand nibbles with `__builtin_amdgcn_perm` into **i8 DP4A operands**, then integer dot vs **Q8** activations (`vec_dot_*_q8_1`). Scale is finite UE4M3, arithmetic decode (LUT rejected on their HIP path).

```
// keep as a lesson
nibble → perm codebook → i8x4 → sdot4 / DP4A vs A8
C *= ue4m3_scale

// drop
sdot8 on raw nibbles          // codebook ≠ {−8…7}
fdot2 on E2M1 LUT             // not NVFP4 {0,±0.5,…,±6}
FlashInfer / WMMA / FP4 MMA
porting llama.cpp MMVQ into vLLM
```

Closest contract we already have: **W4A8** (unpack → `sdot4`), [sdot4-explore.md](../../kernels/sdot4-explore.md). ROCmFP4 is a **different codebook** + UE4M3, same DOT class.

NVFP4 rematch they document: same UE4M3, **7/8 codebook levels**; top mag is 12 (NV) vs 10 (ROCm). Still not our [nvfp4.md](../nvfp4.md) `fdot2` path unless we invent a second LUT.

## Why a vLLM port loses

| Piece | Why not |
|---|---|
| Engine | Custom GGUF types. Stock vLLM GGUF loader accepts Q4_0/K/IQ*, not `Q4_0_ROCMFP4`. |
| Kernels | MMVQ/MMQ + their FA thread-group knobs, occupancy-tuned for **gfx1151**. |
| MTP | llama.cpp `--spec-type draft-mtp`. We already block on fat tile `q>1`. |
| KV | TurboQuant / q8 — separate from weights; not a vLLM dtype. |
| Vulkan | their fastest Strix path. We are HIP/vLLM. |

A “port” would be: new quant key + loader + codebook→`sdot4` GEMM **after W8A8**. That is a new format, not a drop-in.

## Steal (optional, later)

1. **`amdgcn_perm` codebook expand** — no LDS LUT (matches our mxfp4 rule). Useful if we ever ship a non-linear 4-bit codebook.
2. **UE4M3 arithmetic scale** — they rejected a scale LUT on HIP. Same instinct as our NVFP4 E4M3 bit-trick.
3. **Don’t A-quant decode** — their MMVQ is W×Q8; we already said W4A4 decode stays W4A16.

Side project, not a vLLM card: [../llamacpp-rocmfpx.md](../llamacpp-rocmfpx.md). Occupancy still first.

## Sources

- https://github.com/charlie12345/ROCmFPX README + `ggml/rocmfp4/`
- `rocmfp4_hip_codebook.cuh` (`amdgcn_perm` Codebook10)
- `rocmfp4.h` (`block_rocmfp4` 18 B / `block_rocmfp4_fast` 17 B)
- [nvfp4.md](../nvfp4.md), [w4a4.md](../w4a4.md), [silicon/valu.md](../../silicon/valu.md)
