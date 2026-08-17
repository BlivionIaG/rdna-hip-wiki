# ROCmFPX — silicon (codebook → perm → sdot4)

Engine: [engine/rocmfpx.md](../engine/rocmfpx.md). Side project: [engine/llamacpp-rocmfpx.md](../engine/llamacpp-rocmfpx.md). **Later.** Occupancy still first. No vLLM port.

Source tree: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) `ggml/rocmfp4/`. Opcode on gfx1030 is **not dumped yet** — this page is the expected contract. The V620 side project owns the `.s`.

## Expected inner loop

| Step | Op | Notes |
|---|---|---|
| Load W | packed nibbles, 32 weights / block (17 or 18 B) | FAST = 1× UE4M3; dual-scale = 2× |
| Expand | `__builtin_amdgcn_perm` Codebook10 → i8 | `{0,±1,±2,±3,±4,±6,±8,±10}` — **not** `{−8…7}` |
| Scale | UE4M3 arithmetic → f32 (they rejected a scale LUT) | epilogue or pre-scale into the i8 path |
| A | Q8_1 (`vec_dot_*_q8_1`) | same class as W8A8 A-quant |
| DOT | expect `V_DOT4C_I32_I8` / `ggml_cuda_dp4a` | **confirm on dump.** Fallback is scalar FMA after `rocmfp4_decode_i8` (the standalone dequant kernel does that) |

```
// keep
nibble → perm codebook → i8x4 → sdot4 vs A8
C *= ue4m3

// drop
sdot8 on raw nibbles          // codebook is not two’s-complement i4
fdot2 / E2M1 LUT              // not NVFP4 / mxfp4
WMMA / FP4 MMA
```

Closest existing tile: W4A8 after W8A8 ([sdot4-explore.md](sdot4-explore.md)). Same DOT, different codebook. W4A4 `sdot8` is **not** this.

`amdgcn_perm` = no LDS LUT (same llama.cpp #24438 rule as mxfp4).

## gfx1030 vs their tune

They ship `scripts/build-rdna2.sh`. Hot-path knobs in the README (`RDNA35_NWARPS=2`, FA `KQ_NTHREADS`) are **gfx1151**. Do not copy those onto V620. Wave32 + `sdot4` is legal here; occupancy of *their* MMVQ is unknown until the dump.

Standalone `rocmfp4_hip.cu` dequant is `decode_i8 * ue4m3 → f32` — that is the **unfused** path. The steal is the fused MMVQ/MMQ integer-dot.

## Done-when (side project)

- [ ] ISA: `v_dot4c_i32_i8` (or document scalar FMA if that is what hipcc emitted)
- [ ] `perm` in the expand, not a K$ LUT
- [ ] One paragraph vs W4A8: keep codebook / drop pack

## Sources

- `ggml/rocmfp4/rocmfp4_hip.cu`, `rocmfp4_hip_codebook.cuh`, `rocmfp4.h`
- [silicon/valu.md](../silicon/valu.md)
- Engine pages above
