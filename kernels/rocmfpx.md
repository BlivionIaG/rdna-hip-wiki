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

## gfx1030 vs their RDNA4 / Strix numbers

They ship `scripts/build-rdna2.sh`. Published “good results” are **RDNA4 / gfx1151 / Vulkan** (WMMA or their tuned FA), not a V620 ISA dump. Do not copy `RDNA35_NWARPS=2`, FA `KQ_NTHREADS`, or R9700 tok/s onto gfx1030.

Wave32 + `sdot4` is the legal V620 compute path. Occupancy of *their* MMVQ is unknown until the dump. Standalone `rocmfp4_hip.cu` dequant is `decode_i8 * ue4m3 → f32` — unfused. Take fused MMVQ/MMQ integer-dot.

## Concurrency is KV, not the codebook

ROCmFP4 only shrinks **weights**. llama-server concurrency is still:

```text
slots = -np
KV_bytes ≈ -np × -c × 2 × n_layer × n_kv_head × d_head × kv_elem
```

`--cache-prompt` / `-cb` reuse a **slot’s** KV for a longer prompt with the same prefix. They do not allocate shared paged blocks. A second system prompt needs another slot or you evict and redo prefill.

On a 32 GB V620 the lever that changes `-np` is `-ctk`/`-ctv` (and `-c`), not switching Q4_K_M → ROCmFP4. Weight savings of ~10–15% vs Q4_K_M free a little KV, not a farm of slots. 35B-A3B-class FP4 still leaves most of the card in weights + one-context workspace; expect a **handful** of 8k slots unless KV is quantized.

No extra silicon in the prefix-cache path: it is “don’t recompute K/V for the cached prefix,” not a new DOT.

## Done-when (side project)

- [ ] ISA: `v_dot4c_i32_i8` (or document scalar FMA if that is what hipcc emitted)
- [ ] `perm` in the expand, not a K$ LUT
- [ ] One paragraph vs W4A8: keep codebook / drop pack
- [ ] Record `-np`/`-c`/`-ctk` that actually fit 32 GB; do not quote RDNA4 tok/s

## Sources

- `ggml/rocmfp4/rocmfp4_hip.cu`, `rocmfp4_hip_codebook.cuh`, `rocmfp4.h`
- [silicon/valu.md](../silicon/valu.md)
- Engine pages above
- ROCmFPX README / `scripts/build-rdna2.sh`
