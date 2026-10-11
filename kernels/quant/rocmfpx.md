---
title: "ROCmFPX — silicon (codebook → perm → sdot4)"
sources:
  - https://github.com/charlie12345/ROCmFPX
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/ggml-org/llama.cpp/blob/6f165c1c64f77024686dc969c3de6f030f274add/ggml/src/ggml-cuda/common.cuh
  - https://github.com/ggml-org/llama.cpp/issues/24438
  - https://github.com/ggml-org/llama.cpp/blob/6f165c1c64f77024686dc969c3de6f030f274add/tools/server/README.md
---
# ROCmFPX — silicon (codebook → perm → sdot4)

Engine: [engine/quant-formats/rocmfpx.md](../../engine/quant-formats/rocmfpx.md). Side project: [engine/alt-engines/llamacpp-rocmfpx.md](../../engine/alt-engines/llamacpp-rocmfpx.md). **Later.** Occupancy still first. No vLLM port.[^rocmfpx]

Source tree: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) `ggml/rocmfp4/`. Opcode on gfx1030 is **not dumped yet** — this page is the expected contract. The V620 side project owns the `.s`.[^rocmfpx]

## Expected inner loop

| Step | Op | Notes |
|---|---|---|
| Load W | packed nibbles, 32 weights / block (17 or 18 B) | FAST = 1× UE4M3; dual-scale = 2× [^rocmfpx] |
| Expand | `__builtin_amdgcn_perm` Codebook10 → i8 | `{0,±1,±2,±3,±4,±6,±8,±10}` — **not** `{−8…7}` [^rocmfpx] |
| Scale | UE4M3 arithmetic → f32 (they rejected a scale LUT) | epilogue or pre-scale into the i8 path [^rocmfpx] |
| A | Q8_1 (`vec_dot_*_q8_1`) | same class as W8A8 A-quant [^rocmfpx] |
| DOT | expect `V_DOT4C_I32_I8` / `ggml_cuda_dp4a` | **confirm on dump.** Fallback is scalar FMA after `rocmfp4_decode_i8` (the standalone dequant kernel does that) [^isa][^lc-common][^rocmfpx] |

```
// keep
nibble → perm codebook → i8x4 → sdot4 vs A8
C *= ue4m3

// drop
sdot8 on raw nibbles          // codebook is not two’s-complement i4
fdot2 / E2M1 LUT              // not NVFP4 / mxfp4
WMMA / FP4 MMA
```

Closest existing tile: W4A8 after W8A8 ([sdot4-explore.md](sdot4-explore.md)). Same DOT, different codebook. W4A4 `sdot8` is **not** this.[^rocmfpx]

`amdgcn_perm` = no LDS LUT (same llama.cpp #24438 rule as mxfp4).[^lc-24438]

## gfx1030 vs their RDNA4 / Strix numbers

They ship `scripts/build-rdna2.sh`. Published “good results” are **RDNA4 / gfx1151 / Vulkan** (WMMA or their tuned FA), not a V620 ISA dump. Do not copy `RDNA35_NWARPS=2`, FA `KQ_NTHREADS`, or R9700 tok/s onto gfx1030.[^rocmfpx]

Wave32 + `sdot4` is the legal V620 compute path. Occupancy of *their* MMVQ is unknown until the dump. Standalone `rocmfp4_hip.cu` dequant is `decode_i8 * ue4m3 → f32` — unfused. Take fused MMVQ/MMQ integer-dot.[^rocmfpx]

## Concurrency is KV, not the codebook

ROCmFP4 only shrinks **weights**. llama-server concurrency is still:[^rocmfpx]

```text
slots = -np
KV_bytes ≈ -np × -c × 2 × n_layer × n_kv_head × d_head × kv_elem
```

`--cache-prompt` / `-cb` reuse a **slot’s** KV for a longer prompt with the same prefix. They do not allocate shared paged blocks. A second system prompt needs another slot or you evict and redo prefill.[^lc-server]

On a 32 GB V620 the lever that changes `-np` is `-ctk`/`-ctv` (and `-c`), not switching Q4_K_M → ROCmFP4. Weight savings of ~10–15% vs Q4_K_M free a little KV, not a farm of slots. 35B-A3B-class FP4 still leaves most of the card in weights + one-context workspace; expect a **handful** of 8k slots unless KV is quantized.[^rocmfpx][^lc-server]

No extra silicon in the prefix-cache path: it is “don’t recompute K/V for the cached prefix,” not a new DOT.[^rocmfpx]

## Done-when (side project)

- [ ] ISA: `v_dot4c_i32_i8` (or document scalar FMA if that is what hipcc emitted)[^isa]
- [ ] `perm` in the expand, not a K$ LUT[^rocmfpx]
- [ ] One paragraph vs W4A8: keep codebook / drop pack[^rocmfpx]
- [ ] Record `-np`/`-c`/`-ctk` that actually fit 32 GB; do not quote RDNA4 tok/s[^rocmfpx][^lc-server]

## Sources

- `ggml/rocmfp4/rocmfp4_hip.cu`, `rocmfp4_hip_codebook.cuh`, `rocmfp4.h`
- [silicon/valu.md](../../silicon/valu.md)
- Engine pages above
- ROCmFPX README / `scripts/build-rdna2.sh`

[^rocmfpx]: charlie12345/ROCmFPX (`ggml/rocmfp4/`, `scripts/build-rdna2.sh`). <https://github.com/charlie12345/ROCmFPX>
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^lc-common]: llama.cpp `ggml/src/ggml-cuda/common.cuh` @ `6f165c1c` — `ggml_cuda_dp4a` (`RDNA2` → `__builtin_amdgcn_sdot4`, `RDNA3/4` → `sudot4`). <https://github.com/ggml-org/llama.cpp/blob/6f165c1c64f77024686dc969c3de6f030f274add/ggml/src/ggml-cuda/common.cuh>
[^lc-24438]: llama.cpp issue #24438 — ROCm/HIP MoE token-gen at ~40% bandwidth on gfx1151 (per-lane nibble LUT serialization). <https://github.com/ggml-org/llama.cpp/issues/24438>
[^lc-server]: llama.cpp `tools/server/README.md` @ `6f165c1c` (`-np` slots, `-c`, `-ctk`/`-ctv`, `--cache-prompt`). <https://github.com/ggml-org/llama.cpp/blob/6f165c1c64f77024686dc969c3de6f030f274add/tools/server/README.md>
