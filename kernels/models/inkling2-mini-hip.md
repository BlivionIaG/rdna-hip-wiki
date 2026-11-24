---
title: "Inkling-2-Mini HIP kernels on gfx1030 — tip `00f106c5`"
updated: 2026-11-24
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d
  - https://github.com/opengfx1030/vllm-rdna/compare/1e1cac5340a9c9af8c9df3db331978548d709b42...00f106c5b035bd6d562267bfee9e4ac020b7055d
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_fuse_attn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_glue_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_linear_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_vit_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_dspark_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_lowp_dot2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_prefill_bm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/_custom_ops.py
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/fuse_chain.py
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/glue.py
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/moe.py
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-attn-kernels.md
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-moe-prefill-bm.md
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-fuse-attn-chain.md
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-port-spec.md
  - https://github.com/opengfx1030/vllm-rdna/commit/9ceaac464b52dc2198c442935319bcd6f8c6157e
  - https://github.com/opengfx1030/vllm-rdna/commit/7eeb0da81b55e97ed04993096e6554885aa41c07
  - https://github.com/opengfx1030/vllm-rdna/commit/42fffe0e8cb6e1fd3b097992e2ee98ec79036d84
  - https://github.com/opengfx1030/vllm-rdna/commit/45a46d5a960e431deaac07fe20f60ba72c60bf43
  - https://github.com/opengfx1030/vllm-rdna/commit/e505d4d43630dab09412c818a730c93687b180b8
  - https://github.com/opengfx1030/vllm-rdna/commit/d5518d47d05cf5ffb7ed260dfdfe035d74151255
  - https://github.com/opengfx1030/vllm-rdna/commit/eefc5ddba941868cf4d06fc84ac7b42b4e175835
  - https://github.com/opengfx1030/vllm-rdna/commit/0fd3e424dcb4f2921242ebe3f6ff8d1b661f05c8
  - https://github.com/opengfx1030/vllm-rdna/commit/0cd91cb62cf538fe404c4152230bdb7602296546
  - https://github.com/opengfx1030/vllm-rdna/commit/d71bddca6e0f9d9f22390b8dde758cf7baf60481
  - https://github.com/opengfx1030/vllm-rdna/commit/268dd22031c125ecf46d60734a0e30733b0f132e
  - https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/moe_prefill_bm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/vllm/models/inkling2_loop/amd/moe.py
  - https://github.com/opengfx1030/vllm-rdna/blob/5ba9a09af6257395022a5e86254368b132fa17dd/vllm/models/inkling2_loop/amd/moe.py
---

# Inkling-2-Mini HIP kernels on gfx1030 — tip `00f106c5`

Native HIP serving kernels for Inkling-2-Mini (`inkling2_loop`) on dest `rdna_extras`, read at tip [`00f106c5`](https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d) (2026-11-23 03:44 Paris, merge of PR #196). Previous watched tip: `1e1cac53`.[^tip][^cmp] Back to the [models index](README.md).

Every kernel here is wave32 and uses `V_DOT2_F32_F16` (`__builtin_amdgcn_fdot2`) as its matrix primitive, with fp32 accumulation; the one exception is the opt-in W4A8 prefill expert kernel, which uses `V_DOT4_I32_I8`.[^attn-head][^pbm-head][^pbm-sdot4] No WMMA, no MFMA. Benchmarks and tok/s are left out on purpose; see the fork's `docs/rdna2/inkling2-benchmarks.md`.

## Build

- Eight new translation units join the **gfx1030-only** source list (`if(VLLM_GPU_ARCHES MATCHES "gfx1030")`): `moe_lowp_dot2_rdna2.cu`, `moe_prefill_bm_rdna2.cu`, `inkling2_attn_rdna2.cu`, `inkling2_dspark_rdna2.cu`, `inkling2_vit_rdna2.cu`, `inkling2_linear_rdna2.cu`, `inkling2_glue_rdna2.cu`, `inkling2_fuse_attn_rdna2.cu`.[^cmake-list]
- `inkling2_vit_rdna2` and `inkling2_linear_rdna2` are compiled at `-O3`, joining `rdna_hint_flex_dense_gemm_rdna2` (RelWithDebInfo would otherwise give `-O2`).[^cmake-o3]
- Device code exists only for `__gfx1030__` (`__HIP__RDNA2__`). Other device passes get empty kernel bodies with the same signature, so a multi-arch build still compiles.[^attn-gate][^glue-gate][^fuse-gate] The attention TU also rejects anything but wave32 with `#error`.[^attn-gate]
- None of the Inkling-2 kernels uses `amdgpu_waves_per_eu`; occupancy comes from `__launch_bounds__` plus VGPR and LDS use (checked by grep over all eight files at the tip).[^tip]

## Occupancy first

Compiler-reported resources are quoted from the fork's committed docs (`-Rpass-analysis=kernel-resource-usage` / build receipts). They are not hardware counters.

| Kernel | Block | VGPR | LDS / CTA | Waves/SIMD (compiler) | Source |
|---|---|---:|---:|---:|---|
| `i2_tile_pipe_kernel<3>` (prefill / verify tile, **default**) | 128 thr | 244 | 32,512 B | 4 | [^doc-tile-res] |
| `i2_tile_kernel<3>` (prior tile kernel, `tile_impl=0`) | 128 thr | 255 | 55,552 B | 2 | [^doc-tile-res] |
| `i2_tile_pipe_kernel<4>` (not a model shape) | 128 thr | 256, 84 VGPRs spilled | 37,376 B | 3 | [^doc-tile-res] |
| `i2_decode_fused_kernel<G,16>` (sliding-window decode, default) | 16 waves = 512 thr | 124, 0 spills | 8.4 KB | 8 | [^doc-fused-res] |
| `nvfp4_prefill_kernel<32,128,128>` | 128 thr | 231 (SGPR 38) | 16,512 B | 4 | [^doc-pbm-res] |
| `mxfp8_prefill_kernel<32,128,256>` | 128 thr | 243 (SGPR 52) | 16,512 B | 4 | [^doc-pbm-res] |
| `nvfp4_i8_prefill_kernel<32,128,256>` (opt-in W4A8) | 128 thr | 234 (SGPR 49) | 9,344 B | 4 | [^doc-pbm-res] |
| `attn_res_fused_kernel` (FUSE_ATTN_CHAIN) | 192 thr per token | 164 | — | — | [^doc-fuse-res] |
| `i2_sconv_kernel` / `i2_state_write_kernel` / `i2_kv_prep_kernel` | 128 thr | 33–41 / 8–10 / 61–74 | 0 | — | [^doc-conv-res] |

- The pipelined tile kernel computes the same fp32 operations in the same order as `i2_tile_kernel`, so the outputs are bit-identical. The source gives the LDS change at G = 3 as "55.5 KB -> 32.5 KB … four CTAs per WGP instead of two".[^attn-pipe]
- The doc labels the WGP residency as inference: 4 × 32,512 B fits the 128 KB of a WGP.[^doc-tile-res]
- Launch bounds by file: attention `128` (sconv, state write, kv prep, split-K decode, combine, both tile kernels) and `NW*32` for the fused decode. Glue: `256`, `kGateWaves*32`, `2*32`, `8*32`, `kRouteThreads = 256`. Linear: `256` (GEMV, f32w, argmax) and `P::T` (GEMM). ViT: `T = 128` (GEMM) and `256`. DSpark: `256`, `128`, `512`. Prefill expert: `NT = 128` and `kMetaThreads = 256`.[^attn-lb][^glue-lb][^linear-lb][^vit-lb][^dspark-lb][^pbm-lb]
- The decode expert GEMMs (`moe_gemm_{nvfp4,mxfp8}_kernel_rdna2`) have **no** `__launch_bounds__`. They run 256 threads (`THREADS_X`) and stage `block_a[BLOCK_SIZE_M][256 + 8]` in LDS, with `block_size_m` 1/2/4/8.[^lowp-geom]

## Attention (`inkling2_attn_rdna2.cu`)

| Path | Geometry | Notes |
|---|---|---|
| Split-K decode (`i2_decode_kernel`, q_len = 1) | grid `(H_kv, ceil(S_A/4), R)`, 128 thr. One wave per (request, kv head, split), 8 lanes per key | K and V stream straight from the paged cache, with **no LDS**. [^attn-head][^attn-launch] |
| Fused decode (`i2_decode_fused_kernel`) | grid `(H_q, 1, R)`, 16 waves (`kI2FusedWaves`). One CTA per (request, q head) | Wave partials merge in LDS (`s_acc[NW][128]` fp32), so there is no workspace and no combine launch. [^attn-fused][^attn-launch] |
| Tile (`i2_tile_kernel`) | grid `(ceil(max_q/16), H_kv*S_B, R)`, 128 thr. 16 query tokens × G heads per CTA, 64-key tiles | sQ / sK rows are 136 halves (16 B aligned, bank skew). Separate sV and sP tiles. [^attn-tile] |
| Pipelined tile (`i2_tile_pipe_kernel`) | same grid | One `sKV` buffer holds K during QK and V during PV. P stays in registers and is handed out by DPP `row_share`. Row max uses DPP `row_xmask`, not `ds_bpermute`. The V tile layout is bank-conflict free. The page table is double-buffered by tile parity. [^attn-pipe] |
| Combine (`i2_combine_kernel`) | grid `(tiles, H_kv, R)`, 128 thr | Runs only when split-K partials exist. [^attn-launch] |

- Constants: head dim 128, `kI2BR = 16` query tokens per tile CTA, `kI2BC = 64` keys per tile, `kI2MaxSplits = 256`, `kI2FusedWaves = 16`.[^attn-const]
- KV page layout: `[num_blocks, H_kv, block_size, 2*128]` fp16, with K at [0,128) and V at [128,256). The KV cache is fp16, so this file has **no KV quant path**.[^attn-head]
- DPP helpers: `i2_row_xor<M>` is `update_dpp` control `0x160 + M` (`row_xmask`), and `i2_row_share<N>` is `0x150 + N` (`row_share`). Both are gfx10+ encodings and stay within a row of 16 lanes.[^attn-dpp]
- Split policy (host): fused decode is selected for sliding-window layers when `VLLM_INKLING2_DECODE_FUSED` is on. Otherwise `splits_decode = min(ceil(256/groups), ceil(keys/16)/2, 256)`. `splits_tile` grows only while the tile grid is under 144 CTAs (cap 16).[^ops-splits]

## Short conv and conv-state ring

- `i2_sconv_kernel`: depthwise causal k = 4 conv over the paged conv state, one block per token, 128 threads × 4 channels.[^attn-sconv]
- The conv state is position-addressed: position p lives in `block_table[r, p >> blk_shift]` at ring row `p % ring` of an `[H, ring, W]` fp16 page. Every ring row has exactly one writer, and `ring >= k - 1 + draft length` keeps the accepted prefix intact after a speculative rejection. Introduced in `42fffe0e`.[^attn-ring][^c-42fffe0e]

## Fused decode chain (`inkling2_fuse_attn_rdna2.cu`)

- `attn_res_fused_kernel`, `__launch_bounds__(256)`, grid `(T, layers)`. It fuses the static short conv into the `attn_res` consumer and runs all the message `attn_res` of a loop pass in one launch. The source states the results are bit-identical.[^fuse-head][^fuse-lb]
- Cross-lane butterfly: `permlanex16` for lane^16, then DPP `row_xmask:8` (`0x168`), `row_xmask:4` (`0x164`), and `quad_perm` `0x4E` / `0xB1`. There is no LDS `ds_bpermute` on this path.[^fuse-xor]

## Glue (`inkling2_glue_rdna2.cu`)

- The `gate_mix` R-split kernel is 8 waves (`kMixWaves = 8`, `kMixRpw = 8`, R = 64). It is taken when `VLLM_INKLING2_FUSE_GLUE` is on, the weights are fp32 and T ≤ 4.[^glue-mix]
- The C++ host reads `VLLM_INKLING2_FUSE_GLUE` once and treats it as default on. The Python side (`glue.py`) also defaults it to on.[^glue-env][^py-glue]
- The router (`inkling2_glue_moe_route`) is 256 threads (8 waves, one expert row per wave). It produces logits, sigmoid, top-k and the `moe_align_block_size` layout in one launch.[^glue-route]

## Dense linear, ViT, DSpark

- `inkling2_linear_rdna2.cu`: GEMV for M ≤ 16 (8 waves, 48 KiB LDS activation chunk budget). Above that, a register-tiled dot2 GEMM picked from 6 designs (BM×BN 128×128, 256×128, 128×64, 64×128, 32×128, 64×64), with `kGemmMinM = 17`.[^linear-const]
- `inkling2_vit_rdna2.cu`: the GEMM is the "D1" design of `rdna_hint_flex_dense_gemm_kernel.cuh`: a 128×128 tile, BK 32, 8×16 accumulators per thread, 128 threads, XOR-swizzled LDS, B-fragment dword rotation and buffer-descriptor loads.[^vit-gemm]
- `inkling2_dspark_rdna2.cu`: head 128, 8 block rows, GQA 3 (24 q / 8 kv), 64-key attention tiles, Markov rank 256. The GEMV stages a 512-K activation chunk in LDS per 8-wave block.[^dspark-const]

## MoE experts

**Decode (`moe_lowp_dot2_rdna2.cu`, from `7eeb0da8`).**[^c-7eeb0da8]
- NVFP4 (`[E, K/8, N]` u32 + e4m3 scale per 16 K + per-expert global) and MXFP8 (`[E, K/4, N]` + UE8M0 per 32 K) weights dequantise on the fly into `V_DOT2`.
- The kernels keep the MXFP4 skeleton (`block_size_m` 1/2/4/8, packed-fp16 CAS atomic-add epilogue).
- gfx1030 has no fp8 conversion instruction, so e4m3 is decoded with integer ops.[^lowp-head]

**v_perm dequant (`e505d4d4`).** `dequant_e2m1_8_fp16_fast` spreads the sign and 3-bit magnitude of four nibbles into bytes. Four `v_perm_b32` (`__builtin_amdgcn_perm`) then place them in the fp16 high bytes. The source says this takes "19 VALU ops per word instead of ~48" and gives bit-identical results.[^lowp-perm][^c-e505d4d4]

**Prefill (`moe_prefill_bm_rdna2.cu`, from `45a46d5a`).**[^c-45a46d5a]
- A block owns up to `kRows = 32` routed rows of one expert and the whole K range, with `kNt = 128` threads × 4 columns.
- The LDS chunk is `kNvfp4Kc = 128` (double-buffered), `kMxfp8Kc = 256`, or `kI8Kc = 256`.
- NVFP4 activations are K-permuted (0,4,1,5,2,6,3,7) inside each 8-K word with `v_perm`, so one shift and mask decodes two nibbles.[^pbm-head][^pbm-perm]
- **W4A8 (opt-in).** The first dot of a chain is inline asm `v_dot4_i32_i8 %0, %1, %2, 0`, the untied VOP3P form with an inline 0. The source says the tied `v_dot4c` chosen for `sdot4(a,b,0)` would need a `v_mov 0`. e2m1 → int8 uses a three-`v_perm` LUT.[^pbm-sdot4]
- **K-tail gate (update @ `d71bddca`).** The NVFP4 prefill kernel now accepts any `K % 32 == 0`: the host check is `size_k % 32 == 0` (was `size_k % kNvfp4Kc`, i.e. 128), the chunk count rounds up (`nchunks = (size_k + KC - 1) / KC`), activation staging zero-fills K offsets at or past `size_k`, and each chunk runs `ng = min(KC, size_k - ck*KC) >> 4` 16-K scale groups in steps of two (32 K per step). `kNvfp4Kc` stays 128 and `__launch_bounds__(NT)` is unchanged. The initial weight/scale prefetch is now predicated on `active`.[^pbm-ktail]
- **Eligibility (update @ `d71bddca`).** `_prefill_supported()` now requires only `hidden % 32 == 0` and `intermediate % 32 == 0` for both formats, so at TP = 2 (I = 576) and TP = 4 (I = 288) the NVFP4 layers also take the 32-row prefill kernels.[^py-moe-ktail] *Historical @ `5ba9a09a`:* NVFP4 needed K % 128 (MXFP8 K % 32), so at TP = 2 / 4 only the MXFP8 layers used them and NVFP4 kept the decode-shaped kernels.[^py-moe-old]
- The fork's §7.4 resource receipt (source of the occupancy table above) was not re-issued in this change, so VGPR / LDS for the K-tail build are not re-reported and those rows stay pinned to `00f106c5`.[^tip2]
- **Python gates.** `VLLM_INKLING2_PREFILL_BM` defaults to `1`, with `VLLM_INKLING2_PREFILL_BM_MIN_TOKENS` = 128. `VLLM_INKLING2_PREFILL_INT8` defaults to `0`.[^py-moe-prefill]

## Env defaults that gate HIP kernels

| Variable | Default | Selects | Read |
|---|---|---|---|
| `VLLM_INKLING2_DECODE_FUSED` | `1` | fused sliding-window decode; `2` also fuses full layers for a single request (opt-in); `0` uses split-K + combine | once at import (baked into graphs) [^ops-env] |
| `VLLM_INKLING2_PREFILL_ATTN_TILE` | `1` | `i2_tile_pipe_kernel`, except sliding layers with ≤ 32 query tokens, which keep `i2_tile_kernel` | once at import [^ops-env][^ops-tile] |
| `VLLM_INKLING2_FUSE_ATTN_CHAIN` | on | `attn_res_fused_kernel` / one-launch messages for batches ≤ 16 tokens | Python module import [^py-fuse] |
| `VLLM_INKLING2_FUSE_GLUE` | on | SwiGLU fused into the w2 expert GEMM (≤ 16 tokens), 8-wave `gate_mix` split | Python + C++ host, once [^py-glue][^glue-env][^py-moe-swiglu] |
| `VLLM_INKLING2_PREFILL_BM` / `_MIN_TOKENS` / `_INT8` | `1` / `128` / `0` | 32-row prefill expert kernels / W4A8 | Python [^py-moe-prefill] |

## Sources

[^tip]: opengfx1030/vllm-rdna commit [`00f106c5`](https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d) (2026-11-23 03:44 Paris): "Inkling-2-Mini: faster prefill attention at depth, DSpark recipes and adaptive K (#196)". The grep for `amdgpu_waves_per_eu` was run over `csrc/rocm/inkling2_*.cu`, `moe_lowp_dot2_rdna2.cu` and `moe_prefill_bm_rdna2.cu` at this SHA, with 0 hits.
[^cmp]: Compare [`1e1cac53...00f106c5`](https://github.com/opengfx1030/vllm-rdna/compare/1e1cac5340a9c9af8c9df3db331978548d709b42...00f106c5b035bd6d562267bfee9e4ac020b7055d).
[^cmake-list]: `CMakeLists.txt` @ `00f106c5` L1780-L1831. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/CMakeLists.txt#L1780-L1831>
[^cmake-o3]: `CMakeLists.txt` @ `00f106c5` L1880-L1890. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/CMakeLists.txt#L1880-L1890>
[^attn-head]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L1-L45 (header: op families, decode/tile split, KV page layout). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L1-L45>
[^attn-gate]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L62-L71 (`__HIP__RDNA2__`, wave32 `#error`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L62-L71>
[^attn-const]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L75-L81. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L75-L81>
[^attn-dpp]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L114-L134. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L114-L134>
[^attn-ring]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L137-L158 (conv state, `I2ConvMeta`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L137-L158>
[^attn-sconv]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L300-L308. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L300-L308>
[^attn-fused]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L948-L953 (`i2_decode_fused_kernel`, `__launch_bounds__(NW * 32)`, LDS arrays). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L948-L953>
[^attn-tile]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L1086-L1114 (`i2_tile_kernel` geometry and LDS). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L1086-L1114>
[^attn-pipe]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L1519-L1569 (`i2_tile_pipe_kernel` comment, `sKV`, double-buffered `s_page`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L1519-L1569>. Introduced in [`0cd91cb6`](https://github.com/opengfx1030/vllm-rdna/commit/0cd91cb62cf538fe404c4152230bdb7602296546) (2026-11-23 02:04 Paris).
[^attn-launch]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5` L2640-L2665 (grids and launch macro). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu#L2640-L2665>
[^attn-lb]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `00f106c5`: `__launch_bounds__(128)` at L300, L336, L354, L508, L2063; `kI2TileThreads` (128) at L1095, L1547; `NW * 32` at L949. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_attn_rdna2.cu>
[^doc-tile-res]: `docs/rdna2/inkling2-attn-kernels.md` @ `00f106c5` §12.2 resource table L1007-L1018. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-attn-kernels.md#L1007-L1018>
[^doc-fused-res]: `docs/rdna2/inkling2-attn-kernels.md` @ `00f106c5` §11.2 L726 ("124 VGPRs (8 waves per SIMD), 0 spills, 8.4 KB LDS"). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-attn-kernels.md#L726>. Kernel introduced in [`eefc5ddb`](https://github.com/opengfx1030/vllm-rdna/commit/eefc5ddba941868cf4d06fc84ac7b42b4e175835) (2026-11-22 21:18 Paris).
[^doc-pbm-res]: `docs/rdna2/inkling2-moe-prefill-bm.md` @ `00f106c5` §7.4 L380-L388. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-moe-prefill-bm.md#L380-L388>
[^doc-fuse-res]: `docs/rdna2/inkling2-fuse-attn-chain.md` @ `00f106c5` L117. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-fuse-attn-chain.md#L117>
[^doc-conv-res]: `docs/rdna2/inkling2-port-spec.md` @ `00f106c5` L862 (six-arch compile row). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/docs/rdna2/inkling2-port-spec.md#L862>
[^ops-splits]: `vllm/_custom_ops.py` @ `00f106c5` L2097-L2135 (`inkling2_attention_splits`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/_custom_ops.py#L2097-L2135>
[^ops-env]: `vllm/_custom_ops.py` @ `00f106c5` L2056-L2077. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/_custom_ops.py#L2056-L2077>
[^ops-tile]: `vllm/_custom_ops.py` @ `00f106c5` L2303-L2313 (`tile_impl` selection, `small_sliding`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/_custom_ops.py#L2303-L2313>
[^c-42fffe0e]: opengfx1030/vllm-rdna commit [`42fffe0e`](https://github.com/opengfx1030/vllm-rdna/commit/42fffe0e8cb6e1fd3b097992e2ee98ec79036d84) (2026-11-22 09:29 Paris): "feat(inkling2): position-addressed fp16 conv-state ring …".
[^fuse-head]: `csrc/rocm/inkling2_fuse_attn_rdna2.cu` @ `00f106c5` L1-L35. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_fuse_attn_rdna2.cu#L1-L35>. Introduced in [`0fd3e424`](https://github.com/opengfx1030/vllm-rdna/commit/0fd3e424dcb4f2921242ebe3f6ff8d1b661f05c8) (2026-11-22 22:21 Paris).
[^fuse-gate]: `csrc/rocm/inkling2_fuse_attn_rdna2.cu` @ `00f106c5` L34-L36, L51-L53. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_fuse_attn_rdna2.cu#L34-L53>
[^fuse-xor]: `csrc/rocm/inkling2_fuse_attn_rdna2.cu` @ `00f106c5` L61-L83. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_fuse_attn_rdna2.cu#L61-L83>
[^fuse-lb]: `csrc/rocm/inkling2_fuse_attn_rdna2.cu` @ `00f106c5` L750 and L823-L830. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_fuse_attn_rdna2.cu#L750-L830>
[^glue-gate]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `00f106c5` L42-L44, L63-L65. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_glue_rdna2.cu#L42-L65>
[^glue-lb]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `00f106c5` L69-L85 (constants) and launch bounds at L430, L547, L875, L883, L973, L998, L1032, L1069, L1402. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_glue_rdna2.cu#L69-L85>
[^glue-mix]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `00f106c5` L84-L85, L881-L888, L1847-L1848. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_glue_rdna2.cu#L1847-L1848>. Introduced in [`d5518d47`](https://github.com/opengfx1030/vllm-rdna/commit/d5518d47d05cf5ffb7ed260dfdfe035d74151255) (2026-11-22 19:42 Paris).
[^glue-env]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `00f106c5` L1469-L1477 (`fuse_glue()`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_glue_rdna2.cu#L1469-L1477>
[^glue-route]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `00f106c5` L28-L31, L73. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_glue_rdna2.cu#L28-L73>
[^linear-const]: `csrc/rocm/inkling2_linear_rdna2.cu` @ `00f106c5` L1-L24 (header), L44-L46, L870-L874. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_linear_rdna2.cu#L870-L874>
[^linear-lb]: `csrc/rocm/inkling2_linear_rdna2.cu` @ `00f106c5` L198, L418, L665, L769, L810. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_linear_rdna2.cu#L198>
[^vit-gemm]: `csrc/rocm/inkling2_vit_rdna2.cu` @ `00f106c5` L35-L39, L220-L243. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_vit_rdna2.cu#L220-L243>
[^vit-lb]: `csrc/rocm/inkling2_vit_rdna2.cu` @ `00f106c5` L176, L235, L470, L500, L576, L602, L628, L670. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_vit_rdna2.cu#L235>
[^dspark-const]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `00f106c5` L48-L56, L449-L460. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_dspark_rdna2.cu#L48-L56>
[^dspark-lb]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `00f106c5` L453, L552, L647, L789, L844, L884, L946, L968, L1142. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/inkling2_dspark_rdna2.cu#L453>
[^c-7eeb0da8]: opengfx1030/vllm-rdna commit [`7eeb0da8`](https://github.com/opengfx1030/vllm-rdna/commit/7eeb0da81b55e97ed04993096e6554885aa41c07) (2026-11-21 23:24 Paris): "feat(inkling2): Inkling-2-Mini text model, NVFP4/MXFP8 RDNA2 MoE kernels, tiny harness". Native kernels added in [`9ceaac46`](https://github.com/opengfx1030/vllm-rdna/commit/9ceaac464b52dc2198c442935319bcd6f8c6157e) (2026-11-22 07:32 Paris).
[^lowp-head]: `csrc/rocm/moe_lowp_dot2_rdna2.cu` @ `00f106c5` L1-L25 and L54-L65. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_lowp_dot2_rdna2.cu#L1-L65>
[^lowp-geom]: `csrc/rocm/moe_lowp_dot2_rdna2.cu` @ `00f106c5` L41-L42 (`BLOCK_KN_SIZE`, `THREADS_X` 256), L189, L219 (`block_a`), L331-L335 (grid), L359-L376 (`block_size_m` 1/2/4/8). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_lowp_dot2_rdna2.cu#L189-L219>
[^lowp-perm]: `csrc/rocm/moe_lowp_dot2_rdna2.cu` @ `00f106c5` L67-L98. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_lowp_dot2_rdna2.cu#L67-L98>
[^c-e505d4d4]: opengfx1030/vllm-rdna commit [`e505d4d4`](https://github.com/opengfx1030/vllm-rdna/commit/e505d4d43630dab09412c818a730c93687b180b8) (2026-11-22 19:42 Paris): "inkling2: exact cheaper NVFP4 dequant in the decode expert GEMM (v_perm byte placement, dword scale loads)".
[^c-45a46d5a]: opengfx1030/vllm-rdna commit [`45a46d5a`](https://github.com/opengfx1030/vllm-rdna/commit/45a46d5a960e431deaac07fe20f60ba72c60bf43) (2026-11-22 19:41 Paris): "Add the Inkling-2 prefill expert GEMM kernels".
[^pbm-head]: `csrc/rocm/moe_prefill_bm_rdna2.cu` @ `00f106c5` L1-L33, L52-L57. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_prefill_bm_rdna2.cu#L1-L57>
[^pbm-perm]: `csrc/rocm/moe_prefill_bm_rdna2.cu` @ `00f106c5` L84-L92 (`perm_k8`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_prefill_bm_rdna2.cu#L84-L92>
[^pbm-sdot4]: `csrc/rocm/moe_prefill_bm_rdna2.cu` @ `00f106c5` L115-L144. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_prefill_bm_rdna2.cu#L115-L144>
[^pbm-lb]: `csrc/rocm/moe_prefill_bm_rdna2.cu` @ `00f106c5` L216, L234, L383, L520, L537, L770, L925-L927. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/moe_prefill_bm_rdna2.cu#L216-L235>
[^py-fuse]: `vllm/models/inkling2_loop/amd/fuse_chain.py` @ `00f106c5` L7-L31. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/fuse_chain.py#L7-L31>
[^py-glue]: `vllm/models/inkling2_loop/amd/glue.py` @ `00f106c5` L32-L38. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/glue.py#L32-L38>
[^py-moe-swiglu]: `vllm/models/inkling2_loop/amd/moe.py` @ `00f106c5` L625-L631 (`_FUSE_SWIGLU_MAX_TOKENS = 16`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/moe.py#L625-L631>
[^py-moe-prefill]: `vllm/models/inkling2_loop/amd/moe.py` @ `00f106c5` L633-L651 (`_prefill_switches`). <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/models/inkling2_loop/amd/moe.py#L633-L651>
[^tip2]: opengfx1030/vllm-rdna commit [`d71bddca`](https://github.com/opengfx1030/vllm-rdna/commit/d71bddca6e0f9d9f22390b8dde758cf7baf60481) (2026-11-24 12:01 Paris): "Merge #205: 32-row NVFP4 prefill K-tails for TP=2/TP=4 and TP=2 MAXBAT=320", from [`268dd220`](https://github.com/opengfx1030/vllm-rdna/commit/268dd22031c125ecf46d60734a0e30733b0f132e). Compare [`5ba9a09a...d71bddca`](https://github.com/opengfx1030/vllm-rdna/compare/5ba9a09af6257395022a5e86254368b132fa17dd...d71bddca6e0f9d9f22390b8dde758cf7baf60481); `docs/rdna2/inkling2-moe-prefill-bm.md` changed only in its L56-L67 hunk.
[^pbm-ktail]: `csrc/rocm/moe_prefill_bm_rdna2.cu` @ `d71bddca` L25, L54, L216, L245, L267, L280, L346, L351-L354, L927. <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/moe_prefill_bm_rdna2.cu#L245-L354>
[^py-moe-ktail]: `vllm/models/inkling2_loop/amd/moe.py` @ `d71bddca` L827-L831. <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/vllm/models/inkling2_loop/amd/moe.py#L827-L831>
[^py-moe-old]: `vllm/models/inkling2_loop/amd/moe.py` @ `5ba9a09a` `_prefill_supported()` (`align = 32 if self.quant_format == "mxfp8" else 128`). <https://github.com/opengfx1030/vllm-rdna/blob/5ba9a09af6257395022a5e86254368b132fa17dd/vllm/models/inkling2_loop/amd/moe.py#L827-L833>
