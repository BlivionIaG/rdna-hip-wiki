---
title: "Inkling-2-Mini HIP kernels on gfx1030 — tip `00f106c5`"
updated: 2026-11-25
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
  - https://github.com/opengfx1030/vllm-rdna/commit/f22c794a00e609fc8c03f35d374c5258f8e5187c
  - https://github.com/opengfx1030/vllm-rdna/commit/c1ce3ec355f946849fee47be9fbae111e9a6fcbc
  - https://github.com/opengfx1030/vllm-rdna/compare/d71bddca6e0f9d9f22390b8dde758cf7baf60481...f22c794a00e609fc8c03f35d374c5258f8e5187c
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/docs/rdna2/inkling2-decode-glue.md
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/tests/kernels/core/test_inkling2_glue_rdna2.py
  - https://github.com/llvm/llvm-project/blob/278221a5b0be60d025312c5e007a4c4dc01eb59a/llvm/lib/Target/AMDGPU/SIDefines.h
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/opengfx1030/vllm-rdna/commit/ba9b43bcd479acc4bddcc98a375a31152279a872
  - https://github.com/opengfx1030/vllm-rdna/commit/08a30fadda13cc691ca22f5aacd7cd25ab2d21ed
  - https://github.com/opengfx1030/vllm-rdna/compare/f22c794a00e609fc8c03f35d374c5258f8e5187c...ba9b43bcd479acc4bddcc98a375a31152279a872
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_dspark_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/vllm/models/inkling2_loop/amd/dspark.py
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/vllm/models/inkling2_loop/amd/dspark.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/docs/rdna2/inkling2-dspark.md
  - https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html
  - https://github.com/opengfx1030/vllm-rdna/commit/ea8aa2418e6373de1537fa56e4ae0911ef6162a6
  - https://github.com/opengfx1030/vllm-rdna/compare/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa...ea8aa2418e6373de1537fa56e4ae0911ef6162a6
  - https://github.com/opengfx1030/vllm-rdna/commit/97e70270d045ecd80e65c18a6de99d603b5a35fb
  - https://github.com/opengfx1030/vllm-rdna/commit/fae1b90d22168965dea32f8c8138e348e553cdf7
  - https://github.com/opengfx1030/vllm-rdna/commit/e31906640fb3d3e5f0403af6868b553d17789d18
  - https://github.com/opengfx1030/vllm-rdna/commit/56587204cc664ecb75957e46ab42368789852029
  - https://github.com/opengfx1030/vllm-rdna/commit/6e0dbd586bdba484b761e65cb6cf4b279b22fe0e
  - https://github.com/opengfx1030/vllm-rdna/commit/3aee7f0a76337e56e460077535cc5336a45ffaea
  - https://github.com/opengfx1030/vllm-rdna/commit/1850737d284c54160487479857d38581900ea29c
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_glue_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_dspark_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/moe_lowp_dot2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/rdna_expert_cache.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/attention.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/moe.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/glue.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/model.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/loopcd.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/_custom_ops.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/tools/rdna/recipes/inkling2-mini-tp1-offload.env
  - https://github.com/opengfx1030/vllm-rdna/blob/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa/csrc/rocm/inkling2_attn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa/csrc/rocm/moe_lowp_dot2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa/vllm/models/inkling2_loop/amd/moe.py
---

# Inkling-2-Mini HIP kernels on gfx1030 — tip `00f106c5`

Native HIP serving kernels for Inkling-2-Mini (`inkling2_loop`) on dest `rdna_extras`, read at tip [`00f106c5`](https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d) (2026-11-23 03:44 Paris, merge of PR #196). Previous watched tip: `1e1cac53`.[^tip][^cmp] Glue section updated at `f22c794a` (2026-11-24).[^tip3] DSpark draft-attention section updated at `ba9b43bc` (2026-11-24).[^tip4] Attention decode, MoE decode tiles, glue, DSpark GEMM/argmax and env gates updated at `ea8aa241` (2026-11-25), see [Update @ `ea8aa241`](#update--ea8aa241-tp1-gqa-fused-decode-nvfp4-tile-variant-attn_res_sconv-loopcd-2026-11-25).[^e5-tip] Back to the [models index](README.md).

Every kernel here is wave32 and uses `V_DOT2_F32_F16` (`__builtin_amdgcn_fdot2`) as its matrix primitive, with fp32 accumulation; the one exception is the opt-in W4A8 prefill expert kernel, which uses `V_DOT4_I32_I8`.[^attn-head][^pbm-head][^pbm-sdot4] No WMMA, no MFMA. Benchmarks and tok/s are left out on purpose; see the fork's `docs/rdna2/inkling2-benchmarks.md`.

## Build

- Eight new translation units join the **gfx1030-only** source list (`if(VLLM_GPU_ARCHES MATCHES "gfx1030")`): `moe_lowp_dot2_rdna2.cu`, `moe_prefill_bm_rdna2.cu`, `inkling2_attn_rdna2.cu`, `inkling2_dspark_rdna2.cu`, `inkling2_vit_rdna2.cu`, `inkling2_linear_rdna2.cu`, `inkling2_glue_rdna2.cu`, `inkling2_fuse_attn_rdna2.cu`.[^cmake-list]
- `inkling2_vit_rdna2` and `inkling2_linear_rdna2` are compiled at `-O3`, joining `rdna_hint_flex_dense_gemm_rdna2` (RelWithDebInfo would otherwise give `-O2`).[^cmake-o3]
- Device code exists only for `__gfx1030__` (`__HIP__RDNA2__`). Other device passes get empty kernel bodies with the same signature, so a multi-arch build still compiles.[^attn-gate][^glue-gate][^fuse-gate] The attention TU also rejects anything but wave32 with `#error`.[^attn-gate]
- None of the Inkling-2 kernels uses `amdgpu_waves_per_eu`; occupancy comes from `__launch_bounds__` plus VGPR and LDS use (checked by grep over all eight files at the tip).[^tip] Still 0 hits for `waves_per_eu` in the attention, glue, DSpark and `moe_lowp_dot2` files at `ea8aa241`.[^e5-tip]

## Occupancy first

Compiler-reported resources are quoted from the fork's committed docs (`-Rpass-analysis=kernel-resource-usage` / build receipts). They are not hardware counters.

| Kernel | Block | VGPR | LDS / CTA | Waves/SIMD (compiler) | Source |
|---|---|---:|---:|---:|---|
| `i2_tile_pipe_kernel<3>` (prefill / verify tile, **default**) | 128 thr | 244 | 32,512 B | 4 | [^doc-tile-res] |
| `i2_tile_kernel<3>` (prior tile kernel, `tile_impl=0`) | 128 thr | 255 | 55,552 B | 2 | [^doc-tile-res] |
| `i2_tile_pipe_kernel<4>` (not a model shape) | 128 thr | 256, 84 VGPRs spilled | 37,376 B | 3 | [^doc-tile-res] |
| *Historical @ `00f106c5`…`1fdcb3a6`:* `i2_decode_fused_kernel<G,16>` (sliding-window decode, default then). Renamed `i2_decode_fused_nh_kernel` and extended to q ≤ 8 at `ea8aa241`; this receipt was not re-issued | 16 waves = 512 thr | 124, 0 spills | 8.4 KB | 8 | [^doc-fused-res][^e5-nh] |
| `i2_decode_fused_gqa_kernel<G,NW>` (new @ `ea8aa241`, default fused decode / verify ≤ 8 q) | `NW*32` thr, NW = 16 or 8 | not reported (source: 34-VGPR half-tiles, "zero spills up to G = 4") | 25,152 B (G = 3, NW = 16); 33,536 B (G = 4, NW = 16); 12,576 B (G = 3, NW = 8), arithmetic from the declarations | not reported | [^e5-gqa-lds][^e5-gqa-head] |
| `attn_res_sconv_kernel<Q,IT,NORM,MODE>` (new @ `ea8aa241`, TP = 1) | ≤ 256 thr (`row_launch_shape`) | source comment: ~104 (vs 164 with all loads overlapped) | `s_red[8][kMaxStates·(Q+1)]` fp32 with `kMaxStates = 64`: 4,096 B at Q = 1 (arithmetic; same array as `attn_res`) | not reported | [^e5-sconv-vgpr][^e5-sconv-lb] |
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
| *Historical @ `00f106c5`…`1fdcb3a6`:* Fused decode (`i2_decode_fused_kernel`) | grid `(H_q, 1, R)`, 16 waves (`kI2FusedWaves`). One CTA per (request, q head) | Wave partials merge in LDS (`s_acc[NW][128]` fp32), so there is no workspace and no combine launch. [^attn-fused][^attn-launch] At `ea8aa241` this is `i2_decode_fused_nh_kernel`, grid `(H_q, fused_max_q, R)` (see update). [^e5-nh] |
| GQA fused decode / verify (`i2_decode_fused_gqa_kernel`, `ea8aa241`) | grid `(H_kv, fused_max_q, R)`, `NW*32` thr. One CTA per (kv head, query offset, request) owns all G q heads | Removes the G-way repeated K/V/`proj_t` reads of the nh grid; LDS `s_acc[G][NW][128]`. [^e5-gqa-head][^e5-gqa-lds] |
| Tile (`i2_tile_kernel`) | grid `(ceil(max_q/16), H_kv*S_B, R)`, 128 thr. 16 query tokens × G heads per CTA, 64-key tiles | sQ / sK rows are 136 halves (16 B aligned, bank skew). Separate sV and sP tiles. [^attn-tile] |
| Pipelined tile (`i2_tile_pipe_kernel`) | same grid | One `sKV` buffer holds K during QK and V during PV. P stays in registers and is handed out by DPP `row_share`. Row max uses DPP `row_xmask`, not `ds_bpermute`. The V tile layout is bank-conflict free. The page table is double-buffered by tile parity. [^attn-pipe] |
| Combine (`i2_combine_kernel`) | grid `(tiles, H_kv, R)`, 128 thr | Runs only when split-K partials exist. [^attn-launch] |

- Constants: head dim 128, `kI2BR = 16` query tokens per tile CTA, `kI2BC = 64` keys per tile, `kI2MaxSplits = 256`, `kI2FusedWaves = 16`.[^attn-const]
- KV page layout: `[num_blocks, H_kv, block_size, 2*128]` fp16, with K at [0,128) and V at [128,256). The KV cache is fp16, so this file has **no KV quant path**.[^attn-head]
- DPP helpers: `i2_row_xor<M>` is `update_dpp` control `0x160 + M` (`row_xmask`), and `i2_row_share<N>` is `0x150 + N` (`row_share`). Both are gfx10+ encodings and stay within a row of 16 lanes.[^attn-dpp]
- *Historical @ `00f106c5` (policy text unchanged in `inkling2_attention_splits` at `ea8aa241`, but serving now calls `attention_splits_hip` first, see update):* Split policy (host): fused decode is selected for sliding-window layers when `VLLM_INKLING2_DECODE_FUSED` is on. Otherwise `splits_decode = min(ceil(256/groups), ceil(keys/16)/2, 256)`. `splits_tile` grows only while the tile grid is under 144 CTAs (cap 16).[^ops-splits]

## Short conv and conv-state ring

- `i2_sconv_kernel`: depthwise causal k = 4 conv over the paged conv state, one block per token, 128 threads × 4 channels.[^attn-sconv]
- The conv state is position-addressed: position p lives in `block_table[r, p >> blk_shift]` at ring row `p % ring` of an `[H, ring, W]` fp16 page. Every ring row has exactly one writer, and `ring >= k - 1 + draft length` keeps the accepted prefix intact after a speculative rejection. Introduced in `42fffe0e`.[^attn-ring][^c-42fffe0e]

## Fused decode chain (`inkling2_fuse_attn_rdna2.cu`)

- `attn_res_fused_kernel`, `__launch_bounds__(256)`, grid `(T, layers)`. It fuses the static short conv into the `attn_res` consumer and runs all the message `attn_res` of a loop pass in one launch. The source states the results are bit-identical.[^fuse-head][^fuse-lb]
- Cross-lane butterfly: `permlanex16` for lane^16, then DPP `row_xmask:8` (`0x168`), `row_xmask:4` (`0x164`), and `quad_perm` `0x4E` / `0xB1`. There is no LDS `ds_bpermute` on this path.[^fuse-xor]

## Glue (`inkling2_glue_rdna2.cu`)

- The `gate_mix` R-split kernel is 8 waves (`kMixWaves = 8`, `kMixRpw = 8`, R = 64). It is taken when `VLLM_INKLING2_FUSE_GLUE` is on, the weights are fp32 and **T ≤ 16** (since `56587204`, at `ea8aa241`).[^e5-mix] *Historical @ `f22c794a` … `1fdcb3a6`:* T ≤ 8. *Historical @ `00f106c5` … `d71bddca`:* the limit was T ≤ 4.[^glue-mix][^g2-split][^g2-old-split]
- The C++ host reads `VLLM_INKLING2_FUSE_GLUE` once and treats it as default on. The Python side (`glue.py`) also defaults it to on.[^glue-env][^py-glue]
- The router (`inkling2_glue_moe_route`) is 256 threads (8 waves, one expert row per wave). It produces logits, sigmoid, top-k and the `moe_align_block_size` layout in one launch.[^glue-route] Since `f22c794a` this is the path for T ≥ 5 only (or any T with `FUSE_GLUE=0`); T ≤ 4 takes `moe_route_small_kernel` (below). *Historical @ `d71bddca`:* this one kernel served every T.[^g2-host-route][^g2-old-grid]

### Glue update @ `f22c794a`: DPP reductions, small-T router, `gate_mix_split` to T ≤ 8 (2026-11-24)

Dest moved `d71bddca` → [`f22c794a`](https://github.com/opengfx1030/vllm-rdna/commit/f22c794a00e609fc8c03f35d374c5258f8e5187c) (Merge #206, content commit `c1ce3ec3`). The only kernel file in the delta is `inkling2_glue_rdna2.cu`; the rest is the fork doc (§8), the glue test and the JIT wrapper.[^tip3]

**Gate scope (read from the code, not the doc).** `fuse_glue()` is unchanged: `VLLM_INKLING2_FUSE_GLUE` unset means on, `0` or empty means off, read once.[^g2-env] In the C++ only two call sites test it: the `gate_mix` split choice and the small-T router branch.[^g2-split][^g2-host-route] The DPP helpers and the `attn_res` rewrite are compile-time on `__HIP__RDNA2__` and are **not** behind the env var, so `FUSE_GLUE=0` does not bring back the `__shfl_xor` reductions or the old `attn_res` barriers. The fork doc's "`=0` restores the previous paths" holds for the router and `gate_mix` split only.[^g2-xor][^g2-doc]

**Wave32 cross-lane butterfly (`xor_lane_i32<O>`, `xor_lane<O>`, `wave_sum`, `wave_max`, new `wave_sum_n<N>`).**[^g2-xor]
- Under `#if defined(__HIP__RDNA2__)`: O = 16 is `__builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false)`; O = 8 / 4 / 2 / 1 are `__builtin_amdgcn_update_dpp(x, x, ctrl, 0xF, 0xF, false)` with `ctrl` = `0x168` / `0x164` / `0x4E` / `0xB1`. Other device passes keep `__shfl_xor(x, O)`. The source comment calls the butterfly "bit-identical to shfl_xor".
- Encodings: in LLVM's `DppCtrl` enum `ROW_XMASK0 = 0x160` … `ROW_XMASK_LAST = 0x16F` and `QUAD_PERM` is `0x00`–`0xFF`, so `0x168` / `0x164` are `row_xmask:8` / `row_xmask:4`.[^llvm-dppctrl] `quad_perm` packs four 2-bit source lanes, lane 0 in the low bits: `0x4E` is `[2,3,0,1]` (lane ^ 2) and `0xB1` is `[1,0,3,2]` (lane ^ 1).[^isa70648-g2] With identity selects, `v_permlanex16_b32` has every lane read the same lane index in the other 16-lane row, i.e. lane ^ 16.[^isa70648-g2] Same shape as the fused-chain butterfly above and the full-wave Take row in [dpp-crosslane-occupancy.md](../../silicon/occupancy/dpp-crosslane-occupancy.md).[^fuse-xor]
- `wave_sum_n<N>` runs the five steps across an array of N values one step at a time, so N independent reductions are interleaved.[^g2-xor]
- Users in this file: `block_sum` (so `rms_norm`) calls `wave_sum`; `attn_res`, the gate kernels and the routers call `wave_sum` / `wave_sum_n` / `wave_max`; `route_topk_wave`'s arg-max runs the butterfly on both the value and the expert id (`xor_lane` + `xor_lane_i32`).[^g2-xor][^g2-topk]
- `route_topk_wave` now broadcasts the selected weight with `__shfl(mine, bi & 31)` from the owning lane, not a `wave_sum` over a one-hot vector.[^g2-topk]
- *Historical @ `d71bddca`:* `wave_sum` / `wave_max` were a `for (o = 16; o > 0; o >>= 1) __shfl_xor` loop on all archs (each step a `ds_bpermute` + `lgkmcnt` wait on gfx1030, see the DPP page), and the top-k arg-max used the same `__shfl_xor` loop.[^g2-old-xor][^g2-old-topk]

**`attn_res` (`attn_res_body`, still `__launch_bounds__(256)`).**[^g2-attn][^g2-attn-lb]
- Pass 1 issues `fetch()` for all register-resident states (`kNReg = 6`) before reducing any of them, then reduces with `wave_sum_n<Q+1>`.
- `s_tot[kMaxStates*(Q+1)]` and `s_coef[Q][kMaxStates]` are gone. After the one barrier that publishes `s_red`, **every wave** sums `s_red` across waves itself and computes `rsqrt`, the softmax and the coefficients in registers (two states per lane). Pass 2 reads each coefficient with `__shfl(coef[q][u], j)`. That is `kMaxStates·(2Q+1)·4 B` less static LDS per CTA (`kMaxStates = 64`), and the softmax section drops from three barriers to one.
- RMSNorm epilogue (`NORM`): `wave_sum_n<Q>` replaces the barrier before the `s_nrm` write, and the `norm_w` row(s) are loaded into registers before the single remaining barrier. NORM build: five barriers become two; non-NORM: three become one.
- Occupancy: block size and launch bounds are unchanged and the static LDS only shrinks. The cost moves to VGPRs (`coef[Q][2]`, `tot_q[Q][2]` and `nw[Q][ITER]` held live). The fork did not publish a new VGPR count for this kernel, so none is given here.[^g2-doc]
- *Historical @ `d71bddca`:* threads striped over `s_tot`, wave 0 alone did the softmax into `s_coef`, and pass 2 read `s_coef` from LDS, with three barriers in that section and two in the norm epilogue.[^g2-old-attn]

**`moe_route_small_kernel<WT, TB>` (T ≤ 4, `FUSE_GLUE` on).**[^g2-small][^g2-host-route]
- `__launch_bounds__(kRouteSmallThreads)` with `kRouteSmallThreads = 128` (4 waves); `TB` is 1 / 2 / 4 by T. Launched with **0 B dynamic LDS**; grid = `nzero + E`, i.e. one CTA per expert plus the zero-fill CTAs (`NT = 1`, `NEG = E`).
- Each expert CTA's 128 threads split H for that expert's row: fp32 weights via `fmaf` on fp16 `x`, fp16 weights via four `__builtin_amdgcn_fdot2` per `uint4`. `x` is read straight from global memory (no staging in LDS). The result is `wave_sum_n<TB>`, then a 4-way sum through `s_part[4][TB]`.
- One device-wide "barrier": each CTA writes its logits, `__threadfence()`, then `atomicAdd` on `counters[0]`. The CTA that sees `E - 1` does top-k (`route_topk_wave`, which now also writes ids as bytes into `s_flat_ids[8*16]`) and the align in LDS. Each thread owns experts `tid` and `tid + 128` (so E ≤ 255 is covered), counts with a linear scan, prefix-sums with `__shfl_up`, and writes `expert_ids` / `sorted_ids` / `post_pad`. No histogram, ballot or sort pass. The last arriver resets `counters[0]`.
- Static LDS only: `s_part`, `s_flag`, `s_flat_ids`, `s_ws[8]`, plus `route_topk_wave`'s `s_sel_id` / `s_sel_w` (`[8][16]`, sized for 8 waves even though this kernel has 4).
- The host branch is capped at T ≤ 4. The source says that at T = 5..8 "per-CTA register pressure and the serial align loop make the 16-CTA path faster". The device-side comment still says "T <= 8"; the host gate is the one that applies.
- The large path (T ≥ 5) is unchanged: `kRouteThreads = 256`, `NEG = clamp(ceil(kMinCtas/NT), 1, ceil(E/8))` (16 for E = 128), dynamic LDS `max(tb·H·xsz, kAlignSmemBytes)`.[^g2-host-route]

**`gate_mix_split_kernel` to T ≤ 8.** The split condition is now `fuse_glue() && !w16 && T <= 8 && R == kMixWaves * kMixRpw`, with `tb = gate_tb(T, 4)`. So T = 5..8 runs `ntiles = 2` tiles of TB = 4 instead of falling back to the 1-wave `gate_mix` kernel. Block (`kMixWaves*32` = 256) and launch bounds are unchanged.[^g2-split][^g2-gate-tb]

**Qualification (fork doc, not re-run here).** §8 reports the glue oracle `test_inkling2_glue_rdna2.py` passing with both `FUSE_GLUE=1` and `=0`, 5/5 hand mutants killed, and a 6-arch `archdiff` with no differences outside gfx1030. The test adds an `E = 255, K = 16, T = 2` route case and runs the repeated-launch race test at T = 1, 3, 4, 8, 17.[^g2-doc][^g2-test] Microbench and serving numbers are left in the fork doc on purpose.

## Dense linear, ViT, DSpark

- `inkling2_linear_rdna2.cu`: GEMV for M ≤ 16 (8 waves, 48 KiB LDS activation chunk budget). Above that, a register-tiled dot2 GEMM picked from 6 designs (BM×BN 128×128, 256×128, 128×64, 64×128, 32×128, 64×64), with `kGemmMinM = 17`.[^linear-const]
- `inkling2_vit_rdna2.cu`: the GEMM is the "D1" design of `rdna_hint_flex_dense_gemm_kernel.cuh`: a 128×128 tile, BK 32, 8×16 accumulators per thread, 128 threads, XOR-swizzled LDS, B-fragment dword rotation and buffer-descriptor loads.[^vit-gemm]
- `inkling2_dspark_rdna2.cu`: head 128, 8 block rows, GQA 3 (24 q / 8 kv), 64-key attention tiles, Markov rank 256. The GEMV stages a 512-K activation chunk in LDS per 8-wave block.[^dspark-const]

### DSpark draft attention update @ `ba9b43bc`: LDS-staged K/V, DPP reductions, per-query combine (2026-11-24)

Dest moved `f22c794a` → [`ba9b43bc`](https://github.com/opengfx1030/vllm-rdna/commit/ba9b43bcd479acc4bddcc98a375a31152279a872) (Merge #207, content commit `08a30fad`). The only HIP file in the delta is `inkling2_dspark_rdna2.cu`; the rest is Python (graph capture, TP draft vocab), recipes, tests and the fork doc.[^tip4] Constants are unchanged: `kHead = 128`, `kBlockRows = 8`, `kQPerKv = 3`, `kNumQ = 24`, `kTile = 64`, `kPartStride = 130`.[^d4-const]

**`attn_split_kernel` (still `__launch_bounds__(256)`, grid `(splits, kv_heads, num_reqs)`).**[^d4-split][^d4-launch]
- Static LDS, from the declarations and constants (arithmetic, not a compiler receipt): `qs[24][136]` half = 6,528 B; new `kv_s[64][136]` half = 17,408 B; `ss[24][64]` float = 6,144 B; `koff[64]` long long = 512 B; `m_run` / `l_run` / `alpha_s` `[24]` float = 288 B. Total **30,880 B** per CTA (30.2 KiB). The source comment rounds this to "~30.8 KB".[^d4-lds]
- Residency from LDS alone: 4 × 30,880 = 123,520 B fits the 128 KiB WGP pool, and 5 × 30,880 = 154,400 B does not, so LDS caps the kernel at **4 CTAs/WGP** (32 waves). Rounding each CTA to the ISA 1 KB granule (31,744 B) gives the same 4.[^rocm-gpu-arch-specs-d4][^lds-granule-d4] The source comment says "< 32 KB -> 4 CTAs/WGP". The VGPR count was not published and was not compiled here, so whether VGPRs bind before LDS is **unverified**.
- *Historical @ `f22c794a`:* the LDS was only `qs[24][128]`, `ss`, `koff` and the three run arrays, 13,088 B per CTA. K and V were not staged.[^d4-old-lds]
- `qs` and `kv_s` rows are padded to `kHeadPad = kHead + 8` = 136 halves (272 B). The comment says this is to "avoid LDS bank conflicts while keeping 16B alignment". That is the code's stated reason; bank behaviour was not measured here. This is the same 136-half row as the `i2_tile_kernel` sQ / sK rows above.[^d4-lds]
- **K staging.** Each 64-key tile is loaded from the paged cache once with 16 B loads: 16 threads per key (`ld_c8 = tid & 15`), 16 keys per pass (`tid >> 4`), 4 passes. Keys past `nt` are written as zeros. QK then reads K from `kv_s`, with the same mapping as before: 64 keys × 4 thread groups of 6 queries, `dot8` = four `__builtin_amdgcn_fdot2` (`V_DOT2_F32_F16`).[^d4-kstage][^d4-dot8]
- **V prefetch.** After the QK barrier, each thread loads its 4 V chunks (`uint4 v_reg[4]`, 16 VGPRs) from global memory **before** the online softmax and writes them into the same `kv_s` buffer after it, followed by one barrier. So V reuses the K buffer, and the global V latency overlaps the softmax.[^d4-vstage]
- **PV remap.** 8 waves × 3 queries (`wave*3 .. wave*3+2`) and 32 lanes × 4 dims (`lane*4 .. lane*4+3`). Each key is one 8 B `uint2` LDS read per lane and `fmaf` into `o[3][4]`. The partial store is two `float2` per query, because `kPartStride = 130` floats is 8 B-aligned but not 16 B-aligned for odd `qi` (source comment).[^d4-pv]
- *Historical @ `f22c794a`:* QK read K straight from the cache row (`cache + koff[kk]`), so the 4 `g4` threads of a key each read the same 256 B row. PV read V from the cache as one `__half2` per thread per key, with `o[6][2]`, `dp = tid & 63` and `gq = tid >> 6` (4 groups × 6 queries, 2 dims per thread).[^d4-old-split]
- The host split count is unchanged: `attention_splits(R) = clamp(ceil(1024 / (R·8)), 1, 512)`, so 128 splits at R = 1. The `8` is a literal, not `kv_heads`. The fork doc's "Adaptive `num_splits`" line does not match this delta, since `attention_splits` is identical at `f22c794a` and `ba9b43bc`.[^d4-splits][^d4-old-splits][^d4-doc]

**`attn_combine_kernel` (still `__launch_bounds__(128)`, 128 threads = one per dim).**[^d4-combine][^d4-launch]
- Grid is now `num_reqs · kv_heads · kNumQ`, one CTA per (request, kv head, query): `qi = blockIdx.x % 24`. Each CTA does one max pass and one weighted-sum pass over the active splits for its query. With 8 KV heads in the cache (the file's "24 q heads / 8 kv heads" GQA constant), that is 1 · 8 · 24 = **192 CTAs at R = 1**, which matches the source comment ("192 CTAs at b=1 instead of 8"). `kv_heads` is read from the cache tensor, so a cache with fewer local KV heads gets proportionally fewer CTAs.[^d4-combine][^d4-launch]
- *Historical @ `f22c794a`:* grid `num_reqs · kv_heads` (8 CTAs at R = 1), and each CTA looped over all 24 queries serially.[^d4-old-combine]

**Wave32 reductions (`dspark_xor_lane<O>`, `warp_sum`, `warp_max`).**[^d4-xor]
- Under `#if defined(__HIP__RDNA2__)` the butterfly is the same as in the glue update above: O = 16 → `permlanex16(x, x, 0x76543210, 0xfedcba98, false, false)` (lane ^ 16); O = 8 / 4 → `update_dpp` `0x168` / `0x164` (`row_xmask:8` / `:4`); O = 2 / 1 → `quad_perm` `0x4E` / `0xB1` (lane ^ 2 / ^ 1). All use `row_mask = bank_mask = 0xF` and `bound_ctrl = false`. Other device passes fall back to `__shfl_xor(v, O)`.[^d4-xor][^llvm-dppctrl][^isa70648-g2]
- In this file only `warp_sum` and `warp_max` changed. Callers: the split-kernel online softmax (`warp_max`, `warp_sum`), `qkv_prep_kernel`, `fold_bias_kernel` and `block_sum<NV>` (so `mix_rms`, `embed_block`, `add_rmsnorm`). The other `__shfl_xor` loops (GEMV K-split reduce, Markov step, argmax value/index) are untouched at `ba9b43bc`.[^d4-xor][^d4-shfl-left]
- *Historical @ `f22c794a`:* `warp_sum` / `warp_max` were `for (m = 16; m >= 1; m >>= 1) __shfl_xor` loops on all archs.[^d4-old-xor]

**Qualification (fork doc, not re-run here).** The doc reports 29/29 attention oracle tests, 5/5 hand mutants killed and a clean 6-arch `archdiff` (0 diffs on gfx90a/942/950/1100/1200). Microbench, tok/s and serving numbers are left in the fork doc on purpose.[^d4-doc]

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

## Update @ `ea8aa241`: TP=1 GQA fused decode, NVFP4 tile variant, `attn_res_sconv`, LoopCD (2026-11-25)

Dest moved `1fdcb3a6` → [`ea8aa241`](https://github.com/opengfx1030/vllm-rdna/commit/ea8aa2418e6373de1537fa56e4ae0911ef6162a6) (2026-11-25 08:24 Paris, merge of PR #211; the range also merges #209 and #210, 17 commits). The kernel files that changed are `inkling2_attn_rdna2.cu`, `inkling2_dspark_rdna2.cu`, `inkling2_glue_rdna2.cu`, `moe_lowp_dot2_rdna2.cu` and `rdna_expert_cache.cu`. `CMakeLists.txt` and `cmake/` are not in the diff, so the gfx1030 source list and arch list are as above. The range `ba9b43bc` → `1fdcb3a6` (PR #208) touches no `csrc/` file, so the `ba9b43bc` sections still describe `1fdcb3a6`.[^e5-tip][^e5-cmp]

**Occupancy first.**
- New `i2_decode_fused_gqa_kernel<G, NW>` with `__launch_bounds__(NW * 32)`. NW is `kI2FusedWaves = 16` or the new `kI2FusedWavesShort = 8`, and `static_assert(NW >= G)` holds. The LDS is `s_acc[G][NW][128]` fp32 plus `s_ml[G][NW][2]` and `s_w[G][NW]`. That is (12·G·NW + 512·G·NW) B, or 25,152 B at G = 3, NW = 16, and 12,576 B at G = 3, NW = 8 (arithmetic from the declarations, not a compiler receipt).[^e5-gqa-lds]
- The source says each 16-key super-tile is streamed in two 8-key halves (`I2FdHalf`, "34 VGPRs instead of 68") so all G query accumulators stay in registers, with "zero spills up to G = 4". The fork did not commit a resource receipt for this kernel, so VGPR count and waves/SIMD are not recorded here.[^e5-gqa-head]
- The old per-q-head kernel is now `i2_decode_fused_nh_kernel<G, NW>`. It has the same `NW*32` launch bounds and the same `s_acc[NW][128]` LDS (8,384 B at NW = 16), but its grid is now `(H_q, fused_max_q, R)`, and it returns on `q_len > fused_max_q` instead of `q_len != 1`.[^e5-nh]
- New `attn_res_sconv_kernel` is `__launch_bounds__(256)` and launches at the `row_launch_shape` block (≤ 256 threads, 1–4 float4 per thread). The source comment says it computes the conv partial in its own prologue scope "(~104 VGPRs vs 164 VGPRs when all loads overlap)". That is a source comment, not a receipt.[^e5-sconv-vgpr][^e5-sconv-lb]
- New launch bounds: DSpark `indexed_gemm_kernel` `__launch_bounds__(P::T)`; LoopCD `loopcd_{blend,readout,prepare_hidden,logits_stats,logits_apply}_kernel` all `__launch_bounds__(256)`. The tile kernels, split-K decode and combine keep their bounds and LDS (no `__shared__` change outside the two fused-decode kernels).[^e5-indexed][^e5-loopcd]

**Attention host policy (`inkling2_attention`, `fae1b90d` + `97e70270`).**[^e5-policy][^c-fae1b90d][^c-97e70270]
- `splits_decode = splits_tile = 0` (`fused_all`) is now valid for `max_query_len ≤ kI2FusedMaxQ = 8`. That runs one fused launch for every request with no tile or combine launch.[^e5-fusedall]
- `tile_impl` bits: bit 0 picks the pipelined tile; bit 1 skips the `i2_decode_*` launch on uniform multi-query batches (`skip_decode`); bits 2–3 force NW = 8 / 16; bit 4 forces the nh grid.[^e5-tileimpl]
- Default choice: the nh grid when `fused_max_q == 1` and `H_kv · R < 24` (no NW override). Otherwise the GQA grid, with NW = 8 when `H_kv · q · R > 36` and `0 < window ≤ 1024`, else NW = 16. The source comment's reason is that the nh grid fills all 36 WGPs at q = 1 with few CTAs. `VLLM_INKLING2_DECODE_FUSED=nh` and `VLLM_INKLING2_FUSED_WAVES` override it.[^e5-policy]
- Python side (`attention_splits_hip`): it returns `(0, 0, None)` when `1 ≤ max_query_len ≤ 8` and (`0 < window ≤ 1024` or `max_seq_len ≤ 1024`). Uniform multi-query batches on the tile path get `tile_impl = base | 2` (skip decode), with `base` = pipelined tile unless `PREFILL_ATTN_TILE=0` or a sliding layer has ≤ 32 query tokens.[^e5-py-splits]
- `e3190664` moves `kI2BC = 64` and `kI2SubTile = 16` inside the `__HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__` guard, together with the wave32 `#error`. The parameter structs (`I2ConvMeta`, `I2AttnArgs`) moved to global scope so the non-gfx1030 empty stubs share them. Values are unchanged.[^e5-guard][^c-e3190664]

**Conv state written in-op (Python gate, `97e70270`).** `conv_writes_in_op` now lets the conv op write the state when `max_query_len ≤ max(1, ring_rows − (k − 1))` (and not in a piecewise capture). *Historical @ `1fdcb3a6`:* only `max_query_len == 1`. The C++ `write_state` handling did not change.[^e5-convwr]

**MoE decode expert GEMM (`moe_lowp_dot2_rdna2.cu`, `56587204` + `fae1b90d`).**[^c-56587204][^c-fae1b90d]
- `moe_gemm_nvfp4_kernel_rdna2` is now templated on `TX` (threads) and `BK` (K tile), defaulting to `THREADS_X = 256` and `BLOCK_KN_SIZE = 256`, with `static_assert(BK % 32 == 0)`. LDS is `block_a[BLOCK_SIZE_M][BK + 8]` fp16. Activation staging loops `tk = t; tk < BK; tk += TX`. Still no `__launch_bounds__`.[^e5-lowp-kernel]
- Tile variants: `0` (default) = TX 256 × BK 256. `1` = TX 128 × BK 256, only for `block_size_m ≤ 4`. Picked by `VLLM_INKLING2_NVFP4_VARIANT` (read once) or `moe_nvfp4_set_variant_rdna2`, which only the microbench binds. A third variant (TX 256 × BK 384) was removed in `fae1b90d`. The MXFP8 decode kernel has no variant switch and keeps the 256 × 256 tile.[^e5-variant][^e5-bk384]
- Python `block_size_m` (FUSE_GLUE spec table). MXFP8 is now `(1→1, 12→2, 48→4)` (`56587204`); *historical @ `1fdcb3a6`:* `(4→1, 48→2)`. NVFP4 is raised to ≥ 4 when `max_query_len ≥ 6` and `num_tokens ≥ 8` (`fae1b90d`).[^e5-bm][^e5-bm-old]

**Glue (`inkling2_glue_rdna2.cu`, `56587204` + `3aee7f0a`).**[^c-56587204][^c-3aee7f0a]
- `attn_res_sconv_kernel<Q, ITER, NORM, SCONV_MODE>` fuses the 4-tap static short conv (fp32 weight `[H, 1, 4]`, scale `1/√5`) into the `attn_res` prologue. In-chunk taps come from `x`, earlier ones from the paged conv ring (`GlueConvMeta`). Mode 1 is `x0 + sconv(cx)`; mode 2 is `sconv(cx)`. It can write the conv state and a bf16 or fp32 second norm output (Q = 1). The Python gate is TP = 1 and ≤ 64 tokens (`_SCONV_FUSE_MAX_TOKENS`).[^e5-sconv-vgpr][^e5-model-sconv]
- `gate_mix_split_kernel` gate raised to T ≤ 16 (above).[^e5-mix]
- LoopCD kernels (`3aee7f0a`) are fp32 streaming kernels with no DOT and no `fdot2`. `loopcd_logits_*` split the vocab over `S = clamp(72 / B, 1, 32)` CTAs per row (capped at `ceil(V/256)`), with `s_m[8]` / `s_l[8]` LDS and wave reductions.[^e5-loopcd]

**DSpark (`inkling2_dspark_rdna2.cu`, `6e0dbd58`).**[^c-6e0dbd58]
- `gemv_kernel` takes an optional `w_row_ids` (int32/int64) and reads weight rows `w + w_row_ids[r]·ldw` in place, with no subset copy. The row/wave layout (`kGemvWaves = 8`) and the `dot8` = four `fdot2` loop are unchanged.[^e5-gemv]
- New `indexed_gemm_kernel<P>` is the same `rdna_hint_flex` designs D1–D6 (`IdxD1..IdxD6`, DMA = 0), with weight rows gathered through `w_row_ids`. The source says this keeps "exact fp32 accumulation order and single fp16 store rounding". LDS is `P::LDS_BYTES` per design (not re-derived here).[^e5-indexed]
- `argmax_rows_kernel` (`__launch_bounds__(512)`) can remap the winning index through `token_map` and write to a strided column.[^e5-argmax]

**Expert offload (`1850737d`).** `rdna2_expert_cache_copy` accepts 1–8 payload tensors (was exactly 6). It is a host `hipMemcpyBatchAsync` path gated to `gfx103x`, with no device kernel. The new recipe `inkling2-mini-tp1-offload.env` sets only `VLLM_INKLING2_MOE_OFFLOAD_*` and `VLLM_KV_CACHE_MARGIN_FLOOR_MIB`. None of the TP = 1 recipes in the range set a HIP-kernel gate, so the defaults above apply.[^e5-offload][^c-1850737d]

## Env defaults that gate HIP kernels

| Variable | Default | Selects | Read |
|---|---|---|---|
| `VLLM_INKLING2_DECODE_FUSED` | `1` | fused sliding-window decode; `2` also fuses full layers for a single request (opt-in); `0` uses split-K + combine. Since `ea8aa241`: any value other than `""`/`0`/`nh` sends 1 ≤ q ≤ 8 batches with short keys to the one-launch GQA path; `nh` forces the per-q-head grid (C++ host reads it once) | once at import (baked into graphs) [^ops-env][^e5-py-splits][^e5-policy] |
| `VLLM_INKLING2_FUSED_WAVES` (new) | unset | `8` or `16` forces the GQA fused-decode CTA wave count; other values ignored | C++ host, once (static) [^e5-policy] |
| `VLLM_INKLING2_FUSE_SCONV_ATTN_RES` (new) | on | fuses post-attention / post-MLP 4-tap static convs into `attn_res_sconv_kernel` at TP = 1 for steps of ≤ 64 tokens | Python module import [^e5-py-sconv][^e5-model-sconv] |
| `VLLM_INKLING2_NVFP4_VARIANT` (new) | `0` | NVFP4 decode expert tile: `0` = 256 thr × BK 256; `1` = 128 thr × BK 256 for `block_size_m` ≤ 4 | C++ host, once (static) [^e5-variant] |
| `VLLM_INKLING2_SPEC_BM4_MIN` (new) | `0` (off) | forces `block_size_m` ≥ 4 from that token count under FUSE_GLUE (A/B knob) | Python [^e5-bm] |
| `VLLM_INKLING2_MOE_OFFLOAD_SLOTS` / `_COPY` / `_POLICY` / `_DECAY` (new) | `0` / `tensor` / `lfu_lru` / `128` | TP = 1 expert offload (`0` = off). Host copies only, no new device kernel | `vllm/envs.py` + `moe.py` [^e5-envs][^e5-offload] |
| `VLLM_INKLING2_LOOPCD_*` (new) | `ALPHA` 0.0 (off), `MODE` hidden, `BETA` 0.1 | LoopCD blend / readout / logits kernels (only run when alpha ≠ 0) | Python [^e5-loopcd-env] |
| `VLLM_INKLING2_PREFILL_ATTN_TILE` | `1` | `i2_tile_pipe_kernel`, except sliding layers with ≤ 32 query tokens, which keep `i2_tile_kernel` | once at import [^ops-env][^ops-tile] |
| `VLLM_INKLING2_FUSE_ATTN_CHAIN` | on | `attn_res_fused_kernel` / one-launch messages for batches ≤ 16 tokens | Python module import [^py-fuse] |
| `VLLM_INKLING2_FUSE_GLUE` | on | SwiGLU fused into the w2 expert GEMM (≤ 16 tokens), 8-wave `gate_mix` split (T ≤ 16 since `ea8aa241`; T ≤ 8 @ `f22c794a`…`1fdcb3a6`), `moe_route_small_kernel` (T ≤ 4, since `f22c794a`). Not the DPP / `attn_res` changes, which are unconditional on gfx1030 | Python + C++ host, once [^py-glue][^glue-env][^py-moe-swiglu][^g2-env] |
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
[^tip3]: opengfx1030/vllm-rdna commit [`f22c794a`](https://github.com/opengfx1030/vllm-rdna/commit/f22c794a00e609fc8c03f35d374c5258f8e5187c) (2026-11-24 13:07 Paris): "Merge #206: Inkling-2 glue DPP reductions, small-T moe_route, and T<=8 gate_mix_split", from [`c1ce3ec3`](https://github.com/opengfx1030/vllm-rdna/commit/c1ce3ec355f946849fee47be9fbae111e9a6fcbc) (13:05 Paris). Compare [`d71bddca...f22c794a`](https://github.com/opengfx1030/vllm-rdna/compare/d71bddca6e0f9d9f22390b8dde758cf7baf60481...f22c794a00e609fc8c03f35d374c5258f8e5187c): `csrc/rocm/inkling2_glue_rdna2.cu`, `docs/rdna2/inkling2-decode-glue.md`, `tests/kernels/core/test_inkling2_glue_rdna2.py`, `tools/rdna/inkling2/glue_jit.py`. A grep for `amdgpu_waves_per_eu` in the glue file at this SHA has 0 hits.
[^g2-xor]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L88-L147 (`xor_lane_i32`, `xor_lane`, `wave_sum`, `wave_max`, `wave_sum_n`) and L151-L153 (`block_sum` → `wave_sum`). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L88-L153>
[^g2-topk]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L1171-L1242 (`route_topk_wave`: `step_max` butterfly L1210-L1214, `__shfl(mine, bi & (kWave - 1))` L1225, `flat_ids_out` L1239). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L1171-L1242>
[^g2-attn]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L261-L493 (`attn_res_body`: `s_red` only L273; bulk fetch L326-L331; one barrier L355; per-wave coefficients L357-L398; `__shfl` coefficient reads L411, L423; `wave_sum_n<Q>` L448; `norm_w` prefetch L453-L466; single norm barrier L467). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L261-L493>
[^g2-attn-lb]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L496 (`__launch_bounds__(256) attn_res_kernel`). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L496>
[^g2-small]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L74 (`kRouteSmallThreads = 128`), L1473-L1633 (`moe_route_small_body`), L1645-L1650 (`__launch_bounds__(kRouteSmallThreads) moe_route_small_kernel`), L2401-L2419 (`launch_route_small`, 0 B dynamic LDS). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L1473-L1650>
[^g2-host-route]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L2560-L2611 (small-T branch `fuse_glue() && T <= 4`, `NT = 1`, `NEG = E`, `sgrid = nzero + E`; large path `NEG` clamp and `shmem`). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L2560-L2611>
[^g2-split]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L2091-L2094, L2132-L2140. <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L2091-L2140>
[^g2-gate-tb]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L1878-L1882 (`gate_tb`). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L1878-L1882>
[^g2-env]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `f22c794a` L1713-L1721 (`fuse_glue()`, default on); only call sites L2092 and L2568. <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu#L1713-L1721>
[^g2-doc]: `docs/rdna2/inkling2-decode-glue.md` @ `f22c794a` §8 (L450-L473). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/docs/rdna2/inkling2-decode-glue.md#L450-L473>
[^g2-test]: `tests/kernels/core/test_inkling2_glue_rdna2.py` @ `f22c794a` (`t2_e255_k16_f32w`; `test_moe_route_repeated_launches_do_not_race` parametrised over T). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/tests/kernels/core/test_inkling2_glue_rdna2.py>
[^g2-old-xor]: *Historical* `csrc/rocm/inkling2_glue_rdna2.cu` @ `d71bddca` L88-L100 (`wave_sum` / `wave_max` as `__shfl_xor` loops, "arch independent"). <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu#L88-L100>
[^g2-old-topk]: *Historical* `csrc/rocm/inkling2_glue_rdna2.cu` @ `d71bddca` L1134-L1153 (`__shfl_xor` arg-max loop; `wave_sum(mine)` weight broadcast). <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu#L1134-L1153>
[^g2-old-attn]: *Historical* `csrc/rocm/inkling2_glue_rdna2.cu` @ `d71bddca` L225-L227 (`s_red`, `s_tot`, `s_coef`), L308-L346 (three barriers, wave-0 softmax), L359 / L370 (`s_coef` reads), L395-L400 (two norm barriers). <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu#L225-L400>
[^g2-old-split]: *Historical* `csrc/rocm/inkling2_glue_rdna2.cu` @ `d71bddca` L1848 (`T <= 4`). <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu#L1848>
[^g2-old-grid]: *Historical* `csrc/rocm/inkling2_glue_rdna2.cu` @ `d71bddca` L2316 (single route grid `nzero + NT*NEG` for all T). <https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu#L2290-L2322>
[^llvm-dppctrl]: LLVM `llvm/lib/Target/AMDGPU/SIDefines.h` `DppCtrl` enum (`QUAD_PERM_FIRST = 0`, `QUAD_PERM_LAST = 0xFF`, `ROW_XMASK0 = 0x160`, `ROW_XMASK_LAST = 0x16F`) @ llvm-project `278221a5b0be60d025312c5e007a4c4dc01eb59a`. <https://github.com/llvm/llvm-project/blob/278221a5b0be60d025312c5e007a4c4dc01eb59a/llvm/lib/Target/AMDGPU/SIDefines.h#L1207-L1248>
[^isa70648-g2]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648: DPP_CTRL `QUAD_PERM` (Table 91) and the `V_PERMLANEX16_B32` opcode entry. <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^tip4]: opengfx1030/vllm-rdna merge [`ba9b43bc`](https://github.com/opengfx1030/vllm-rdna/commit/ba9b43bcd479acc4bddcc98a375a31152279a872) (2026-11-24 14:28 Paris): "Merge pull request #207 from opengfx1030/perf/inkling2-dspark-conc-tp", from [`08a30fad`](https://github.com/opengfx1030/vllm-rdna/commit/08a30fadda13cc691ca22f5aacd7cd25ab2d21ed) (14:24 Paris) "perf(rdna2): optimize DSpark draft attention, schedule-aware graph capture, and TP1/2/4 recipes". Compare [`f22c794a...ba9b43bc`](https://github.com/opengfx1030/vllm-rdna/compare/f22c794a00e609fc8c03f35d374c5258f8e5187c...ba9b43bcd479acc4bddcc98a375a31152279a872): 14 files; the only `.cu` is `csrc/rocm/inkling2_dspark_rdna2.cu` (+124 / −51).
[^d4-const]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L48-L55. <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L48-L55>
[^d4-split]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L678-L858 (`attn_split_kernel`). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L678-L858>
[^d4-launch]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L1765 (`kv_heads = kv_cache.size(1)`), L1789-L1803 (split grid `(splits, kv_heads, num_reqs)` × 256; combine grid `num_reqs * kv_heads * kNumQ` × `kHead`). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L1765-L1803>
[^d4-lds]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L685-L693 (padding comment, `kHeadPad`, `qs`, `kv_s`, `ss`, `koff`, run arrays). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L685-L693>
[^rocm-gpu-arch-specs-d4]: ROCm docs 7.2.4, "GPU architecture specifications" (Radeon PRO V620 row: gfx1030, LDS 128 KiB). https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11; see [lds-occupancy.md](../../silicon/lds-and-tiles/lds-occupancy.md))
[^lds-granule-d4]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, §3.6.6: LDS allocation unit 256 dwords = 1024 B (as tabled in [lds-occupancy.md](../../silicon/lds-and-tiles/lds-occupancy.md) §3). <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^d4-kstage]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L718-L724 (staging and QK geometry), L748-L779 (K stage, barrier, QK from `kv_s`). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L718-L779>
[^d4-dot8]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L94-L112 (`dot2u` → `__builtin_amdgcn_fdot2`, `dot8`). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L94-L112>
[^d4-vstage]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L781-L820 (V prefetch to `v_reg[4]`, online softmax, `kv_s` write, barrier). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L781-L820>
[^d4-pv]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L725-L730, L821-L852 (PV geometry, `uint2` LDS reads, `kPartStride` alignment comment, `float2` stores). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L821-L852>
[^d4-combine]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L860-L895 (`attn_combine_kernel`, comment "One CTA per (req, kvh, qi) -> 192 CTAs at b=1 instead of 8"). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L860-L895>
[^d4-xor]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L155-L196 (`dspark_xor_lane<O>`, `warp_sum`, `warp_max`); callers L205 (`block_sum`), L320 / L359 / L373 / L426, L629, L796 / L805, L1030. <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L155-L196>
[^d4-shfl-left]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ba9b43bc` L550-L556, L997-L1002, L1073-L1090 (remaining `__shfl_xor`). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/inkling2_dspark_rdna2.cu#L550-L556>
[^d4-splits]: `vllm/models/inkling2_loop/amd/dspark.py` @ `ba9b43bc` L81-L83, L123-L126 (`_ATTN_TARGET_BLOCKS = 1024`, `_ATTN_MAX_SPLITS = 512`, `attention_splits`). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/vllm/models/inkling2_loop/amd/dspark.py#L81-L126>
[^d4-old-splits]: *Historical* `vllm/models/inkling2_loop/amd/dspark.py` @ `f22c794a` L81-L83, L123-L126 (same text). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/vllm/models/inkling2_loop/amd/dspark.py#L81-L126>
[^d4-doc]: `docs/rdna2/inkling2-dspark.md` @ `ba9b43bc` L807-L826 (2026-11-24 optimization pass, item 1). <https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/docs/rdna2/inkling2-dspark.md#L807-L826>
[^d4-old-lds]: *Historical* `csrc/rocm/inkling2_dspark_rdna2.cu` @ `f22c794a` L654-L657. <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_dspark_rdna2.cu#L654-L657>
[^d4-old-split]: *Historical* `csrc/rocm/inkling2_dspark_rdna2.cu` @ `f22c794a` L681-L685 (`dp`, `gq`, `o[6][2]`), L711 (K from `cache + koff[kk]`), L753-L770 (PV from cache). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_dspark_rdna2.cu#L681-L770>
[^d4-old-combine]: *Historical* `csrc/rocm/inkling2_dspark_rdna2.cu` @ `f22c794a` L788-L822 (`attn_combine_kernel` with the `qi` loop), L1726 (grid `num_reqs * kv_heads`). <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_dspark_rdna2.cu#L788-L822>
[^d4-old-xor]: *Historical* `csrc/rocm/inkling2_dspark_rdna2.cu` @ `f22c794a` L155-L165. <https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_dspark_rdna2.cu#L155-L165>

[^e5-tip]: opengfx1030/vllm-rdna merge [`ea8aa241`](https://github.com/opengfx1030/vllm-rdna/commit/ea8aa2418e6373de1537fa56e4ae0911ef6162a6) (2026-11-25 08:24 Paris): "Merge pull request #211 from opengfx1030/feat/inkling2-tp1-moe-offload". The grep for `waves_per_eu` was run over `inkling2_attn_rdna2.cu`, `inkling2_glue_rdna2.cu`, `inkling2_dspark_rdna2.cu` and `moe_lowp_dot2_rdna2.cu` at this SHA, with 0 hits. `git diff --stat 1fdcb3a6 ea8aa241 -- CMakeLists.txt cmake` is empty, and `git diff --stat ba9b43bc 1fdcb3a6` lists no `csrc/` path.
[^e5-cmp]: Compare [`1fdcb3a6...ea8aa241`](https://github.com/opengfx1030/vllm-rdna/compare/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa...ea8aa2418e6373de1537fa56e4ae0911ef6162a6) (17 commits; merges #209 `d7c44a94`, #210 `41ef6ba1`, #211 `ea8aa241`).
[^e5-gqa-lds]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L72-L78 (`kI2FusedWaves`, `kI2FusedWavesShort = 8`, `kI2FusedMaxQ = 8`), L1197-L1203 (`__launch_bounds__(NW * 32)`, `static_assert(NW >= G)`, LDS arrays). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L1197-L1203>
[^e5-gqa-head]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L1098-L1109 (GQA KV-sharing comment: grid, `I2FdHalf` "34 VGPRs instead of 68", "zero spills up to G = 4"). Kernel added in [`97e70270`](https://github.com/opengfx1030/vllm-rdna/commit/97e70270d045ecd80e65c18a6de99d603b5a35fb). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L1098-L1109>
[^e5-nh]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L774-L800 (nh-grid comment), L959-L973 (`i2_decode_fused_nh_kernel`, LDS, `q_len > a.fused_max_q` return), L3050 (`grid_f_nh(nh, fused_max_q, R)`). *Historical:* `i2_decode_fused_kernel` @ `1fdcb3a6` L948-L960 (`qsl[r+1]-qs != 1` return). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L959-L973> / <https://github.com/opengfx1030/vllm-rdna/blob/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa/csrc/rocm/inkling2_attn_rdna2.cu#L948-L960>
[^e5-sconv-vgpr]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `ea8aa241` L70 (`kMaxStates = 64`), L264-L286 (`GlueConvMeta`, `AttnResSconvArgs`: conv weight `[H, 1, 4]` fp32, `conv_inv` = 1/sqrt(5)), L560-L570 (`SCONV_MODE` 1 / 2, "~104 VGPRs vs 164 VGPRs"); added in [`56587204`](https://github.com/opengfx1030/vllm-rdna/commit/56587204cc664ecb75957e46ab42368789852029). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_glue_rdna2.cu#L560-L570>
[^e5-sconv-lb]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `ea8aa241` L723-L729 (`__launch_bounds__(256) attn_res_sconv_kernel`), L2223-L2232 (`row_launch_shape`: iter 1..4, block rounded to a wave). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_glue_rdna2.cu#L2223-L2232>
[^e5-indexed]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ea8aa241` L607-L626 (indexed GEMM comment, `__launch_bounds__(P::T)`, `lds[P::LDS_BYTES]`), L1894-L1903 (`IdxD1..IdxD6`), L1920-L1935 (`launch_indexed_gemm`, `kBks`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_dspark_rdna2.cu#L607-L626>
[^e5-loopcd]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `ea8aa241` L1913-L2160 (LoopCD kernels, `__launch_bounds__(256)`, `s_m[8]` / `s_l[8]`), L3641-L3643 (`S`). Added in [`3aee7f0a`](https://github.com/opengfx1030/vllm-rdna/commit/3aee7f0a76337e56e460077535cc5336a45ffaea). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_glue_rdna2.cu#L1913-L2160>
[^e5-policy]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L3020-L3044 (`VLLM_INKLING2_DECODE_FUSED` = `nh`, `VLLM_INKLING2_FUSED_WAVES` 8/16, `total_gqa_ctas < 24` nh rule with the "fills all 36 WGPs" comment, NW = 8 when `> 36` and `short_window`), L3049-L3072 (grids, launch macro). The `< 24` rule landed in [`fae1b90d`](https://github.com/opengfx1030/vllm-rdna/commit/fae1b90d22168965dea32f8c8138e348e553cdf7). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L3020-L3072>
[^e5-fusedall]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L2932-L2944 (`fused_all`, `max_query_len <= kI2FusedMaxQ` check). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L2932-L2944>
[^e5-tileimpl]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L2945-L2959 (`tile_impl` bits, `skip_decode`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L2945-L2959>
[^e5-py-splits]: `vllm/models/inkling2_loop/amd/attention.py` @ `ea8aa241` L469-L471 (`_FUSED_MAX_QUERY_LEN = 8`, `_FUSED_MAX_KEYS = 1024`, `_TILE_SKIP_DECODE = 2`), L518-L559 (`attention_splits_hip`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/attention.py#L518-L559>
[^e5-guard]: `csrc/rocm/inkling2_attn_rdna2.cu` @ `ea8aa241` L68-L132 (global constants, shared structs, guarded `kI2BC` / `kI2SubTile` and wave32 `#error`); change is [`e3190664`](https://github.com/opengfx1030/vllm-rdna/commit/e31906640fb3d3e5f0403af6868b553d17789d18). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_attn_rdna2.cu#L66-L132>
[^e5-convwr]: `vllm/models/inkling2_loop/amd/attention.py` @ `ea8aa241` L474-L496 (`conv_writes_in_op`, `max_in_op`). *Historical:* `@ 1fdcb3a6` returned `md.max_query_len == 1 and not _tracing_piecewise_graph()`. <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/attention.py#L474-L496>
[^e5-lowp-kernel]: `csrc/rocm/moe_lowp_dot2_rdna2.cu` @ `ea8aa241` L43-L44 (`BLOCK_KN_SIZE`, `THREADS_X` 256), L159-L187 (`load_activations<T, BM, TX, BK>`), L193-L226 (kernel, `static_assert(BK % 32 == 0)`, `block_a[BM][BK + 8]`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/moe_lowp_dot2_rdna2.cu#L159-L226>
[^e5-variant]: `csrc/rocm/moe_lowp_dot2_rdna2.cu` @ `ea8aa241` L331-L345 (`moe_nvfp4_set_variant_rdna2`, `VLLM_INKLING2_NVFP4_VARIANT`), L368-L392 (`if constexpr (BLOCK_SIZE_M <= 4)`, variant 1 = `<T, BM, 128, 256>`), L394-L406 (dispatch). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/moe_lowp_dot2_rdna2.cu#L331-L406>
[^e5-bk384]: [`fae1b90d`](https://github.com/opengfx1030/vllm-rdna/commit/fae1b90d22168965dea32f8c8138e348e553cdf7) removes `if (variant == 2) launch_moe_nvfp4_cfg<T, BLOCK_SIZE_M, 256, 384>` from `launch_moe_nvfp4`.
[^e5-bm]: `vllm/models/inkling2_loop/amd/moe.py` @ `ea8aa241` L710-L719 (`_BLOCK_M_BY_TOKENS_SPEC`, `VLLM_INKLING2_SPEC_BM4_MIN`), L1088-L1103 (`_default_block_size_m`, NVFP4 `max_query_len >= 6 and num_tokens >= 8`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/moe.py#L1088-L1103>
[^e5-bm-old]: *Historical* `vllm/models/inkling2_loop/amd/moe.py` @ `1fdcb3a6` L620-L623 (`"mxfp8": ((4, 1), (48, 2))` in `_BLOCK_M_BY_TOKENS_SPEC`). <https://github.com/opengfx1030/vllm-rdna/blob/1fdcb3a66e82b1c0b9ebe3ce1da1e0b25fc3c4fa/vllm/models/inkling2_loop/amd/moe.py#L620-L623>
[^e5-mix]: `csrc/rocm/inkling2_glue_rdna2.cu` @ `ea8aa241` L2814-L2817 (`T <= 16`, `tb = gate_tb(T, 4)`); change in [`56587204`](https://github.com/opengfx1030/vllm-rdna/commit/56587204cc664ecb75957e46ab42368789852029). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_glue_rdna2.cu#L2814-L2817>
[^e5-py-sconv]: `vllm/models/inkling2_loop/amd/glue.py` @ `ea8aa241` L41-L48 (`VLLM_INKLING2_FUSE_SCONV_ATTN_RES`, default on). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/glue.py#L41-L48>
[^e5-model-sconv]: `vllm/models/inkling2_loop/amd/model.py` @ `ea8aa241` L106-L110 (`_SCONV_FUSE_MAX_TOKENS = 64`), L424-L431 (`sconv_fuse` needs `tp.size == 1`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/model.py#L106-L110>
[^e5-gemv]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ea8aa241` L507 (`kGemvWaves = 8`), L509-L565 (`gemv_kernel` with `IntVec w_row_ids`, `wr[NR]`, `dot8`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_dspark_rdna2.cu#L507-L565>
[^e5-argmax]: `csrc/rocm/inkling2_dspark_rdna2.cu` @ `ea8aa241` L1350-L1416 (`argmax_rows_kernel`, `token_map`, `out_stride`). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/inkling2_dspark_rdna2.cu#L1350-L1416>
[^e5-offload]: `csrc/rocm/rdna_expert_cache.cu` @ `ea8aa241` L48-L70 (`kMaxPayloads = 8`, `gfx103` check), L134-L145 (`hipMemcpyBatchAsync`); `tools/rdna/recipes/inkling2-mini-tp1-offload.env` @ `ea8aa241` L38-L44. Added in [`1850737d`](https://github.com/opengfx1030/vllm-rdna/commit/1850737d284c54160487479857d38581900ea29c). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/rdna_expert_cache.cu#L48-L70>
[^e5-envs]: `vllm/envs.py` @ `ea8aa241` L204-L207, L1687-L1700 (`VLLM_INKLING2_MOE_OFFLOAD_*` defaults). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/envs.py#L1687-L1700>
[^e5-loopcd-env]: `vllm/models/inkling2_loop/amd/loopcd.py` @ `ea8aa241` L158-L190 (`read_loopcd_env_defaults`: alpha 0.0, mode `hidden`, beta 0.1). <https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/vllm/models/inkling2_loop/amd/loopcd.py#L158-L190>
[^c-97e70270]: opengfx1030/vllm-rdna commit [`97e70270`](https://github.com/opengfx1030/vllm-rdna/commit/97e70270d045ecd80e65c18a6de99d603b5a35fb) (2026-11-24 20:49 Paris): "perf(inkling2/rocm): TP=1 GQA KV-sharing, 1-launch SWA verify & in-op conv state".
[^c-fae1b90d]: opengfx1030/vllm-rdna commit [`fae1b90d`](https://github.com/opengfx1030/vllm-rdna/commit/fae1b90d22168965dea32f8c8138e348e553cdf7) (2026-11-24 21:48 Paris): "perf(inkling2/rocm): lock in V620 microbench winners (q=1 nh-grid, spec NVFP4 bm=4, drop BK=384)".
[^c-e3190664]: opengfx1030/vllm-rdna commit [`e3190664`](https://github.com/opengfx1030/vllm-rdna/commit/e31906640fb3d3e5f0403af6868b553d17789d18) (2026-11-25 00:15 Paris): "fix(inkling2/rocm): guard gfx1030-only tile constants on non-RDNA2 device passes".
[^c-56587204]: opengfx1030/vllm-rdna commit [`56587204`](https://github.com/opengfx1030/vllm-rdna/commit/56587204cc664ecb75957e46ab42368789852029) (2026-11-24 21:03 Paris): "perf(inkling2): fuse TP=1 glue sconv/taps, reclaim workspace, fix MoE/gate cliffs".
[^c-6e0dbd58]: opengfx1030/vllm-rdna commit [`6e0dbd58`](https://github.com/opengfx1030/vllm-rdna/commit/6e0dbd586bdba484b761e65cb6cf4b279b22fe0e) (2026-11-24 20:42 Paris): "perf(rdna2/inkling2): reclaim 342 MiB DSpark VRAM, fuse strided argmax & per-batch Markov steps".
[^c-3aee7f0a]: opengfx1030/vllm-rdna commit [`3aee7f0a`](https://github.com/opengfx1030/vllm-rdna/commit/3aee7f0a76337e56e460077535cc5336a45ffaea) (2026-11-25 01:25 Paris): "feat(rdna2): implement LoopCD hidden and logits contrastive decoding for Inkling2Loop".
[^c-1850737d]: opengfx1030/vllm-rdna commit [`1850737d`](https://github.com/opengfx1030/vllm-rdna/commit/1850737d284c54160487479857d38581900ea29c) (2026-11-25 08:13 Paris): "feat(rdna2): add TP=1 MoE expert offloading and unified 16/32 GiB recipe for Inkling-2-Mini".
