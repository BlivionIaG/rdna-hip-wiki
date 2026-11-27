---
title: "Models"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/93c59de2b50ccea1b26f299e2ac5713861ffb2e3
  - https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc
  - https://github.com/opengfx1030/vllm-rdna/commit/ec349e9079f6d19865362010bebcd36bfbbe0f8d
  - https://github.com/opengfx1030/vllm-rdna/commit/0f596e6e0e9b2683ef7a4ec3eb6ef93eac9e1fd9
  - https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d
  - https://github.com/opengfx1030/vllm-rdna/commit/f22c794a00e609fc8c03f35d374c5258f8e5187c
  - https://github.com/opengfx1030/vllm-rdna/commit/ba9b43bcd479acc4bddcc98a375a31152279a872
  - https://github.com/opengfx1030/vllm-rdna/commit/ea8aa2418e6373de1537fa56e4ae0911ef6162a6
  - https://github.com/opengfx1030/vllm-rdna/commit/8d141bdc80b3124315a4a1e52ab4485cb6974f5e
---
# Models

Model-specific HIP glue (one page per model family). Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [qwen4exp-flash-next-hip.md](qwen4exp-flash-next-hip.md) | T46 fdot2 glue; T47–T49 HC/QSA/PLE scalar | HC compute-correct @ `8960a3bc`. At `1dc3bac4`: `VLLM_RDNA_FUSED_HC` default **on**; fused HC prefill (`hc_prefill_rdna2.cu`) default mode 2; HC_PREFILL_HIP / QSA / PLE default off [^c-8960a3bc]. Update @ `8d141bdc`: QSA pipe `s_off` in 16-B units (reach 2³¹ × 16 B), opt-in W8A8 `sdot4` dense/HC prefill, default-on grouped fp16 MTP MoE prefill at ≥ 64 tokens [^c-8d141bdc] |
| [inkling2-mini-hip.md](inkling2-mini-hip.md) | fdot2 everywhere (opt-in W4A8 prefill: sdot4); DPP `row_xmask` / `row_share`, `permlanex16` | Native HIP serving at `00f106c5`: attention (split-K / fused decode, tile and pipelined tile, 244 VGPR / 32.5 KB LDS at G = 3), conv-state ring, glue, linear, ViT, DSpark, NVFP4/MXFP8 decode and 32-row prefill expert GEMMs. Fused decode, pipelined tile, FUSE_ATTN_CHAIN and FUSE_GLUE are on by default [^c-00f106c5]. Glue update @ `f22c794a`: DPP/permlanex16 wave reductions, `attn_res` LDS/barrier cut, 128-thread one-expert-per-CTA router for T ≤ 4, `gate_mix_split` to T ≤ 8 [^c-f22c794a]. DSpark update @ `ba9b43bc`: split-KV K/V staged in a padded 30,880 B LDS tile (LDS-bound 4 CTAs/WGP), DPP/permlanex16 `warp_sum`/`warp_max`, combine one CTA per query [^c-ba9b43bc]. Update @ `ea8aa241`: new `i2_decode_fused_gqa_kernel` (one CTA per kv head × query offset, NW 16/8, `s_acc[G][NW][128]` LDS, fused verify q ≤ 8), nh grid kept for q = 1 with < 24 GQA CTAs; NVFP4 decode tile variant 128 thr × BK 256 (opt-in) and BK 384 dropped; `attn_res_sconv_kernel` (TP = 1); `gate_mix_split` to T ≤ 16; DSpark zero-copy indexed GEMM/GEMV; LoopCD fp32 kernels [^c-ea8aa241]. Update @ `93c59de2`: one-launch `gate_down_mix_fused_kernel` (256 thr, ≤ 36,864 B LDS at H = 2304, fdot2 down + fp32 mix, grid ≤ CU count, T ≤ 16); NVFP4 decode auto tail tiles BK 192 (K % 192 = 0, K % 256 ≠ 0) and TX 64 (w13 N mod 1024 = 128/256); offload prefetch is host-only (`PREFETCH` 1/4/4) [^c-93c59de2]. Update @ `8d141bdc`: opt-in zero-copy hybrid NVFP4/MXFP8 offload expert GEMMs (`__gfx1030__` only, fdot2, VRAM slot or mapped pinned host per expert), MXFP8 decode BK 192 K-tail [^c-8d141bdc] |
| [dsv4-mhc-hip.md](dsv4-mhc-hip.md) | scalar fp32 (no DOT); 16-lane xor-shuffle Sinkhorn | DSv4 mHC pre/post HIP (`rdna/dsv4/mhc.cu`, `ec349e90` + `0f596e6e`, unchanged at `64d59b89`); decode pre GEMM split over 32 blocks; ≤16 tokens fused GEMM, else rocBLAS. Selected after AITER / TileLang for 4 streams fp16/bf16 [^c-0f596e6e] |

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-8960a3bc]: opengfx1030/vllm-rdna commit [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc) (2026-09-16) — "fix(rocm): make the Qwen4Exp HC HIP chain compute correctly".
[^c-0f596e6e]: opengfx1030/vllm-rdna commit [`0f596e6e`](https://github.com/opengfx1030/vllm-rdna/commit/0f596e6e0e9b2683ef7a4ec3eb6ef93eac9e1fd9) (2026-10-09) — "perf(rdna): split the decode mHC pre GEMM across blocks, parallel Sinkhorn" (file last touched before tip `64d59b89`; first added in [`ec349e90`](https://github.com/opengfx1030/vllm-rdna/commit/ec349e9079f6d19865362010bebcd36bfbbe0f8d)).
[^c-00f106c5]: opengfx1030/vllm-rdna commit [`00f106c5`](https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d) (2026-11-23 03:44 Paris): "Inkling-2-Mini: faster prefill attention at depth, DSpark recipes and adaptive K (#196)". Kernel facts and per-line permalinks are on the linked page.
[^c-f22c794a]: opengfx1030/vllm-rdna commit [`f22c794a`](https://github.com/opengfx1030/vllm-rdna/commit/f22c794a00e609fc8c03f35d374c5258f8e5187c) (2026-11-24 13:07 Paris), Merge #206.
[^c-ea8aa241]: opengfx1030/vllm-rdna merge [`ea8aa241`](https://github.com/opengfx1030/vllm-rdna/commit/ea8aa2418e6373de1537fa56e4ae0911ef6162a6) (2026-11-25 08:24 Paris), Merge #211 (range `1fdcb3a6...ea8aa241` also merges #209 and #210). Launch bounds, LDS arithmetic and per-line permalinks are on the linked page.
[^c-93c59de2]: opengfx1030/vllm-rdna merge [`93c59de2`](https://github.com/opengfx1030/vllm-rdna/commit/93c59de2b50ccea1b26f299e2ac5713861ffb2e3) (2026-11-25 21:29 Paris), Merge #217 (range `f6ed68b4...93c59de2` merges #213–#217; only #213 `8e9fbef` touches `csrc/`). Launch bounds, LDS arithmetic and per-line permalinks are on the linked page.
[^c-ba9b43bc]: opengfx1030/vllm-rdna merge [`ba9b43bc`](https://github.com/opengfx1030/vllm-rdna/commit/ba9b43bcd479acc4bddcc98a375a31152279a872) (2026-11-24 14:28 Paris), Merge #207 (content commit `08a30fad`). LDS arithmetic and per-line permalinks are on the linked page.
[^c-8d141bdc]: opengfx1030/vllm-rdna merge [`8d141bdc`](https://github.com/opengfx1030/vllm-rdna/commit/8d141bdc80b3124315a4a1e52ab4485cb6974f5e) (2026-11-27 08:45 Paris), merge of #229; range [`93c59de2...8d141bdc`](https://github.com/opengfx1030/vllm-rdna/compare/93c59de2b50ccea1b26f299e2ac5713861ffb2e3...8d141bdc80b3124315a4a1e52ab4485cb6974f5e) merges #221, #222, #224, #225, #228, #229. HIP changes are in `9a706f6a` + `cda6ef6e` (#221) and `750bbc8e` (#222). Launch bounds, LDS arithmetic and per-line permalinks are on the linked pages.
