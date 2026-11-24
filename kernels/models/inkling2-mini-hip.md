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
  - https://github.com/opengfx1030/vllm-rdna/commit/f22c794a00e609fc8c03f35d374c5258f8e5187c
  - https://github.com/opengfx1030/vllm-rdna/commit/c1ce3ec355f946849fee47be9fbae111e9a6fcbc
  - https://github.com/opengfx1030/vllm-rdna/compare/d71bddca6e0f9d9f22390b8dde758cf7baf60481...f22c794a00e609fc8c03f35d374c5258f8e5187c
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/csrc/rocm/inkling2_glue_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/d71bddca6e0f9d9f22390b8dde758cf7baf60481/csrc/rocm/inkling2_glue_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/docs/rdna2/inkling2-decode-glue.md
  - https://github.com/opengfx1030/vllm-rdna/blob/f22c794a00e609fc8c03f35d374c5258f8e5187c/tests/kernels/core/test_inkling2_glue_rdna2.py
  - https://github.com/llvm/llvm-project/blob/278221a5b0be60d025312c5e007a4c4dc01eb59a/llvm/lib/Target/AMDGPU/SIDefines.h
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---

# Inkling-2-Mini HIP kernels on gfx1030 — tip `00f106c5`

Native HIP serving kernels for Inkling-2-Mini (`inkling2_loop`) on dest `rdna_extras`, read at tip [`00f106c5`](https://github.com/opengfx1030/vllm-rdna/commit/00f106c5b035bd6d562267bfee9e4ac020b7055d) (2026-11-23 03:44 Paris, merge of PR #196). Previous watched tip: `1e1cac53`.[^tip][^cmp] Glue section updated at `f22c794a` (2026-11-24).[^tip3] Back to the [models index](README.md).

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

- The `gate_mix` R-split kernel is 8 waves (`kMixWaves = 8`, `kMixRpw = 8`, R = 64). It is taken when `VLLM_INKLING2_FUSE_GLUE` is on, the weights are fp32 and T ≤ 8 (update @ `f22c794a`, below). *Historical @ `00f106c5` … `d71bddca`:* the limit was T ≤ 4.[^glue-mix][^g2-split][^g2-old-split]
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
| `VLLM_INKLING2_FUSE_GLUE` | on | SwiGLU fused into the w2 expert GEMM (≤ 16 tokens), 8-wave `gate_mix` split (T ≤ 8 since `f22c794a`), `moe_route_small_kernel` (T ≤ 4, since `f22c794a`). Not the DPP / `attn_res` changes, which are unconditional on gfx1030 | Python + C++ host, once [^py-glue][^glue-env][^py-moe-swiglu][^g2-env] |
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
