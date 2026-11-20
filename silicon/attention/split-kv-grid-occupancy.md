---
title: "Split-KV / flash-decoding grid sizing vs CU fill (gfx1030)"
sources:
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/bc948335644816d5a24af618fbf4606366a7cd47
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/bc948335644816d5a24af618fbf4606366a7cd47/benchmarks/2026-11-19-band-split-blocks-per-sm.md
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/bc948335644816d5a24af618fbf4606366a7cd47/patches/README.md
  - https://github.com/ggml-org/llama.cpp/blob/5f837145d86ecc62333f3b83835c64275fdaa90e/ggml/src/ggml-cuda/ggml-cuda.cu#L410
  - https://github.com/ggml-org/llama.cpp/blob/5f837145d86ecc62333f3b83835c64275fdaa90e/ggml/src/ggml-cuda/fattn-common.cuh#L1254-L1282
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocsettings.cpp#L180-L187
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/utils/flags.hpp#L152-L153
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1150-L1161
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1712-L1732
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_device.cpp#L712-L715
  - https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_platform.cpp#L124-L140
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1155-L1162
  - https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/vllm/v1/attention/backends/rdna_attn.py#L484-L524
  - https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/fa_rdna2.cu#L3195-L3254
  - https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/fa_rdna2.cu#L98
  - https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/tools/rdna/rdna_launcher_common.sh#L137-L142
  - https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/mono_qsa_decode_rdna2.cu#L301-L307
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/fd6b1e1dd568f22a5f68ac47d6e905f626d7bd12/silicon/occupancy/vgpr-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/fd6b1e1dd568f22a5f68ac47d6e905f626d7bd12/silicon/occupancy/wgp-cu-mode-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/fd6b1e1dd568f22a5f68ac47d6e905f626d7bd12/silicon/occupancy/spi-ace-occupancy.md
---

# Split-KV / flash-decoding grid sizing vs CU fill (gfx1030)

Date: 2026-11-20 (idle HIP/ISA pass). Occupancy page, not a perf page. There are no tok/s numbers of our own here. Every number from stew675's tree is **theirs** and was measured on **gfx1201 (R9700)**, not on gfx1030. This page does **not** change extras HIP or open a ticket.

Companions: [fa-occupancy.md](fa-occupancy.md) (per-kernel LDS/VGPR/`waves_per_eu`), [fa-gqa.md](fa-gqa.md) (GQA decode kernel), [attention-dispatch.md](../../engine/attention/attention-dispatch.md) (decode vs prefill routing, "split-K = grid-fill"), [wgp-cu-mode-occupancy.md](../occupancy/wgp-cu-mode-occupancy.md), [spi-ace-occupancy.md](../occupancy/spi-ace-occupancy.md).

## What is new here (vs existing pages)

The existing pages already lock "split-K is for grid fill when B×H leaves CUs idle" ([attention-dispatch.md](../../engine/attention/attention-dispatch.md)). They also note that the HIP occupancy API counts 36 units on a V620 WGP build ([wgp-cu-mode-occupancy.md](../occupancy/wgp-cu-mode-occupancy.md)).[^wiki-wgp] This page adds four things:

1. A CLR source proof that `hipDeviceProp_t::multiProcessorCount` (llama.cpp's `nsm`) counts **WGPs** in default WGP mode.
2. The r322 band-grid rule and why it exists: the per-device tile count falls under a tensor split.
3. That rule ported to gfx1030 numbers.
4. A read-only check of how the dest decode kernels pick `kv_splits`.

## Take / Leave

| Verdict | Item |
|---|---|
| **Take Now** (fact) | On gfx1030 in default WGP mode, HIP `multiProcessorCount` = CU count / 2. On a 72-CU V620 that is **36** (WGPs). Any CUDA-ported `nsm`-based grid formula therefore works in **WGP** units.[^clr-wgp-flag][^clr-settings][^clr-cu-halve][^clr-props][^llama-nsm] |
| **Import Now** (principle, wiki only) | Size the split-KV count from the **per-device** tile count, `tiles × P ≥ fill target`. Do not use a flat per-device constant. A tensor split cuts the K/V heads per device, so it cuts tiles, and a fixed P then leaves units idle.[^r322-commit][^r322-bench] |
| **Leave** | stew675's band kernel itself. It is an RDNA4 **WMMA** kernel ("RDNA4 band", gfx1201), and gfx1030 has no WMMA. Their `16 warps / SM` saturation constant is a gfx1201 measurement, so it is not a gfx1030 number.[^r322-bench] |
| **Later** (measure first) | A tile-aware `kv_splits` for the dest GQA decode path at TP-split shapes with `H_kv = 1` per rank. Today it is fixed at 16, capped by `MAX_SPLITS = 16`. Our arithmetic below shows a small shortfall against the *ported* rule, but the rule has never been measured on gfx1030. No extras change, no card.[^dest-attn][^dest-fa-launch][^dest-maxsplits] |
| **Dead** | Raising `P` to "fix occupancy" for shapes that already have ≥ 4 tiles per device. r322 keeps `nsm/2` as the floor and only ever raises `P`.[^r322-commit] |

## 1. The r322 formula, in their words

stew675's band FA launches `gridDim = (tiles, P)`, with `tiles = ntiles_z_gqa × K->ne[2] × Q->ne[3]` (the output tiles **this device's slice** gets). `P` is the KV interleave period, i.e. the number of blocks per output tile. r322 changes `P` from r316's flat `max(2, nsm/2)` to:[^r322-commit][^r322-bench]

```
P = max(nsm/2, ceil(nsm * blocks_per_sm / tiles)),   blocks_per_sm = max(1, 16 / nwarps)
```

- In the code, `need = (nsm*bps + ntiles - 1) / ntiles` and `return max(max(2, nsm/2), need)`. The call site passes `ntiles_z_gqa * K->ne[2] * Q->ne[3]` and `nwarps`.[^r322-commit]
- Their reasoning: a `-sm tensor` split cuts `K->ne[2]` (K/V heads), so adding devices **lowers** `tiles`. Their example: qwen4exp (2 K/V heads) has 4 tiles on one GPU and 2 on two, and 27B (4 K/V heads) has 2/1/1 on three.[^r322-commit]
- On gfx1201 (`nsm` 32), 2 tiles × P 16 = 32 blocks on 32 "SMs", one block per SM out of a runtime 3-block limit. At 1 tile, half the SMs are idle.[^r322-bench]
- **Their numbers, attributed:** a 2-device qwen4exp decode lost 4–9 % with `P` 16 (their issue #333). After r322, plain decode measured +2.8 % at 65k and +5.4 % at 105k on 2× R9700 `-sm tensor` q8_0, and the 3-device 27B arm was flat (+0.4 %). Per-op, the new default lands on the measured optimum at 1/2/4/8/16 tiles (P 64/32/16/8/4).[^r322-commit][^r322-patches-readme]
- A purity property they keep: `P` depends only on K/V heads, the GQA ratio and `Q->ne[3]`, **never** on `n_q` or KV length. So a decode and its own verify reduce in the same order.[^r322-commit]

Upstream llama.cpp's tile FA path makes the same move for its `parallel_blocks`: it searches upward from `max_blocks_per_sm` while `ntiles_dst × parallel_blocks` fills whole waves of `nsm × max_blocks_per_sm`.[^llama-fattn] r322 applies that "fill the machine from the tile count" idea to a kernel whose `P` must stay a constant per shape.

## 2. What `nsm` is on gfx1030 (HIP/CLR source, `rocm-systems` main `3b37ab00`)

| Step | Source |
|---|---|
| `GPU_ENABLE_WGP_MODE` defaults **true** ("Enables WGP Mode in HW if available") | [^clr-wgp-flag] |
| For `gfxipMajor >= 10`, `enableWgpMode_ = GPU_ENABLE_WGP_MODE`. Only gfx12.5+ forces it off. gfx1030 (10.3) is therefore WGP by default | [^clr-settings] |
| `maxComputeUnits_` comes from HSA (cooperative CU count for HIP), then `= enableWgpMode_ ? maxComputeUnits_/2 : maxComputeUnits_` | [^clr-cu-halve] |
| `deviceProps.multiProcessorCount = info.maxComputeUnits_` | [^clr-props] |
| llama.cpp: `info.devices[id].nsm = prop.multiProcessorCount` | [^llama-nsm] |

So on a 72-CU V620,[^wiki-spi] `nsm = 36`. This is consistent with the 36 the wiki already recorded from the occupancy API.[^wiki-wgp] stew675's own gfx1201 record says `nsm` 32,[^r322-bench] which is the same halving applied to a 64-CU part. That CU count is (unsourced) here and was not checked this pass. Only the `rocm-systems` main snapshot was checked; older ROCm releases were not (unsourced for those).

The per-"SM" capacities follow the same unit:

- CLR doubles `simdPerCU_` and `max_waves_per_cu` in WGP mode, so `maxThreadsPerMultiProcessor` is per **WGP**.[^clr-waves][^clr-props]
- The occupancy helper says "the count above is per scheduling unit of the kernel: a WGP for a WGP mode kernel". It rescales ×2 or ÷2 when the code object's mode differs from the device setting.[^clr-occ]
- LLVM's `getMaxWorkGroupsPerCU` is `getMaxWavesPerEU × getNumWorkGroupSIMDs(isFullSIMDMode)`, which gives 4 SIMDs per WG in WGP mode and 2 in CU mode.[^llvm-wgpercu] gfx1030 max waves/SIMD32 is **16**,[^wiki-vgpr] so a WGP holds at most 4 × 16 = 64 waves. The wiki's whole-device figure is 36 × 4 × 16 = 2304.[^wiki-spi]

**Caveat already on the wiki:** on a V620 WGP build, the occupancy API was measured to under-report LDS residency 2×. It counted 36 units at 64 KiB, while 72 × 64 KiB WGs were co-resident.[^wiki-wgp] A port that computes `nsm × max_blocks_per_sm` from the HIP API can therefore *under*-estimate one full wave on gfx1030. That only makes a fill formula err toward more splits, not fewer. This is an inference from the two cited facts and was not measured.

## 3. Why few KV heads per device idles WGPs (gfx1030 units)

The grid is `tiles × P` workgroups. In flash-decoding at B = 1, the tile count is the head count the kernel parallelizes over: K/V heads for a GQA-grouped kernel, Q heads for a per-head kernel. Under TP, each rank owns `H / TP` heads. A GQA model with 4–8 K/V heads at TP = 4 leaves **1–2 K/V heads per device**. With a fixed `P`, the grid then shrinks by the same factor, and once `tiles × P < 36` some WGPs get no workgroup at all. Long KV cannot fix this, because more KV only lengthens each block's loop.[^r322-bench]

The r322 rule ported to gfx1030 (`nsm = 36`; **arithmetic only, not measured on gfx1030**):

| block width | `blocks_per_sm` | tiles 1 | 2 | 4 | ≥ 8 |
|---|---|---|---|---|---|
| 8 waves (256 thr) | 2 | P 72 | 36 | 18 | 18 (floor `nsm/2`) |
| 4 waves (128 thr) | 4 | P 144 | 72 | 36 | 18 |

"~16 warps per SM" is 16 waves per WGP, or 4 of 16 waves per SIMD32. That is their gfx1201 saturation point.[^r322-bench] Whether gfx1030 saturates at the same wave count is (unsourced).

## 4. Dest RDNA decode (read-only, `opengfx1030/vllm-rdna` `rdna_extras` @ `1e1cac53`)

- `RDNA_ATTN` calls `fa_rdna2_decode_paged(..., kv_splits=16, ...)` for pure decode and for split-decode verify rows. The comment records a 2026-09-04 sweep ("s16 >= s8 … decode CTAs are few (B*H_q) so more splits = more occupancy").[^dest-attn] `MAX_SPLITS = 16`.[^dest-maxsplits]
- The **per-head** kernels use `grid1(num_tokens, H_q, kv_splits)`, so the tile axis is **Q heads**, not K/V heads.[^dest-fa-launch] TP divides `H_q` too, but per-rank `H_q` is `G ×` larger than per-rank `H_kv`. Example from the dest comment: a Qwen3.8-27B rank is H_q 6 / H_kv 1, which gives 6 × 16 = 96 WGs at B = 1, ≥ 72 (ported 8-wave target).[^dest-attn]
- The **GQA** kernel (`fa_decode_paged_splitk_gqa_kernel_256`, D = 256, 1 < G ≤ GQA_MAX_G) uses `grid_gqa(num_tokens, H_kv, kv_splits)`. It is only taken when `num_tokens * H_kv >= 4`, so at H_kv = 1 per rank, B < 4 falls back to the per-head kernel.[^dest-fa-launch] The env switch is opt-in in the kernel, but `rdna_select_attention` exports `VLLM_FA_RDNA2_GQA_DECODE=${…:-1}`, so the launcher turns it on by default.[^dest-launcher]
- So the dest's smallest GQA grid is 4 tiles × 16 = **64** WGs. That is the shape r316/r322 called saturated on gfx1201 (4 × 16 = 64). Under the *ported* rule (P 18 at 4 tiles), 64 is 72 − 8 WGs short. Even so, all 36 WGPs get ≥ 1 WG (64 / 36 ≈ 1.8). A larger P is impossible without raising `MAX_SPLITS`. This is arithmetic; the gap has not been measured. → **Later, measure first; no change.**
- The fixed `kv_splits=16` is independent of `n_q` and KV length, which is the same purity property r322 keeps.[^dest-attn][^r322-commit]
- The QSA sparse decode helper already scales its split count by tiles: `s = 288 / max(1, M*KVH)`, clamped to `[1, kMaxS]`, then capped by window work.[^dest-qsa] (288 = 8 × 36 numerically; the intended meaning is (unsourced).) It also depends on `M` (rows in the batch) and `W`, so by r322's own argument a decode and a multi-row verify there can reduce in a different order. Whether dest requires byte purity on that path is (unsourced). **Leave** (observation only).

## Sources

1. stew675/llama-cpp-rdna-boosts r322 commit `bc948335644816d5a24af618fbf4606366a7cd47` (2026-11-19T23:07Z = 2026-11-20 00:07 CET) — https://github.com/stew675/llama-cpp-rdna-boosts/commit/bc948335644816d5a24af618fbf4606366a7cd47
2. Their benchmark record at that commit — https://github.com/stew675/llama-cpp-rdna-boosts/blob/bc948335644816d5a24af618fbf4606366a7cd47/benchmarks/2026-11-19-band-split-blocks-per-sm.md
3. Their patches/README entry at that commit — https://github.com/stew675/llama-cpp-rdna-boosts/blob/bc948335644816d5a24af618fbf4606366a7cd47/patches/README.md
4. llama.cpp `nsm = prop.multiProcessorCount` — https://github.com/ggml-org/llama.cpp/blob/5f837145d86ecc62333f3b83835c64275fdaa90e/ggml/src/ggml-cuda/ggml-cuda.cu#L410
5. llama.cpp tile FA `parallel_blocks` wave-efficiency search — https://github.com/ggml-org/llama.cpp/blob/5f837145d86ecc62333f3b83835c64275fdaa90e/ggml/src/ggml-cuda/fattn-common.cuh#L1254-L1282
6. CLR `GPU_ENABLE_WGP_MODE` default — https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/utils/flags.hpp#L152-L153
7. CLR WGP mode for gfx10+ — https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocsettings.cpp#L180-L187
8. CLR `maxComputeUnits_` halving — https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1150-L1161
9. CLR waves/SIMDs doubled in WGP mode — https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1712-L1732
10. HIP `multiProcessorCount` / `maxThreadsPerMultiProcessor` — https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_device.cpp#L712-L715
11. HIP occupancy helper per scheduling unit — https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_platform.cpp#L124-L140
12. LLVM `getMaxWorkGroupsPerCU` — https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1155-L1162
13. Dest `rdna_attn.py` decode `kv_splits=16` — https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/vllm/v1/attention/backends/rdna_attn.py#L484-L524
14. Dest `fa_rdna2.cu` decode grids and GQA gate — https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/fa_rdna2.cu#L3195-L3254
15. Dest `MAX_SPLITS` — https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/fa_rdna2.cu#L98
16. Dest launcher GQA decode default — https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/tools/rdna/rdna_launcher_common.sh#L137-L142
17. Dest QSA split helper — https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/mono_qsa_decode_rdna2.cu#L301-L307
18. Wiki pages at `fd6b1e1d`: vgpr-occupancy, wgp-cu-mode-occupancy, spi-ace-occupancy (links below)

[^r322-commit]: stew675/llama-cpp-rdna-boosts r322, commit message and `ggml_cuda_fattn_band_wmma_split()` diff. https://github.com/stew675/llama-cpp-rdna-boosts/commit/bc948335644816d5a24af618fbf4606366a7cd47 (checked 2026-11-20)
[^r322-bench]: stew675 benchmark record `2026-11-19-band-split-blocks-per-sm.md` (gfx1201, `nsm` 32, 8-warp block `max_blocks_per_sm` 3). https://github.com/stew675/llama-cpp-rdna-boosts/blob/bc948335644816d5a24af618fbf4606366a7cd47/benchmarks/2026-11-19-band-split-blocks-per-sm.md (checked 2026-11-20)
[^r322-patches-readme]: stew675 `patches/README.md` r322 entry. https://github.com/stew675/llama-cpp-rdna-boosts/blob/bc948335644816d5a24af618fbf4606366a7cd47/patches/README.md (checked 2026-11-20)
[^llama-nsm]: llama.cpp `ggml-cuda.cu` L410. https://github.com/ggml-org/llama.cpp/blob/5f837145d86ecc62333f3b83835c64275fdaa90e/ggml/src/ggml-cuda/ggml-cuda.cu#L410 (checked 2026-11-20)
[^llama-fattn]: llama.cpp `fattn-common.cuh` L1254–L1282. https://github.com/ggml-org/llama.cpp/blob/5f837145d86ecc62333f3b83835c64275fdaa90e/ggml/src/ggml-cuda/fattn-common.cuh#L1254-L1282 (checked 2026-11-20)
[^clr-wgp-flag]: CLR `flags.hpp` `GPU_ENABLE_WGP_MODE` default true. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/utils/flags.hpp#L152-L153 (checked 2026-11-20)
[^clr-settings]: CLR `rocsettings.cpp` gfx10+ WGP mode. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocsettings.cpp#L180-L187 (checked 2026-11-20)
[^clr-cu-halve]: CLR `rocdevice.cpp` `maxComputeUnits_` /2 in WGP mode. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1150-L1161 (checked 2026-11-20)
[^clr-waves]: CLR `rocdevice.cpp` `simdPerCU_` and `max_waves_per_cu` ×2 in WGP mode; `maxThreadsPerCU_`. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1712-L1732 (checked 2026-11-20)
[^clr-props]: HIP `hip_device.cpp` `multiProcessorCount`, `maxThreadsPerMultiProcessor`. https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_device.cpp#L712-L715 (checked 2026-11-20)
[^clr-occ]: HIP `hip_platform.cpp` occupancy per scheduling unit (WGP vs `-mcumode`). https://github.com/ROCm/rocm-systems/blob/3b37ab00fa01a1af23833318d375b6ce74b59299/projects/clr/hipamd/src/hip_platform.cpp#L124-L140 (checked 2026-11-20)
[^llvm-wgpercu]: LLVM `AMDGPUBaseInfo.cpp` `getMaxWorkGroupsPerCU`. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1155-L1162 (checked 2026-11-20)
[^dest-attn]: dest `rdna_attn.py` decode / split-decode calls with `kv_splits=16` and the 2026-09-04 sweep comment. https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/vllm/v1/attention/backends/rdna_attn.py#L484-L524 (checked 2026-11-20)
[^dest-fa-launch]: dest `fa_rdna2.cu` `grid1(num_tokens, H_q, kv_splits)` and the GQA branch `num_tokens * H_kv >= 4`, `grid_gqa(num_tokens, H_kv, kv_splits)`. https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/fa_rdna2.cu#L3195-L3254 (checked 2026-11-20)
[^dest-maxsplits]: dest `fa_rdna2.cu` `constexpr int MAX_SPLITS = 16`. https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/fa_rdna2.cu#L98 (checked 2026-11-20)
[^dest-launcher]: dest `rdna_launcher_common.sh` `rdna_select_attention` exports `VLLM_FA_RDNA2_GQA_DECODE=${VLLM_FA_RDNA2_GQA_DECODE:-1}`. https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/tools/rdna/rdna_launcher_common.sh#L137-L142 (checked 2026-11-20)
[^dest-qsa]: dest `mono_qsa_decode_rdna2_splits`. https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/mono_qsa_decode_rdna2.cu#L301-L307 (checked 2026-11-20)
[^wiki-vgpr]: wiki vgpr-occupancy.md (max waves/SIMD 16 on gfx1030, from LLVM `getMaxWavesPerEU`). https://github.com/BlivionIaG/rdna-hip-wiki/blob/fd6b1e1dd568f22a5f68ac47d6e905f626d7bd12/silicon/occupancy/vgpr-occupancy.md
[^wiki-wgp]: wiki wgp-cu-mode-occupancy.md (occupancy API counts 36 units on V620 WGP build, under-reports LDS residency 2×). https://github.com/BlivionIaG/rdna-hip-wiki/blob/fd6b1e1dd568f22a5f68ac47d6e905f626d7bd12/silicon/occupancy/wgp-cu-mode-occupancy.md
[^wiki-spi]: wiki spi-ace-occupancy.md (V620 72 CU / 36 WGP, fill 36 × 4 × 16 = 2304). https://github.com/BlivionIaG/rdna-hip-wiki/blob/fd6b1e1dd568f22a5f68ac47d6e905f626d7bd12/silicon/occupancy/spi-ace-occupancy.md
