---
title: "FA prefill cols_per_block / tile caps vs occupancy (gfx1030): stew675 r334 and the dest tile"
sources:
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/cca23942f10f660e898d49b99a2e22bda1b40f16
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/WORKLOG.md#L1-L58
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/INVARIANTS.md#L1577-L1598
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L1985-L2012
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L4711-L4745
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L5470-L5477
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/a28d4e699041e04fa10c5b5365ab2e1d39172032/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L5470-L5530
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/a28d4e699041e04fa10c5b5365ab2e1d39172032/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L1985-L2012
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/a28d4e699041e04fa10c5b5365ab2e1d39172032/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L5596
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L10-L18
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L280-L285
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L870-L890
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L1158-L1231
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-common.cuh#L1197-L1201
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/common.cuh#L280-L286
  - https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/common.cuh#L424-L426
  - https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/hipamd/src/hip_platform.cpp#L45-L122
  - https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1728-L1731
  - https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1804
  - https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/hipamd/src/hip_device.cpp#L742
  - https://github.com/ROCm/rocm-systems/commit/b9ec5a21d702ebdbb681304a0b054acf7f1ab711
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L96-L100
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L3534-L3574
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L3778-L3790
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L1737-L1746
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L3966-L4002
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L4104-L4110
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L4400-L4410
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L4448-L4483
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/vllm/v1/attention/backends/rdna_attn.py#L94-L95
  - https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/vllm/v1/attention/backends/rdna_attn.py#L814-L894
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/77ace785ab8cdf311300a7f23f45f58c8844590b/silicon/occupancy/vgpr-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/77ace785ab8cdf311300a7f23f45f58c8844590b/silicon/occupancy/wgp-cu-mode-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/77ace785ab8cdf311300a7f23f45f58c8844590b/silicon/attention/fa-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/edd25f7bf039bba64075b6ddcb813f6b475b7305/silicon/attention/split-kv-grid-occupancy.md
---

# FA prefill `cols_per_block` / tile caps vs occupancy (gfx1030)

Date: 2026-11-23 (idle HIP/ISA pass). This is an occupancy page, not a perf page. It has no tok/s or speedup figures. The contributor's V620 measurements live in the fork's WORKLOG and are not repeated here. This page does **not** change extras HIP, open a ticket or touch the board.

Trigger: stew675/llama-cpp-rdna-boosts **r334** `cca23942` (contributor PR #338, folded into blocks 05/08/10, 2026-11-21 UTC). It deleted the r76-era gfx1030-only FA `cols_per_block` caps and `rdna_cap_ncols2`.[^r334-commit][^r334-worklog]

Companions: [fa-occupancy.md](fa-occupancy.md) (per-kernel LDS/VGPR table for the dest FA kernels), [fa-gqa.md](fa-gqa.md) (GQA prefill instantiations), [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md), [wgp-cu-mode-occupancy.md](../occupancy/wgp-cu-mode-occupancy.md). The decode-side follow-up (split-KV grid sizing from stew675 r322) is on the unmerged branch `idle/split-kv-grid-occupancy-2026-11-20`: [split-kv-grid-occupancy.md @ edd25f7b](https://github.com/BlivionIaG/rdna-hip-wiki/blob/edd25f7bf039bba64075b6ddcb813f6b475b7305/silicon/attention/split-kv-grid-occupancy.md).[^wiki-splitkv] This page covers the prefill tile only.

## Verdicts

| Verdict | Item |
|---|---|
| **Take Now** (fact) | The r334 caps were a **crash workaround, not an occupancy tune**. They existed because `launch_fattn` hits `GGML_ASSERT(max_blocks_per_sm > 0)` when HIP's occupancy query returns 0 for the wide tiles. The fork's own RDNA2 occupancy estimator (from VGPRs and LDS against a WGP) was **already in the tree** at the parent commit, so the caps had become dead weight before r334 removed them.[^r334-invariants][^fork-est-parent][^fork-caps-parent][^llama-assert] |
| **Take Now** (fact) | What the caps cost was **KV reuse per block**, not resident waves. llama.cpp's tile kernel loads each `nbatch_fa`-row K/V slab into LDS once per block and uses it for all `cols_per_block` Q columns. At an 8-column cap, each K/V slab is staged 8× as often per output column as at 64 columns (arithmetic only). The fork names the same two costs: "8-column prefill tiles where every other arch takes 32/64", and a GQA-8 D=128 decode "re-reading the KV cache 4x".[^r334-worklog][^llama-smem][^llama-select] |
| **Take Now** (fact, runtime) | On `rocm-systems` develop `9a92c8ba`, HIP's occupancy routine does **not** read `regsPerMultiprocessor`. It divides `vgprsPerSimd_` by the granule-rounded VGPR count, doubles SIMDs in WGP mode, and (since #9571, 2026-08-04 UTC) doubles the LDS pool in WGP mode. The fork's "HIP assumes `regsPerMultiprocessor = 32768`" therefore describes an older runtime. Which ROCm the contributor ran is not stated (unsourced), and which release first ships #9571 was not checked.[^clr-occ][^clr-9571][^fork-est] |
| **Leave** | The fork's manual estimator as code. Dest FA prefill never queries occupancy at run time: its tiles are compile-time constants and there is no assert path to protect. The estimator also rounds VGPRs to 8 where the wiki's gfx1030 wave32 granule is 16, so it can sit one rung high (arithmetic only).[^fork-est][^wiki-vgpr][^dest-consts] |
| **Later** (measure first) | The "audit dest RDNA_ATTN prefill tile caps" item. The default dest prefill (`VLLM_FA_RDNA2_GQA_MODE=subgroup`) is the GQA kernel. **Every** instantiation holds **16 query rows per CTA** (2×8, 1×16, bf16 4×4) and steps K/V by `BC = 16`. That is the same width class as the caps r334 deleted. LDS is **not** what holds it there: the D=128 instance uses about 13.7 KiB, and a 32-row variant would still allow 6 CTAs per WGP by LDS (arithmetic only). The binding term is VGPRs, because O and the softmax state live in registers, and the wiki has no `.vgpr_count` for this kernel yet. Next step: dump `.vgpr_count` per instantiation, then decide. Note that the dest source records 6 heads per CTA as measured **slower** at D=256. Wider is not free here. No extras change, no card.[^dest-gqa][^dest-gqa-host][^dest-dispatch][^dest-mode] |
| **Dead** | Porting any `cols_per_block` cap or "cap the tile for occupancy" rule into dest. The failure mode it guarded (runtime occupancy query → 0 → abort) does not exist in `fa_rdna2.cu`'s prefill launchers.[^dest-plain][^dest-gqa-host] |

The other three r334 items are logged as **Leave** (awareness only), since each is a llama.cpp dispatch or kernel detail with no dest counterpart named in the commit. They are a one-warp `RMS_NORM` for `ncols <= 128` (bit-identical by `block_reduce` butterfly order), the RDNA2 GDN chunked-prefill gate `H * n_seqs <= 4 && n_tokens > 256` (motivated by "no BF16 WMMA on gfx1030"), and `mul_mat_vec_q_moe` rows-abreast (bit-identical). All three are compile-gated to the RDNA2 device pass and have run-time kill switches.[^r334-worklog][^r334-invariants]

## 1. What r334 deleted

Pre-r334, `launch_fattn_tile_switch_ncols1` gated each wider tile on four RDNA2 booleans. The parent-commit comment gives the reasons:[^fork-caps-parent]

| Shape | Cap | Reason given in the deleted comment |
|---|---|---|
| D=512 (Gemma global) | `cols_per_block <= 8` | `ncols2 <= 2` kernels not instantiated for DV=512, keep accumulators in registers |
| D=128 | `cols_per_block <= 8` (plus `ncols2 <= 2` from `rdna_cap_ncols2`) | "cols_per_block 16/32/64 need 33-43 KB LDS" |
| D=576/DV=512 (MLA) | `cols_per_block <= 16` | `<512,8,4>` (32 cols) "fails" |
| D=256 MHA / odd GQA (`ncols2 == 1`) | `cols_per_block <= 8` | `<256,32,1>` "fails" |

The header of that block says they were needed because, "even at occupancy 1 … some wide-column kernels still fail the occupancy query and abort".[^fork-caps-parent] `rdna_cap_ncols2` (`RDNA2 && DKQ == 128`) blocked the `ncols2 = 8/4` GQA variants.[^fork-ncols2-parent] r334 deletes all of these and replaces them with a comment: the assert "is now unreachable on RDNA2" because `launch_fattn` computes occupancy against a WGP and only falls back to HIP's figure when that is larger.[^fork-caps-new]

The fork's INVARIANTS entry is careful about scope. The assert stays reachable whenever `4*waves_per_simd < waves_per_block`, and the deciding evidence is the contributor's full `FLASH_ATTN_EXT` suite, not the deletion itself.[^r334-invariants]

## 2. Why the query returned 0 (the fork's account, and current CLR)

Upstream llama.cpp seeds `max_blocks_per_sm` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (HIP's occupancy API on ROCm) and asserts it is `> 0`.[^llama-assert] The fork's RDNA2 block (present at both `a28d4e69` and `cca23942`) explains the zero as follows: HIP assumes `regsPerMultiprocessor = 32768` on RDNA2, "a quarter of a WGP's real register file (4 SIMDs x 1024 VGPRs x 32 lanes in wave32)". It therefore "reports 0 blocks for kernels above 128 VGPRs that the hardware runs".[^fork-est][^fork-est-parent]

Applying that stated 32768 figure (arithmetic only), a block of `T` threads at `V` VGPRs fits if `T × V ≤ 32768`. A 128-thread block fits up to 256 VGPRs (the wave32 addressable maximum[^wiki-vgpr]), so it can **never** report 0. A 256-thread block reports 0 above 128 VGPRs. This lines up with the fork's own RDNA2 table: the 8-column D=128 entry is a **128-thread** config, while 16/32/64 columns are **256-thread**.[^fork-table] The fit is consistent but not a proof, because the VGPR counts of those kernels were not checked this pass.

**Current CLR is different.** At `rocm-systems` develop `9a92c8ba`, `ihipOccupancyMaxActiveBlocksPerMultiprocessor` works like this:[^clr-occ]

- `VgprWaves = vgprsPerSimd_ / roundUp(usedVGPRs_, VgprGranularity)`, capped at `MaxWavesPerSimd = 16` for gfx10+.
- `simdPerCU` is doubled when the kernel's `isWGPMode_` is set.
- `lds_pool_size = localMemSizePerCU_ × 2` in WGP mode, and LDS is rounded to the ISA's LDS alignment.
- `maxBlocksPerCU = min(alu_limited_threads / aligned_block, lds_pool / used_lds)`.

`regsPerMultiprocessor` is only a reported property: `vgprsPerSimd_ × simdPerCU_ × wavefrontWidth_`, with `simdPerCU_` already doubled in WGP mode.[^clr-regs][^clr-simd-wgp] The WGP-mode LDS fix is #9571. Its stated motivation is gfx11, but the code keys on the WGP mode flag, not the arch.[^clr-9571] So on a runtime that carries #9571, the zero the caps guarded against may not reproduce at all (unverified on V620). This also bears on the "API counts 36 units at 64 KiB … under-reports 2×" note in [wgp-cu-mode-occupancy.md](../occupancy/wgp-cu-mode-occupancy.md), which was measured on an earlier stack.[^wiki-wgp]

### The fork's replacement estimator (read, not adopted)

```
waves_per_block = ceil(nthreads / 32)
vgprs           = max(8, roundUp(numRegs, 8))
waves_per_simd  = min(8, 1024 / vgprs)          // 8-cap: their decode measurement
blocks          = 4 * waves_per_simd / waves_per_block
blocks          = min(blocks, 128 KiB / (static LDS + dynamic LDS))
use it only if blocks > HIP's figure
```
[^fork-est]

Notes (arithmetic only):

- With `nthreads ≤ 512` (static-asserted in the tile config macro[^llama-macro]) and `numRegs ≤ 256`, `waves_per_simd ≥ 4` and `waves_per_block ≤ 16`, so `blocks ≥ 1`. For the tile kernel, the INVARIANTS abort case can only arise if `numRegs` were reported above 256.
- Rounding to 8 differs from the gfx1030 wave32 VGPR granule of 16 recorded in [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md).[^wiki-vgpr] At 200 VGPRs the estimator gets 1024/200 → 5 waves, while the 16-granule rule gives 1024/208 → 4.
- The 8 waves/SIMD cap is the fork's choice for split-KV decode (more resident blocks picked a worse KV split in their measurement). It is not a hardware limit. CLR uses 16.[^fork-est][^clr-occ]

## 3. The llama.cpp tile kernel's LDS per block (gfx1030 path)

On HIP, `FAST_FP16_AVAILABLE` is defined and `ggml_cuda_get_max_cpy_bytes()` returns 16, so `cpy_ne = 4`.[^llama-fp16][^llama-cpy] The kernel's static LDS is then:[^llama-smem]

```
Q_tmp  : half2[ncols * DKQ/2]                         -> ncols*DKQ*2 B
KV_tmp : half2[nbatch_fa * (nbatch_K/2 + 4) + DVp-DV] -> 4 B each
KQ     : half [ncols * nbatch_fa]                     -> 2 B each
```

`nbytes_shared = 0` is passed (no dynamic LDS).[^llama-select] Using the fork's RDNA2 table rows `(DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K)`[^fork-table] (arithmetic only; `DVp = DV` for 128/256/512):

| Tile (DKQ, ncols) | threads → waves | Q_tmp | KV_tmp | KQ | LDS / block | blocks / WGP by LDS (128 KiB) |
|---|---|---|---|---|---|---|
| 128, 8 (old cap) | 128 → 4 | 2048 | 9216 | 1024 | **12288** | 10 |
| 128, 16 | 256 → 8 | 4096 | 34816 | 4096 | **43008** | 3 |
| 128, 32 | 256 → 8 | 8192 | 18432 | 8192 | **34816** | 3 |
| 128, 64 | 256 → 8 | 16384 | 9216 | 8192 | **33792** | 3 |
| 512, 32 | 256 → 8 | 32768 | 18432 | 8192 | **59392** | 2 |
| 576/512, 32 | 256 → 8 | 36864 | 18432 | 8192 | **63488** | 2 |

- The D=128 rows reproduce the deleted comment's "33-43 KB" exactly (33792–43008 B). Every row fits under 64 KiB per block. **LDS never made these zero.** Under a CU-sized 64 KiB pool (pre-#9571 accounting) the wide D=128 tiles get 1 block, and under the WGP pool they get 3.[^fork-caps-parent][^clr-9571]
- Every row in the fork's RDNA2 table passes `occupancy = 1` to `__launch_bounds__`. Upstream's RDNA table uses 3 for the same D=128 16/32/64 shapes.[^fork-table][^llama-rdna-table] The fork's own comment says the occupancy-1 entries "date from when HIP's occupancy query … made larger configs abort".[^fork-table]
- Selection: on HIP, `DKQ ≤ 128` takes 64 columns when `Q->ne[1] > 32/ncols2`, then 32 when `> 16/ncols2`, then 16, 8, 4, 2.[^llama-select] So for any real prefill, the cap pinned D=128 to the 8-column (4-wave) tile.

**Occupancy vs reuse (arithmetic only).** The capped 8-column D=128 tile is cheap on LDS (12 KiB, 10 per WGP by LDS) and on waves (4 per block). It did not starve WGP slots. What it gave up is K/V reuse: each K/V slab staged into `KV_tmp` serves 8 Q columns instead of 64. That is the "pure loss" the fork describes. Occupancy entered only through the assert.[^r334-worklog][^llama-smem]

## 4. Dest RDNA_ATTN prefill tile (read-only, `rdna_extras` @ `d0b0c2b9`)

Dispatch: `_forward_prefill_kernels` takes the GQA kernel when `num_heads % num_kv_heads == 0` and (`VLLM_FA_RDNA2_GQA_MODE == "subgroup"` (the default), or bf16, or non-causal). Otherwise it uses `_short` (D=128, `max_seqlen_k < 4096`), split-K, or the plain per-head kernel.[^dest-dispatch][^dest-mode]

| Kernel (dest) | Q rows / CTA | KV step | threads → waves | dynamic LDS (B) | blocks / WGP by LDS (arith.) |
|---|---|---|---|---|---|
| GQA `<128, 2, 8>` / `<128, 1, 16>` / bf16 `<128, 4, 4>` | **16** | 16 | 256 → 8 | 13824 | 9 |
| GQA `<256, 2, 8>` / `<256, 1, 16>` / bf16 `<256, 4, 4>` | **16** | 16 | 256 → 8 | 26112 | 4 |
| plain `fa_prefill_paged_varlen_kernel_128` | 16 | 64 | 128 → 4 | 49364 | 2 |
| `_short` D=128 | 32 | 32 | 256 → 8 | 45476 | 2 |
| plain D=256 | 16 | 32 | 256 → 8 | 60644 (+ 256 static)[^dest-256-static] | 2 |

Sources: the GQA instantiations and the "a CTA always holds 16 query rows" comment;[^dest-gqa][^dest-gqa-host] the GQA LDS formula `HEADS·BR·D·2 + BC·(D+8)·2·2 + HEADS·BR·BC·4` with "O and the softmax state live in registers";[^dest-gqa-host] small static `s_blk`/`s_slot` arrays (about 160 B in the non-INT4 build, arithmetic only);[^dest-gqa-static] plain/`_short` LDS formulas;[^dest-plain][^dest-short] constants `BC = 64`, `BC_256 = 32`, `BR_PREFILL = 16`, `THREADS_PREFILL = 128`.[^dest-consts] The plain D=128 and `_short` byte counts match [fa-occupancy.md](fa-occupancy.md) (49364 / 45476). The plain D=256 count is now 60644 B, because the current formula pads K/V rows to `HEAD_DIM + 8`. That page's 59620 B is from an older snapshot without that pad (arithmetic only). The same page records a 1 KB LDS allocation granule and an LDS-bound plain D=128 prefill.[^wiki-fa-occ] The "blocks / WGP" column applies that granule against 128 KiB and ignores VGPRs.

Reading (arithmetic only):

- In llama.cpp terms the default dest prefill is a **16-column** tile (`ncols1 × ncols2 = BR_STEP × HEADS_PER_CTA = 16`). That is narrower than upstream's 32/64 choice for long Q, and the same class as the deleted caps. The difference is that dest has **no** runtime occupancy query and no abort path. The width is a compile-time design choice, not a workaround.
- LDS is not the limiter for the GQA kernel. A hypothetical 32-row `<128, 2, 16>` would need `2·16·128·2 + 16·136·4 + 2·16·16·4 = 18944 B`, still 6 CTAs/WGP by LDS. The open question is VGPRs: 16→32 rows doubles each thread's register-resident O/softmax rows. At 6 CTAs × 8 waves = 12 waves/SIMD, the VGPR budget is ≤ 80 per wave (alignDown(1024/12, 16)).[^wiki-vgpr] Fewer CTAs is the likely outcome. Whether more KV reuse beats lost waves is an empirical question for gfx1030 with packed `fdot2` (no WMMA).
- The dest comment records that 6 heads/CTA was measured slower than 2 heads/CTA at D=256.[^dest-gqa] That is the one in-tree data point on widening, and it points away from "wider is free".

## Sources

1. stew675/llama-cpp-rdna-boosts r334 commit `cca23942f10f660e898d49b99a2e22bda1b40f16` (message, files).
2. Same repo at `cca23942`: `WORKLOG.md` 261121.013, `INVARIANTS.md` r334 entry, `patches/0005-…` (RDNA2 occupancy estimator, RDNA2 tile table, new cap-deletion comment).
3. Same repo at parent `a28d4e699041e04fa10c5b5365ab2e1d39172032`: `patches/0005-…` (pre-r334 caps, `rdna_cap_ncols2`, estimator already present).
4. ggml-org/llama.cpp at `919794500475862f6768c353255469dc661cb734` (the fork's base): `fattn-tile.cuh`, `fattn-common.cuh`, `common.cuh`.
5. ROCm/rocm-systems develop `9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33`: `hip_platform.cpp` occupancy, `rocdevice.cpp`, `hip_device.cpp`; commit `b9ec5a21d702ebdbb681304a0b054acf7f1ab711` (#9571).
6. opengfx1030/vllm-rdna `rdna_extras` @ `d0b0c2b9af82d283141ba811f39b9293f1859014`: `csrc/rocm/fa_rdna2.cu`, `vllm/v1/attention/backends/rdna_attn.py` (read-only).
7. This wiki @ `77ace785`: vgpr-occupancy.md, wgp-cu-mode-occupancy.md, fa-occupancy.md; unmerged branch @ `edd25f7b`: split-kv-grid-occupancy.md.

[^r334-commit]: stew675/llama-cpp-rdna-boosts commit r334 (contributor PR #338; author date 2026-11-21T20:41:19Z). https://github.com/stew675/llama-cpp-rdna-boosts/commit/cca23942f10f660e898d49b99a2e22bda1b40f16
[^r334-worklog]: `WORKLOG.md` 261121.013, integration table rows for `0103` FA tile caps, `0102` RMS_NORM, `0102` GDN gate, `0104` MoE rows abreast (L23–L26) and the architecture-gate proof (L41), @ cca23942. https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/WORKLOG.md#L1-L58
[^r334-invariants]: `INVARIANTS.md` r334 entry, facts (1)–(3): RDNA2-only code is invisible to a gfx1201 build; narrow RMS_NORM bit-identity; "the RDNA2 FA tile caps were a workaround for `launch_fattn`'s `GGML_ASSERT(max_blocks_per_sm > 0)`" and still abort if `4*waves_per_simd < waves_per_block`, @ cca23942. https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/INVARIANTS.md#L1577-L1598
[^fork-est]: RDNA2 manual occupancy block in `launch_fattn` (`regsPerMultiprocessor = 32768` comment, `waves_per_simd = min(8, 1024 / vgprs)`, `vgprs` rounded to 8, `128*1024 / lds`, replaces HIP's figure only when larger), block-05 patch L1985–L2012 @ cca23942. https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L1985-L2012
[^fork-est-parent]: The same estimator block is already present at r334's parent, block-05 patch L1985–L2012 @ a28d4e69. https://github.com/stew675/llama-cpp-rdna-boosts/blob/a28d4e699041e04fa10c5b5365ab2e1d39172032/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L1985-L2012
[^fork-caps-parent]: Pre-r334 caps comment and gates (`rdna512`, `rdna_d128`, `rdna576`, `rdna256_mha`; "some wide-column kernels still fail the occupancy query and abort"; "D=128: cols_per_block 16/32/64 need 33-43 KB LDS"), block-05 patch L5470–L5530 @ a28d4e69. https://github.com/stew675/llama-cpp-rdna-boosts/blob/a28d4e699041e04fa10c5b5365ab2e1d39172032/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L5470-L5530
[^fork-ncols2-parent]: `const bool rdna_cap_ncols2 = GGML_CUDA_CC_IS_RDNA2(device_cc) && DKQ == 128;`, block-05 patch L5596 @ a28d4e69. https://github.com/stew675/llama-cpp-rdna-boosts/blob/a28d4e699041e04fa10c5b5365ab2e1d39172032/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L5596
[^fork-caps-new]: Replacement comment ("That assert is now unreachable on RDNA2 … RDNA2 takes the same 32/64-column prefill tiles every other arch takes"), block-05 patch L5470–L5477 @ cca23942. https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L5470-L5477
[^fork-table]: `ggml_cuda_fattn_tile_get_config_amd_rdna2` (all rows `occupancy = 1`; comment on the occupancy-1 history), block-05 patch L4711–L4745 @ cca23942. https://github.com/stew675/llama-cpp-rdna-boosts/blob/cca23942f10f660e898d49b99a2e22bda1b40f16/patches/0005-rdna-boosts-block-05-flash-attention-band-native-qua.patch#L4711-L4745
[^llama-macro]: `GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K)` with `nthreads <= 512`, `occupancy <= 8`, llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L10-L18
[^llama-rdna-table]: Upstream `ggml_cuda_fattn_tile_get_config_amd_rdna` D=128 rows (occupancy 8/8/8/3/3/3), llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L280-L285
[^llama-smem]: `flash_attn_tile` `cpy_ne`, `cpw`/`np`, and `__shared__ Q_tmp / KV_tmp / KQ` declarations (FAST_FP16 path), llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L870-L890
[^llama-select]: `launch_fattn_tile_switch_ncols1`: `nbytes_shared = 0`; `cols_per_block` 64/32/16/8/4/2 thresholds `Q->ne[1] > {32,16,8,4,2}/ncols2`, llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-tile.cuh#L1158-L1231
[^llama-assert]: `cudaOccupancyMaxActiveBlocksPerMultiprocessor(...)`; `GGML_ASSERT(max_blocks_per_sm > 0)`; `parallel_blocks = max_blocks_per_sm`, llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/fattn-common.cuh#L1197-L1201
[^llama-fp16]: `FP16_AVAILABLE` for `GGML_USE_HIP`, then `FAST_FP16_AVAILABLE`, llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/common.cuh#L280-L286
[^llama-cpy]: `ggml_cuda_get_max_cpy_bytes()` returns 16 on HIP, llama.cpp @ 91979450. https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/common.cuh#L424-L426
[^clr-occ]: `ihipOccupancyMaxActiveBlocksPerMultiprocessor`: `MaxWavesPerSimd` 16 for gfx10+, `VgprWaves = vgprsPerSimd_ / roundUp(usedVGPRs_, granule)`, `simdPerCU × 2` when `isWGPMode_`, `lds_pool_size = localMemSizePerCU_ × 2` in WGP mode, LDS rounded to `ldsAlignment()`, rocm-systems develop @ 9a92c8ba. https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/hipamd/src/hip_platform.cpp#L45-L122
[^clr-simd-wgp]: `if (settings().enableWgpMode_) { info_.simdPerCU_ *= 2; max_waves_per_cu *= 2; }`, rocm-systems @ 9a92c8ba. https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1728-L1731
[^clr-regs]: `availableRegistersPerCU_ = vgprsPerSimd_ * simdPerCU_ * wavefrontWidth_` (rocdevice.cpp L1804) → `deviceProps.regsPerMultiprocessor` (hip_device.cpp L742), rocm-systems @ 9a92c8ba. https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/rocclr/device/rocm/rocdevice.cpp#L1804 and https://github.com/ROCm/rocm-systems/blob/9a92c8ba111c1e3e1e4c226286ceb3ef7b86ce33/projects/clr/hipamd/src/hip_device.cpp#L742
[^clr-9571]: "fix(clr): Calculate LDS correctly in hip occupancy for gfx11 in WGP mode (#9571)", committed 2026-08-04T17:27:40Z; makes `sharedMemPerMultiprocessor` and the occupancy LDS pool per-WGP when `enableWgpMode_`. https://github.com/ROCm/rocm-systems/commit/b9ec5a21d702ebdbb681304a0b054acf7f1ab711
[^dest-consts]: `BC = 64`, `BC_256 = 32`, `MAX_SPLITS = 16`, `BR_PREFILL = 16`, `THREADS_PREFILL = 128`, `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L96-L100
[^dest-plain]: `fa_rdna2_prefill_paged_varlen` host: grid `(ceil(num_tokens/BR_PREFILL), H_q, num_seqs)`, D=128 and D=256 smem formulas, compile-time constants, no occupancy query, `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L3534-L3574
[^dest-short]: `fa_rdna2_prefill_paged_varlen_short` host: `BR_PREFILL = 32`, `BC = 32`, `THREADS = 256`, smem formula, `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L3778-L3790
[^dest-gqa]: GQA prefill kernel comment ("HEADS_PER_CTA=2, BR_STEP=8 … at D=256, measured faster than 6-heads/CTA and than the per-head varlen kernel"; "a CTA always holds 16 query rows"), `__launch_bounds__(256, 1)`, `BC = 16`, `THREADS = 256`, `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L3966-L4002
[^dest-gqa-static]: Static `__shared__ int s_blk[BC]`, `s_slot[BC]`, `uint64_t s_int4_data[2][IS_INT4 ? BC : 1]`, `s_int4_scales[…]` in the GQA prefill KV loop, `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L4104-L4110
[^dest-gqa-host]: GQA prefill host: `BC = 16`, `THREADS = 256`, `DSK = HEAD_DIM + 8`, grid `(ceil(num_tokens/BR_STEP), H_kv·groups, num_seqs)`, smem formula with "O and the softmax state live in registers", `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L4400-L4410 ; instantiation choice (`<D,4,4>` bf16 groups of four, `<D,2,8>` even, `<D,1,16>` odd): https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L4448-L4483
[^dest-mode]: `_gqa_mode()` returns `VLLM_FA_RDNA2_GQA_MODE`, default `"subgroup"`, `rdna_attn.py` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/vllm/v1/attention/backends/rdna_attn.py#L94-L95
[^dest-dispatch]: `_forward_prefill_kernels` dispatch order GQA → `_short` → split-K → plain, `rdna_attn.py` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/vllm/v1/attention/backends/rdna_attn.py#L814-L894
[^wiki-vgpr]: This wiki, vgpr-occupancy.md (wave32 1024-VGPR file per SIMD32, granule 16, addressable V0–V255, waves = min(16, floor(1024/aligned))) @ 77ace785. https://github.com/BlivionIaG/rdna-hip-wiki/blob/77ace785ab8cdf311300a7f23f45f58c8844590b/silicon/occupancy/vgpr-occupancy.md
[^wiki-wgp]: This wiki, wgp-cu-mode-occupancy.md (WGP vs CU LDS pool; occupancy API under-report note @ dest 1c1426d5) @ 77ace785. https://github.com/BlivionIaG/rdna-hip-wiki/blob/77ace785ab8cdf311300a7f23f45f58c8844590b/silicon/occupancy/wgp-cu-mode-occupancy.md
[^wiki-fa-occ]: This wiki, fa-occupancy.md (§2 short table LDS bytes; §5 "ISA LDS allocations are 1 KB granules"; §8 plain prefill_128 limiter = LDS) @ 77ace785. https://github.com/BlivionIaG/rdna-hip-wiki/blob/77ace785ab8cdf311300a7f23f45f58c8844590b/silicon/attention/fa-occupancy.md
[^wiki-splitkv]: This wiki, unmerged branch `idle/split-kv-grid-occupancy-2026-11-20`, split-kv-grid-occupancy.md @ edd25f7b. https://github.com/BlivionIaG/rdna-hip-wiki/blob/edd25f7bf039bba64075b6ddcb813f6b475b7305/silicon/attention/split-kv-grid-occupancy.md
[^dest-256-static]: `fa_prefill_paged_varlen_kernel_256`: `extern __shared__ smem_raw` plus static `__shared__ int s_blk[BC_256]`, `s_slot[BC_256]` (2 × 32 × 4 B = 256 B, arithmetic only), `fa_rdna2.cu` @ d0b0c2b9. https://github.com/opengfx1030/vllm-rdna/blob/d0b0c2b9af82d283141ba811f39b9293f1859014/csrc/rocm/fa_rdna2.cu#L1737-L1746
