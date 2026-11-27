---
title: "Skinny GEMM — silicon / HIP contract"
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/2c6d5d450d0c74893ae8c10daee8352dac234e11/vllm/model_executor/layers/utils.py
  - https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/tests/kernels/quantization/test_rocm_skinny_gemms.py
  - https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/vllm/model_executor/layers/utils.py
  - https://github.com/opengfx1030/vllm-rdna/commit/4a971f82f549c35c2773762b75d56fed1dd932ad
  - https://github.com/opengfx1030/vllm-rdna/blob/3b59ee16e553168b1e3809c9a141088662171448/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/3b59ee16e553168b1e3809c9a141088662171448/csrc/rocm/fa_rdna2.cu
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/csrc/rocm/skinny_gemms_int4.cu
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/opengfx1030/vllm-rdna/commit/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4
  - https://github.com/opengfx1030/vllm-rdna/commit/59237b3deced668d22f2ed9afa84fde89c02fe4e
  - https://github.com/opengfx1030/vllm-rdna/commit/1c1dbee840cc953b647e02462f77aa5204644b61
  - https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/csrc/rocm/moe_q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/pull/11
  - https://github.com/opengfx1030/vllm-rdna/pull/5
  - https://github.com/opengfx1030/vllm-rdna/blob/3092d635295aa02f714832f539fa647be95db8f7/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/3092d635295aa02f714832f539fa647be95db8f7/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/3092d635295aa02f714832f539fa647be95db8f7/vllm/distributed/device_communicators/rdna_all_reduce.py
  - https://github.com/opengfx1030/vllm-rdna/blob/3b59ee16e553168b1e3809c9a141088662171448/vllm/platforms/rocm.py
  - https://github.com/opengfx1030/vllm-rdna/commit/3b59ee16e553168b1e3809c9a141088662171448
  - https://github.com/opengfx1030/vllm-rdna/commit/5c4ab9891910c4b7f3bde8621d1a58c8ad47c28b
  - https://github.com/opengfx1030/vllm-rdna/commit/e131562936f9c6f27c26449b00aaaeeb34159b4c
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/moe_resident_decode.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/ops.h
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/torch_bindings.cpp
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/moe_q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/mxfp4_dot2_common.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/commit/2bf779784d938e2981c56405f2586c3623e20b7e
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/skinny_gemms_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/skinny_gemms.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/skinny_gemms_wvsplitk.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/model_executor/layers/utils.py
  - https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/skinny_gemms_int4.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/53f157410b1ef2768ec59d74ac83a0ff4456b9cf
  - https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/csrc/rocm/skinny_gemms_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/71299d56b2526dc4df8c74d261d0c57c39865dd2/csrc/rocm/skinny_gemms_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/1155c85e4ffffc47ad83415b9c9b521cc1c496de
---
# Skinny GEMM — silicon / HIP contract

## Current state at dest `rdna_extras` @ `1dc3bac4` (checked 2026-10-11)

These are unchanged at the tip:
- `wvSplitKrc_` still carries `amdgpu_waves_per_eu(1, 1)`.
- `VLLM_ROCM_MOE_SKINNY` is still default on.
- *(Historical @ `1dc3bac4`; the variable was removed by the v0.34 merge `2bf77978`, see [Dest @ `00f106c5`](#dest--00f106c5-v034-merge-splits-the-skinny-build-2026-11-23))* `VLLM_RDNA_DENSE_GEMV` is still default off.
- Resident MoE (`VLLM_RDNA_MOE_RESIDENT` / `_SKINNY`) is still default off.
- `_rocm_C_skinny` (`VLLM_SKINNY_SRC`) is still built for all archs except gfx1250.[^sk-tip]


Live: `skinny_gemms.cu`. **Incomplete.** Same occupancy ticket as `fa_rdna2`.[^f-3b59ee16-skinny-gemms-cu][^f-3b59ee16-fa-rdna2-cu]

## gfx1030 path

`use_mfma` is `#if __HIP__MI3XX__` only. On gfx1030 it is **false** — scalar / packed FMA, not MFMA. The `mfma_f32_4x4x4bf16_1k` / `16x16x32` / FP8 MFMA sites are CDNA/GFX12 and must not run here.[^f-3e05abc9-skinny-gemms-cu]

`wvSplitKrc_` is `__attribute__((amdgpu_waves_per_eu(1, 1)))` — same trap as `fa_rdna2` `__launch_bounds__(*, 1)`. Other `wvSplitK_hf_*` kernels are `__launch_bounds__(WvPrGrp* THRDS)` with no min-waves (safer).[^f-3b59ee16-fa-rdna2-cu][^f-3b59ee16-skinny-gemms-cu]

Fix: drop `(1, 1)`, use `amdgpu_waves_per_eu(4, 8)` after the FA occupancy flip. Do not “enable MFMA” on gfx1030.[^f-3e05abc9-skinny-gemms-cu]

## MoE skinny int4 (recipe 0008) @ `5c3c0c6f`

- `skinny_gemms_int4.cu` gained wave-per-row W4A16 MoE skinny GEMV (LDS A tile, `__builtin_amdgcn_fdot2` / gfx10 path; LDS_SIZE 64 KiB).[^f-5c3c0c6f-skinny-gemms-int4-cu][^isa]
- `VLLM_ROCM_MOE_SKINNY` **default True** at tip `5c3c0c6f` (restored in `59237b3d`). Tip history: briefly flipped default-off (`1c1dbee8`, bad test), then restored — kernel was correct; unit-scale synthetic weights overflowed FP16.[^c-5c3c0c6f][^c-59237b3d][^c-1c1dbee8][^f-5c3c0c6f-envs-py]
- **Leave** RDNAHybridW4A16 + bf16 on gfx10 via skinny (Triton abort) — op rejects that combo cleanly; fp16 path stays.[^c-5c3c0c6f]
- Does not replace shuffled RDNA2 fused HIP MoE (`moe_gptq_gemm_rdna2`).[^f-5c3c0c6f-moe-q-gemm-rdna2-cu]
- Occupancy note: existing `wvSplitKrc_` `amdgpu_waves_per_eu(1,1)` trap still open; same ticket as FA.[^f-5c3c0c6f-skinny-gemms-cu]

## extras lock 2026-09-16 (tip `3092d635`, silicon commit `c350fa21`) — SUPERSEDED for dispatch

PR #11 / donor PR #5 (George Muravei-Alkhavoi / GeorgeMA-Strong). Dest: `opengfx1030/vllm-rdna` `rdna_extras`.[^fpr-11][^fpr-5]

| Surface | Delta |
|---|---|
| `csrc/rocm/skinny_gemms.cu` `wvSplitK` | Drop shared static `Rdna2PersistBuf` + `rdna2_persist_zeros`. Output is per-call `torch::empty({N,M}, …)`. Shared persist storage aliased live projection outs across later GEMV / graph nodes. [^f-3092d635-skinny-gemms-cu] |
| `vllm/.../layers/utils.py` | gfx1030 FP16/BF16 decode: qualified `wvSplitK` for **n = 1..5**. `gemv_f16_rdna2` for other gfx10x, for **n = 6..8** on gfx1030, and when `VLLM_RDNA_DENSE_GEMV=1`. New `on_gfx1030()` gate. [^f-3092d635-skinny-gemms-cu][^f-3092d635-envs-py] |
| `vllm/envs.py` | *(Historical @ `3092d635`; removed at `2bf77978`)* `VLLM_RDNA_DENSE_GEMV` default off (force GEMV for A/B). [^f-3092d635-envs-py] |
| `rdna_all_reduce.py` | Integer device-index normalize for opt-in `rdna_ar` (still `VLLM_RDNA_AR=0` default; not production AR). [^f-3092d635-rdna-all-reduce-py][^f-3092d635-envs-py] |

No `__launch_bounds__` / `waves_per_eu` change on `wvSplitKrc_`. No DOT builtin, LDS-tile, KV-quant, or CMake gfx1030 list change. Occupancy leftover still FA + skinny `(1,1)`. Do not invent numbers.[^f-3092d635-skinny-gemms-cu]

## extras lock 2026-09-18 (tip `3b59ee16`, silicon commit `5c4ab989`) — LIVE decode dispatch

**Revert** of the PR #5 / PR #11 gfx1030 `wvSplitK` decode port. Verified on `.176` (4× V620): patched path device-asserts in `wvSplitK_hf_big_` (`skinny_gemms.hip:1168`, `assert(false)`) and dies during cudagraph capture; revert restores PASS 3/3.[^f-3b59ee16-skinny-gemms-cu][^fpr-5][^fpr-11]

| Surface | Delta |
|---|---|
| `vllm/.../layers/utils.py` | gfx1030 decode: **`gemv_f16_rdna2` for all FP16 M≤8**. No gfx1030 `wvSplitK` / LLMM1 path. Comment: wvSplitK/LLMM1 are gfx9/gfx11+ only and RDNA build is numerically wrong on gfx1030. [^f-3b59ee16-skinny-gemms-cu] *(Historical @ `3b59ee16`: the LLMM1 "numerically wrong" part has a named cause and fix @ `a63a6b00`, see below; the dispatch itself is unchanged.)* |
| `vllm/platforms/rocm.py` | Remove `on_gfx1030()` (only used by the reverted port). `on_gfx10x()` stays. [^f-3b59ee16-rocm-py] |
| tests | Drop `test_gfx1030_decode_dispatch`. [^c-3b59ee16][^c-5c4ab989] |

PersistBuf ownership fix inside `wvSplitK` host (`torch::empty` per call) remains in tree for non-gfx1030 skinny users; **gfx1030 no longer dispatches into it for decode**. No DOT / LDS-tile / CMake gfx1030 list / `__launch_bounds__` change. Occupancy leftover still FA + skinny `(1,1)`. Do not invent numbers.[^f-3b59ee16-skinny-gemms-cu]

## extras lock 2026-09-24 (tip `e1315629`, PR #17) — resident MoE decode + MoE dequant fix

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. HIP/ISA only from the V620 baseline port merge.[^c-e1315629]

| Surface | Delta |
|---|---|
| `csrc/rocm/moe_resident_decode.cu` (**new**) | Native resident-layout W4A16 MoE skinny GEMV. Packed `int32` weights `[E,K/8,N]` after RDNA2 shuffle; scales fp16 `[E,K/gs,N]`. Inner loop `__builtin_amdgcn_fdot2`. Nibble dequant magic `0x64006400` / bias `0x64086408` with per-pair scales in `moe_resident_dequant_pair`. Symmetric uint4b8 only. [^f-e1315629-moe-resident-decode-cu] |
| same | WG = **128** (4× wave32). `w13` grid `(ceil(inter/32), topk, M)`; `w2` grid `(ceil(hidden/32), 1, M)`. Four waves split K and reduce FP32 partials in LDS: `gates[4][32]` + `ups[4][32]` (SiLU gate×up), `routed_sums[4][32]` (down). Host gate **M = 1..4**; `K%8==0`; `intermediate`/`hidden` `%32==0`. Template `GroupSize=128` specialized else runtime. [^c-e1315629] |
| `CMakeLists.txt` | Adds `moe_resident_decode.cu` to `VLLM_SKINNY_SRC` (`_rocm_C_skinny` object with `skinny_gemms*.cu`). Not a gfx1030 arch-list change. [^f-e1315629-CMakeLists-txt][^f-e1315629-moe-resident-decode-cu][^f-e1315629-skinny-gemms-cu] |
| `ops.h` / `torch_bindings.cpp` | Register `moe_resident_int4_decode`. Sibling `moe_skinny_int4_decode` (sequential Triton pack) unchanged. [^f-e1315629-ops-h][^f-e1315629-torch-bindings-cpp][^f-e1315629-moe-resident-decode-cu] |
| `moe_q_gemm_rdna2.cu` | **Take** scale-after-signed-INT4: local `refresh_moe_group` builds ZP with scale=`1.0`, then `__hmul2(dq, group_scales)` before `dot22_8_f`. Folding scale into the 1024/64 encoding rounded quantized zero to nonzero fp16. Same GPTQv1 `zero_offset=1` and `fdot2` path. [^f-e1315629-moe-q-gemm-rdna2-cu][^f-e1315629-mxfp4-dot2-common-cuh] |

**Dispatch (opt-in, not dest default):** `VLLM_RDNA_MOE_RESIDENT=0`, `VLLM_RDNA_MOE_RESIDENT_SKINNY=0`. When both armed + SILU + contiguous fp16 + shape gates, M≤4 fires resident skinny; else tiled `_rdna2_fused_moe` on the same resident pack.[^f-e1315629-envs-py]

No `__launch_bounds__` / `waves_per_eu` on the new kernels. No EXL3 DOT, FA, KV-quant, or CMake gfx1030 list change. Occupancy leftover still FA + skinny `(1,1)`. Leave tok/s and flipping resident defaults on. Do not invent numbers.[^c-e1315629]

## Dest @ `a63a6b00` — LLMM1 wave32 fold, wvSplitK refused on gfx10

Merged into `rdna_extras` as PR #54 (`port/faisal-skinny-wave32`, merge `4a971f82`); tip `a63a6b00`, previous watched tip `69e94731`.[^c-a63a6b00-pr54]

| Where | Lock |
|---|---|
| `LLGemm1_kernel` second-stage reduce | On wave32, LLMM1's K ≤ 8192 range can leave up to 32 partial waves in `red_smem`, but the 16-lane `__shfl_xor` tree only read partials 0..15. New guard `if (WARP_SIZE == 32 && qthreadid + 16 < num_warps) acc += red_smem[qwarpid][qthreadid + 16]` folds 16..31 in before the shuffle. [^f-a63a6b00-skinny-l275] |
| `wvSplitK` host | New `on_gfx10()` (matches `gcnArchName` containing `gfx10`) and `TORCH_CHECK(!on_gfx10(), "wvSplitK is not built for gfx10 (RDNA2); use gemv_f16_rdna2")`. Reason in the source comment: gfx10 is neither `__HIP__GFX9__` nor `__HIP__GFX1X__`, so the `wvSplitK` kernels compile to their `assert(false)` stub and the dispatch would fall into the GFX9 wave64 path (queue fault). A host error replaces the device assert recorded @ `3b59ee16`. [^f-a63a6b00-skinny-l74][^f-a63a6b00-skinny-l1239] |
| Tests | `test_rocm_skinny_gemms.py` adds LLMM1 shapes `(1, 4104, 1024)` and `(1, 4352, 1024)` (17 waves on wave32; per the test comment K=4352 fails without the fix) and skips the two `wvSplitK` tests on gfx10x. [^f-a63a6b00-test-skinny] |
| Dispatch | *(Historical @ `a63a6b00` for the LM head: @ `2c6d5d45` the LM head can take up to 32 rows, see below.)* Unchanged: `layers/utils.py` still routes gfx10x FP16 decode M ≤ 8 to `gemv_f16_rdna2`, and its comment still says wvSplitK/LLMM1 are gfx9/gfx11+ only. LLMM1 being correct on wave32 does not put it on the gfx1030 path. [^f-a63a6b00-utils] |

Occupancy: no `__launch_bounds__` / `waves_per_eu` change. `wvSplitKrc_` still carries `amdgpu_waves_per_eu(1, 1)` (now at L1369), so the FA + skinny `(1,1)` leftover is still open. No DOT, LDS-tile, KV-quant, CMake gfx1030 list, or `envs.py` change. [^f-a63a6b00-skinny-l1369]

## Dest @ `2c6d5d45` — LM head GEMV row cap

`rocm_unquantized_gemm_impl` gains `max_gemv_rows` (default 8): the gfx10x `gemv_f16_rdna2` branch now fires for `0 < n <= max_gemv_rows`. Only a `ParallelLMHead` layer passes more than 8, from `VLLM_RDNA_LM_HEAD_GEMV_MAX_M` clamped to [8, 32] (default 8 = previous behaviour); every other dense GEMM keeps 8. The source comment says `gemv_f16_rdna2` serves 1..32 rows. The `27b-hi2-tp2` recipe sets 32.[^f-2c6d5d45-utils][^f-2c6d5d45-recipe] No kernel, `__launch_bounds__` or `waves_per_eu` change in `skinny_gemms.cu`; `wvSplitKrc_` `(1, 1)` leftover unchanged.

## Dest @ `00f106c5`: v0.34 merge splits the skinny build (2026-11-23)

The upstream vLLM v0.34.0 merge `2bf77978` is in the range `1e1cac53` → `00f106c5`.[^c-2bf77978-sk]

| Where | Lock |
|---|---|
| Build split | The skinny GEMM launchers are now one translation unit per family (`skinny_gemms_llmm1.cu`, `_wvsplitk.cu`, `_wvsplitkrc.cu`, `_wvsplitkq.cu`), sharing kernels through `skinny_gemms.cuh`. The int4 skinny GEMM is split per `N_in` (`skinny_gemms_int4_api.cu` + `_n1`…`_n5`). All of them still go into `_rocm_C_skinny`, which is built for every arch except gfx1250.[^f-00f106c5-sk-cmake] |
| Fork gfx10 kernels | The fork-only `gemv_f16_rdna2_` / `gemv_i8_rdna2_` and the MoE decode GEMVs (`moe_w13_silu_gemv_`, `moe_w2_gemv_`) moved from `skinny_gemms_int4.cu` (at `1e1cac53`) to the new `skinny_gemms_rdna2.cu`. They are still `__launch_bounds__(WAVES * 32)`.[^f-00f106c5-sk-rdna2][^f-1e1cac53-sk-int4] |
| `wvSplitKrc_` occupancy | Still `__launch_bounds__(WvPrGrp * THRDS)` + `amdgpu_waves_per_eu(1, 1)`, now in `skinny_gemms.cuh` L1178-L1179, so the `(1,1)` leftover is unchanged.[^f-00f106c5-sk-cuh] |
| gfx10 refusal | `wvSplitK` still has `TORCH_CHECK(!on_gfx10(), "wvSplitK is not built for gfx10 (RDNA2); use gemv_f16_rdna2")`, now in `skinny_gemms_wvsplitk.cu`.[^f-00f106c5-sk-wvsplitk] |
| `VLLM_RDNA_DENSE_GEMV` | **Removed.** It now sits in `removed_environment_variables`, and setting it raises an error ("gfx1030 FP16 decode already uses gemv_f16_rdna2 unconditionally"). The gfx10x FP16 branch in `layers/utils.py` fires for `0 < n <= max_gemv_rows`, with no env opt-in.[^f-00f106c5-sk-envs][^f-00f106c5-sk-utils] |

No change to the gfx1030 arch list (`HIP_SUPPORTED_ARCHS` is the same line as at `1e1cac53`). No new DOT path in the skinny kernels.[^f-00f106c5-sk-cmake]

## Dest @ `53f15741`: `gemv_i8_rdna2` to M = 32 (2026-11-27)

| Item | State |
|---|---|
| M range | `gemv_i8_rdna2` accepts M 1..32. M 1..8 keep exact MT; 9..12, 13..16, 17..24 and 25..32 run MT = 12 / 16 / 24 / 32 with `GUARD = true`, which skips rows `m >= M` (wave-uniform, `M` is a kernel argument). *Historical @ `71299d56`:* `TORCH_CHECK(M <= 8)`, and M = 8 was the `default:` case.[^sk53-chk][^sk53-old][^sk53-tip] |
| Unroll vs VGPR | `constexpr int UNROLL = (MT > 8) ? 1 : 4`: above MT 8 each lane holds one `uint4` of weights per K step instead of four. Commit `1155c85e` says this frees 12 VGPRs/thread at MT 12..32 (author's figure).[^sk53-unroll][^sk53-c1155] *Historical @ `71299d56`:* fixed unroll 4.[^sk53-old] |
| Launch / LDS | Unchanged: `__launch_bounds__(WAVES * 32)`, 8/4/2/1 waves (256/128/64/32 threads) at N ≥ 1152/576/288/below, one wave per output row, no LDS, wave32 `__shfl_xor` reduce over offsets 16..1.[^sk53-launch] |
| ISA | Only `v_dot2c_f32_f16` (928 in the 5 instances compiled); int8 → fp16 by `__floats2half2_rn`, no `sdot4`.[^sk53-cc] |
| VGPRs (hipcc **5.7.1**, 8-wave) | MT 8: 110; MT 12/16/24/32: 59/69/95/59, no scratch. Not the dest 7.14 compiler.[^sk53-cc] |
| Dispatch | `rdna_ops._rdna_dense_gemm` sends int8-shadow decode with M ≤ `VLLM_RDNA_DECODE_I8_MAX` (default 32) here; it only exists with `VLLM_RDNA_DENSE_INT8=1`. Setting the env to 8 restores the old cap. A value above 32 fails the `M <= 32` check at run time (no clamp).[^sk53-ops][^sk53-chk] |

## Sources

[^f-3b59ee16-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`3b59ee16`](https://github.com/opengfx1030/vllm-rdna/blob/3b59ee16e553168b1e3809c9a141088662171448/csrc/rocm/skinny_gemms.cu).
[^f-3b59ee16-fa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/fa_rdna2.cu` @ [`3b59ee16`](https://github.com/opengfx1030/vllm-rdna/blob/3b59ee16e553168b1e3809c9a141088662171448/csrc/rocm/fa_rdna2.cu).
[^f-3e05abc9-skinny-gemms-cu]: BlivionIaG/vllm `csrc/rocm/skinny_gemms.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/skinny_gemms.cu).
[^f-5c3c0c6f-skinny-gemms-int4-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms_int4.cu` @ [`5c3c0c6f`](https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/csrc/rocm/skinny_gemms_int4.cu).
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^c-5c3c0c6f]: opengfx1030/vllm-rdna commit [`5c3c0c6f`](https://github.com/opengfx1030/vllm-rdna/commit/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4) (2026-09-14) — "docs(rdna2): 27B AWQ bench matrix (1k/512 + 16k/1k at c=1/4/8)".
[^c-59237b3d]: opengfx1030/vllm-rdna commit [`59237b3d`](https://github.com/opengfx1030/vllm-rdna/commit/59237b3deced668d22f2ed9afa84fde89c02fe4e) (2026-09-14) — "fix(rdna2): repair 0008 skinny default, gfx10 bf16 abort, warmup timeout".
[^c-1c1dbee8]: opengfx1030/vllm-rdna commit [`1c1dbee8`](https://github.com/opengfx1030/vllm-rdna/commit/1c1dbee840cc953b647e02462f77aa5204644b61) (2026-09-13) — "fix(rocm): disable VLLM_ROCM_MOE_SKINNY by default (recipe 0008 kernel is wrong)".
[^f-5c3c0c6f-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`5c3c0c6f`](https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/vllm/envs.py).
[^f-5c3c0c6f-moe-q-gemm-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/moe_q_gemm_rdna2.cu` @ [`5c3c0c6f`](https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/csrc/rocm/moe_q_gemm_rdna2.cu).
[^f-5c3c0c6f-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`5c3c0c6f`](https://github.com/opengfx1030/vllm-rdna/blob/5c3c0c6f53a3bffc20319de5621f1e560ffa4ab4/csrc/rocm/skinny_gemms.cu).
[^fpr-11]: opengfx1030/vllm-rdna PR #11 — Port unintegrated PR #5 wvSplitK + rdna_ar pieces (superseded). <https://github.com/opengfx1030/vllm-rdna/pull/11>
[^fpr-5]: opengfx1030/vllm-rdna PR #5 — Integrate Flash-Next FP16/CPU PLE and V620 performance fixes. <https://github.com/opengfx1030/vllm-rdna/pull/5>
[^f-3092d635-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`3092d635`](https://github.com/opengfx1030/vllm-rdna/blob/3092d635295aa02f714832f539fa647be95db8f7/csrc/rocm/skinny_gemms.cu).
[^f-3092d635-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`3092d635`](https://github.com/opengfx1030/vllm-rdna/blob/3092d635295aa02f714832f539fa647be95db8f7/vllm/envs.py).
[^f-3092d635-rdna-all-reduce-py]: opengfx1030/vllm-rdna `vllm/distributed/device_communicators/rdna_all_reduce.py` @ [`3092d635`](https://github.com/opengfx1030/vllm-rdna/blob/3092d635295aa02f714832f539fa647be95db8f7/vllm/distributed/device_communicators/rdna_all_reduce.py).
[^f-3b59ee16-rocm-py]: opengfx1030/vllm-rdna `vllm/platforms/rocm.py` @ [`3b59ee16`](https://github.com/opengfx1030/vllm-rdna/blob/3b59ee16e553168b1e3809c9a141088662171448/vllm/platforms/rocm.py).
[^c-3b59ee16]: opengfx1030/vllm-rdna commit [`3b59ee16`](https://github.com/opengfx1030/vllm-rdna/commit/3b59ee16e553168b1e3809c9a141088662171448) (2026-09-18) — "feat(rocm): opt-in leap T44b rdna_ar (VRAM flags + wedge check) (#13)".
[^c-5c4ab989]: opengfx1030/vllm-rdna commit [`5c4ab989`](https://github.com/opengfx1030/vllm-rdna/commit/5c4ab9891910c4b7f3bde8621d1a58c8ad47c28b) (2026-09-17) — "revert(rocm): drop the PR #5 wvSplitK gfx1030 decode port".
[^c-e1315629]: opengfx1030/vllm-rdna commit [`e1315629`](https://github.com/opengfx1030/vllm-rdna/commit/e131562936f9c6f27c26449b00aaaeeb34159b4c) (2026-09-23) — "Merge pull request #17 from GeorgeMA-Strong/codex/v620-baseline-port-20260922".
[^f-e1315629-moe-resident-decode-cu]: opengfx1030/vllm-rdna `csrc/rocm/moe_resident_decode.cu` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/moe_resident_decode.cu).
[^f-e1315629-CMakeLists-txt]: opengfx1030/vllm-rdna `CMakeLists.txt` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/CMakeLists.txt).
[^f-e1315629-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/skinny_gemms.cu).
[^f-e1315629-ops-h]: opengfx1030/vllm-rdna `csrc/rocm/ops.h` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/ops.h).
[^f-e1315629-torch-bindings-cpp]: opengfx1030/vllm-rdna `csrc/rocm/torch_bindings.cpp` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/torch_bindings.cpp).
[^f-e1315629-moe-q-gemm-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/moe_q_gemm_rdna2.cu` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/moe_q_gemm_rdna2.cu).
[^f-e1315629-mxfp4-dot2-common-cuh]: opengfx1030/vllm-rdna `csrc/rocm/mxfp4_dot2_common.cuh` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/mxfp4_dot2_common.cuh).
[^f-e1315629-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`e1315629`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/vllm/envs.py).
[^sk-tip]: `wvSplitKrc_` `amdgpu_waves_per_eu(1, 1)` at `csrc/rocm/skinny_gemms.cu` L1345; `VLLM_ROCM_MOE_SKINNY` default "1" (`vllm/envs.py` L1430), `VLLM_RDNA_DENSE_GEMV` default "0" (L1460), `VLLM_RDNA_MOE_RESIDENT` / `_SKINNY` default 0 (L1424-L1429); `VLLM_SKINNY_SRC` / `_rocm_C_skinny` at `CMakeLists.txt` L1491-L1511, dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/skinny_gemms.cu#L1345 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/envs.py#L1424-L1460 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1491-L1511
[^c-a63a6b00-pr54]: opengfx1030/vllm-rdna merge [`4a971f82`](https://github.com/opengfx1030/vllm-rdna/commit/4a971f82f549c35c2773762b75d56fed1dd932ad) (2026-11-12) — "Merge port/faisal-skinny-wave32: LLMM1 wave32 reduction fix and refuse wvSplitK on gfx10 (#54)"; branch tip [`a63a6b00`](https://github.com/opengfx1030/vllm-rdna/commit/a63a6b0010fd4e761fb14393e32d3491f4a1460b); compare from [`69e94731`](https://github.com/opengfx1030/vllm-rdna/compare/69e947313c49d2052d6cf9f8199b2eaa2441def9...a63a6b0010fd4e761fb14393e32d3491f4a1460b).
[^f-a63a6b00-skinny-l275]: `csrc/rocm/skinny_gemms.cu` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b` L268-L283. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu#L268-L283
[^f-a63a6b00-skinny-l74]: `csrc/rocm/skinny_gemms.cu` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b` `on_gfx10()` L72-L81. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu#L72-L81
[^f-a63a6b00-skinny-l1239]: `csrc/rocm/skinny_gemms.cu` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b` L1236-L1240. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu#L1236-L1240
[^f-a63a6b00-skinny-l1369]: `csrc/rocm/skinny_gemms.cu` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b` L1369. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/csrc/rocm/skinny_gemms.cu#L1369
[^f-a63a6b00-test-skinny]: `tests/kernels/quantization/test_rocm_skinny_gemms.py` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b`. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/tests/kernels/quantization/test_rocm_skinny_gemms.py
[^f-a63a6b00-utils]: `vllm/model_executor/layers/utils.py` @ `a63a6b0010fd4e761fb14393e32d3491f4a1460b` L361-L376. https://github.com/opengfx1030/vllm-rdna/blob/a63a6b0010fd4e761fb14393e32d3491f4a1460b/vllm/model_executor/layers/utils.py#L361-L376
[^f-2c6d5d45-utils]: opengfx1030/vllm-rdna `vllm/model_executor/layers/utils.py` [L33–51, 359–398](https://github.com/opengfx1030/vllm-rdna/blob/2c6d5d450d0c74893ae8c10daee8352dac234e11/vllm/model_executor/layers/utils.py#L33-L51) @ 2c6d5d450d0c74893ae8c10daee8352dac234e11.
[^f-2c6d5d45-recipe]: opengfx1030/vllm-rdna `tools/rdna/recipes/27b-hi2-tp2.env` [L114](https://github.com/opengfx1030/vllm-rdna/blob/2c6d5d450d0c74893ae8c10daee8352dac234e11/tools/rdna/recipes/27b-hi2-tp2.env#L114) @ 2c6d5d450d0c74893ae8c10daee8352dac234e11.
[^c-2bf77978-sk]: opengfx1030/vllm-rdna commit [`2bf77978`](https://github.com/opengfx1030/vllm-rdna/commit/2bf779784d938e2981c56405f2586c3623e20b7e) (2026-11-18 14:11 Paris): "Merge upstream vLLM v0.34.0 into RDNA integration branch".
[^f-00f106c5-sk-cmake]: `CMakeLists.txt` @ `00f106c5` L55, L1620-L1645, L1751-L1774. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/CMakeLists.txt#L1620-L1645>
[^f-00f106c5-sk-rdna2]: `csrc/rocm/skinny_gemms_rdna2.cu` @ `00f106c5` L45, L99, L225-L227, L380-L381. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/skinny_gemms_rdna2.cu#L225-L227>
[^f-1e1cac53-sk-int4]: `csrc/rocm/skinny_gemms_int4.cu` @ `1e1cac53` L1222 (`moe_w13_silu_gemv_`), L1401-L1402 (`gemv_f16_rdna2_`). <https://github.com/opengfx1030/vllm-rdna/blob/1e1cac5340a9c9af8c9df3db331978548d709b42/csrc/rocm/skinny_gemms_int4.cu#L1401-L1402>
[^f-00f106c5-sk-cuh]: `csrc/rocm/skinny_gemms.cuh` @ `00f106c5` L1176-L1180. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/skinny_gemms.cuh#L1176-L1180>
[^f-00f106c5-sk-wvsplitk]: `csrc/rocm/skinny_gemms_wvsplitk.cu` @ `00f106c5` L33-L34; `on_gfx10()` at `skinny_gemms.cuh` L92. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/csrc/rocm/skinny_gemms_wvsplitk.cu#L33-L34>
[^f-00f106c5-sk-envs]: `vllm/envs.py` @ `00f106c5` L729-L738. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/envs.py#L729-L738>
[^f-00f106c5-sk-utils]: `vllm/model_executor/layers/utils.py` @ `00f106c5` L486-L503. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/model_executor/layers/utils.py#L486-L503>
[^sk53-tip]: opengfx1030/vllm-rdna `rdna_extras` head [`53f15741`](https://github.com/opengfx1030/vllm-rdna/commit/53f157410b1ef2768ec59d74ac83a0ff4456b9cf), "Merge pull request #227 from wsantos/perf/rdna2-hip-decode", committed 2026-11-27 16:59 Europe/Paris; range [`71299d56...53f15741`](https://github.com/opengfx1030/vllm-rdna/compare/71299d56b2526dc4df8c74d261d0c57c39865dd2...53f157410b1ef2768ec59d74ac83a0ff4456b9cf) (7 commits: `bbdf2415`, `dc0b4d03`, `bb7adb0e`, `f6ef3eee`, `1155c85e`, `8f36c75d`, merge). Read 2026-11-27 ~17:50 Paris.
[^sk53-cc]: VGPR counts compiled on the box with **Debian hipcc 5.7.1 (clang 17.0.6)**, `--offload-arch=gfx1030 -O3`, `-Rpass-analysis=kernel-resource-usage`, kernels extracted standalone from the files at `53f15741` (and `71299d56` for the before column), 8-wave instances. This is **not** the dest ROCm 7.14 compiler; 7.14 was not available on the box. Treat the counts as indicative and re-check on 7.14.
[^sk53-chk]: `TORCH_CHECK(M >= 1 && M <= 32)` and the guarded MT switch @ `53f15741`: https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/csrc/rocm/skinny_gemms_rdna2.cu#L476 ; https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/csrc/rocm/skinny_gemms_rdna2.cu#L498-L533
[^sk53-unroll]: `UNROLL = (MT > 8) ? 1 : 4` @ `53f15741`: https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/csrc/rocm/skinny_gemms_rdna2.cu#L379-L452
[^sk53-launch]: `gemv_i8_rdna2_launch` waves by N @ `53f15741`: https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/csrc/rocm/skinny_gemms_rdna2.cu#L455-L472
[^sk53-old]: fixed unroll 4 and `M must be 1..8` @ `71299d56`: https://github.com/opengfx1030/vllm-rdna/blob/71299d56b2526dc4df8c74d261d0c57c39865dd2/csrc/rocm/skinny_gemms_rdna2.cu#L395 ; https://github.com/opengfx1030/vllm-rdna/blob/71299d56b2526dc4df8c74d261d0c57c39865dd2/csrc/rocm/skinny_gemms_rdna2.cu#L468
[^sk53-c1155]: commit [`1155c85e`](https://github.com/opengfx1030/vllm-rdna/commit/1155c85e4ffffc47ad83415b9c9b521cc1c496de) "perf(rdna2): use UNROLL=1 for MT > 8 in gemv_i8_rdna2 to cut VGPR pressure" (in `53f15741`).
[^sk53-ops]: `_DECODE_I8_MAX` and `_rdna_dense_gemm` @ `53f15741`: https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/vllm/model_executor/layers/rdna_ops.py#L31 ; https://github.com/opengfx1030/vllm-rdna/blob/53f157410b1ef2768ec59d74ac83a0ff4456b9cf/vllm/model_executor/layers/rdna_ops.py#L87-L100
