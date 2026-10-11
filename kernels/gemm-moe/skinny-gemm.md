---
title: "Skinny GEMM — silicon / HIP contract"
sources:
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
---
# Skinny GEMM — silicon / HIP contract

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
| `vllm/envs.py` | `VLLM_RDNA_DENSE_GEMV` default off (force GEMV for A/B). [^f-3092d635-envs-py] |
| `rdna_all_reduce.py` | Integer device-index normalize for opt-in `rdna_ar` (still `VLLM_RDNA_AR=0` default; not production AR). [^f-3092d635-rdna-all-reduce-py][^f-3092d635-envs-py] |

No `__launch_bounds__` / `waves_per_eu` change on `wvSplitKrc_`. No DOT builtin, LDS-tile, KV-quant, or CMake gfx1030 list change. Occupancy leftover still FA + skinny `(1,1)`. Do not invent numbers.[^f-3092d635-skinny-gemms-cu]

## extras lock 2026-09-18 (tip `3b59ee16`, silicon commit `5c4ab989`) — LIVE decode dispatch

**Revert** of the PR #5 / PR #11 gfx1030 `wvSplitK` decode port. Verified on `.176` (4× V620): patched path device-asserts in `wvSplitK_hf_big_` (`skinny_gemms.hip:1168`, `assert(false)`) and dies during cudagraph capture; revert restores PASS 3/3.[^f-3b59ee16-skinny-gemms-cu][^fpr-5][^fpr-11]

| Surface | Delta |
|---|---|
| `vllm/.../layers/utils.py` | gfx1030 decode: **`gemv_f16_rdna2` for all FP16 M≤8**. No gfx1030 `wvSplitK` / LLMM1 path. Comment: wvSplitK/LLMM1 are gfx9/gfx11+ only and RDNA build is numerically wrong on gfx1030. [^f-3b59ee16-skinny-gemms-cu] |
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
