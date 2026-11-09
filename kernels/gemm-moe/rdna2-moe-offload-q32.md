---
title: "RDNA2 MoE expert offload and Q32 fixed-point accumulation (W4A16) — silicon lock"
updated: 2026-10-29
sources:
  - https://github.com/opengfx1030/vllm-rdna/compare/8015ea485b7f6c0298d8f98ce6e824ce1c444c45...b74028935393d1e21f73f80f65f506c9ea4639af
  - https://github.com/opengfx1030/vllm-rdna/commit/2436df23358ef8b9397076c0663707ec3dce69bf
  - https://github.com/opengfx1030/vllm-rdna/commit/68ec88ec215b0f53df02d04b71480c109b513215
  - https://github.com/opengfx1030/vllm-rdna/commit/bc85197571a7857a0651eb0d5db453f252778419
  - https://github.com/opengfx1030/vllm-rdna/commit/37360d8144edde25f5afe8dbb96801bf4a42d460
  - https://github.com/opengfx1030/vllm-rdna/commit/b74028935393d1e21f73f80f65f506c9ea4639af
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2_common.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_accum_rdna2.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2_prefill.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/w4a8_sdot4_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_expert_cache.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/fused_moe/rdna_expert_cache.py
---
# RDNA2 MoE expert offload and Q32 fixed-point accumulation (W4A16)

Read at dest `rdna_extras` tip `b7402893` (`b74028935393d1e21f73f80f65f506c9ea4639af`, merge of PR #147, read 2026-10-29). The range brought PR #146 "graph-safe RDNA2 MoE expert offloading" (`2436df23`, merged `68ec88ec`) and PR #147 (opt-in native expert-copy submission `bc851975`, slot-map update `37360d81`).[^c-2436][^c-bc85][^c-b740] Kernel tiles, `__launch_bounds__` and DOT are **unchanged**: the new work is a third epilogue mode (Q32 integer atomics) and a host-side copy planner. The MonoKernel K2 split for offload is on [../persistent/monokernel-gfx1030-dest.md](../persistent/monokernel-gfx1030-dest.md#update-at-dest-rdna_extras--b7402893-checked-2026-10-29). No tok/s here.

## Q32 fixed-point epilogue (new third mode)

| Item | State at `b7402893` |
|---|---|
| Shared helper | `fixed_accum_row` in `q_gemm_rdna2_common.cuh`: each of the 4 N-column partials is checked with `isfinite` and `fabsf(v) < fixed_limit`; a bad partial `atomicOr`s a sticky row-status word and is skipped; a good one becomes `__float2ll_rn(v · 2^32)` and goes through a 64-bit integer `atomicAdd`.[^common] |
| Layout | Caller-zeroed int64 `[rows, N+1]`: N fixed-point columns with 32 fractional bits plus one status column per row.[^accum][^moeq-layout] |
| Overflow bound | Dense: `fixed_limit = 2^30 / gridDim.z` (K-slices). MoE: `2^30 / (gridDim.z · max(1, output_topk))`, so the absolute sum of all partials that reach one cell stays below 2^30 before scaling.[^common][^moeq-limit] |
| Finalize | `moe_fixedpoint_to_half_kernel`, 256 threads, one thread per output element: int64 → `__ll2float_rn` → × 2^-32 → `__float2half_rn`; a row whose status word is non-zero is written as NaN.[^fin] |
| Where it plugs in | `moe_accum_row<C_T>` gains a `C_T == long long` branch (next to fp32 scratch atomics and packed-fp16 CAS).[^accum] `moe_gptq_gemm_rdna2` takes it when `c` is `torch::kLong` (fp16 activations only).[^moeq-host] The dense `gemm_q4_kernel_rdna2` and both prefill kernels (`gemm_static_kernel`, `gemm_dynamic_kernel`) pass an optional `long long* fixed` to `epilogue<M_TILE>`.[^qg][^qgp] |
| Tiles / occupancy | No change to block sizes, `M_COUNT` dispatch, prefill configs, LDS or `fdot2`; only the epilogue pointer is new.[^qg][^qgp] |

**Why integers:** the source comments say integer additions keep the result independent of K-slice scheduling and of expert-cache pass order, which is what lets the offload path split one MoE layer into several passes and round once.[^moeq-host][^moe-py]

## Gates

- **Dense deterministic W4A16** (`gptq_gemm_rdna2(…, deterministic)` / `gptq_gemm_rdna2_prefill(…, deterministic)`): opt-in `VLLM_RDNA2_W4A16_DETERMINISTIC_DENSE=1` (default `"0"`). It raises if W4A8 is on, and moves the `exllama` selection to the RDNA2 `prefill` kernel so every shape uses the guarded reduction.[^w4a16-py][^w4a16-sel] The host checks K % 32 == 0, N % 8 == 0, int32 contiguous packing; both need group size ≥ 32, a multiple of 32 for the decode entry and of 8 for the prefill entry.[^qg-host][^qgp-host] The output is `torch::empty` (not zeros) and the int64 scratch is per call, so it is graph-pool owned under capture.[^qg-host][^qgp-host]
- **W4A8:** `w4a8_sdot4_rdna2.cu` only updates its forward declaration of `gptq_gemm_rdna2_prefill` with `deterministic = false`; no sdot4 kernel change.[^w4a8]
- **MoE offload:** the routed path allocates one int64 `[tokens, hidden + 1]` accumulator, runs one `_rdna2_fused_moe` per planned cache pass into it, and finalizes once with `moe_fixedpoint_to_half_rdna2`. The W1 (gate/up) GEMM also uses an int64 `[total_tokens, N_gate_up + 1]` accumulator on this path. It refuses to run while expert work is being captured.[^moe-py][^moe-w1]

## Expert cache copy (host only, no kernel)

`rdna_expert_cache.cu` (new, in the base ROCm extension list, not the gfx1030-only list) plans copies of the six SoA payload tensors from pinned CPU rows into device slots.[^cmake][^ec]

- Refuses graph capture (`hipStreamIsCapturing`), and refuses any device whose `gcnArchName` does not start with `gfx103`.[^ec-guard]
- With HIP ≥ 7.14 it resolves `hipMemcpyBatchAsync` through `dlsym(RTLD_DEFAULT, …)` (no second HIP DSO); on failure it consumes the sticky error and does not retry a partially submitted batch. Otherwise it falls back to one `hipMemcpyAsync` per payload row.[^ec-batch]
- Opt-in: `expert_cache_copy` defaults to `"tensor"` (torch copies); `"native"` selects this op and needs a rebuilt extension. The cache itself needs `expert_cache_slots` under `additional_config.rocm_mono_decode`, `--monokernel`, Qwen4Exp, gfx10x, and symmetric INT4 FP16 compressed-tensors experts.[^ec-py-opt][^ec-py-native][^ec-py-gate]

## Supersedes

- [w4a16.md](../quant/w4a16.md) "MoE epilogue dual-mode (dest tip `30b0bd4e`)" listed two epilogues (packed-fp16 CAS default, fp32 scratch opt-in). *(Historical @ `30b0bd4e`.)* At `b7402893` there is a third, Q32 integer mode, used by the offload path and the deterministic dense opt-in; the default is unchanged.[^accum][^w4a16-py]

## Sources

[^c-2436]: opengfx1030/vllm-rdna commit [`2436df23`](https://github.com/opengfx1030/vllm-rdna/commit/2436df23358ef8b9397076c0663707ec3dce69bf) "[ROCm] Add graph-safe RDNA2 MoE expert offloading"; merged by [`68ec88ec`](https://github.com/opengfx1030/vllm-rdna/commit/68ec88ec215b0f53df02d04b71480c109b513215) (PR #146).
[^c-bc85]: Commits [`bc851975`](https://github.com/opengfx1030/vllm-rdna/commit/bc85197571a7857a0651eb0d5db453f252778419) "perf(rdna2): add opt-in native expert-cache copy submission" and [`37360d81`](https://github.com/opengfx1030/vllm-rdna/commit/37360d8144edde25f5afe8dbb96801bf4a42d460) "perf(rdna2): update pinned expert slot maps without scalar tensor dispatch".
[^c-b740]: Dest tip [`b7402893`](https://github.com/opengfx1030/vllm-rdna/commit/b74028935393d1e21f73f80f65f506c9ea4639af), "Merge pull request #147 from opengfx1030/codex/rdna2-moe-offload-improvements" (2026-10-29 01:23 Paris).
[^common]: [q_gemm_rdna2_common.cuh:133–171](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2_common.cuh#L133-L171) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^accum]: [moe_accum_rdna2.cuh:16–23, 53–71](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_accum_rdna2.cuh#L16-L71) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moeq-layout]: [moe_q_gemm_rdna2.cu:357–362](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_q_gemm_rdna2.cu#L357-L362) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moeq-limit]: [moe_q_gemm_rdna2.cu:245–261](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_q_gemm_rdna2.cu#L245-L261) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moeq-host]: [moe_q_gemm_rdna2.cu:417–440](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_q_gemm_rdna2.cu#L417-L440) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^fin]: [moe_q_gemm_rdna2.cu:487–526](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/moe_q_gemm_rdna2.cu#L487-L526) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^qg]: [q_gemm_rdna2.cu:78–84, 216–219, 236–250, 263–291](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2.cu#L236-L291) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^qgp]: [q_gemm_rdna2_prefill.cu:89–95, 209, 363, 396–436](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2_prefill.cu#L396-L436) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^qg-host]: [q_gemm_rdna2.cu:311–346, 383–409](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2.cu#L311-L409) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^qgp-host]: [q_gemm_rdna2_prefill.cu:581–617, 642–668](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/q_gemm_rdna2_prefill.cu#L581-L668) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^w4a8]: [w4a8_sdot4_rdna2.cu:49–53](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/w4a8_sdot4_rdna2.cu#L49-L53) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^w4a16-py]: [rdna2_w4a16.py:122–128](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py#L122-L128) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^w4a16-sel]: [rdna2_w4a16.py:151–157](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py#L151-L157) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moe-py]: [compressed_tensors_moe_wna16_rdna2.py:258–286](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py#L258-L286) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moe-w1]: [compressed_tensors_moe_wna16_rdna2.py:383–388, 425](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna2.py#L383-L425) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^cmake]: [CMakeLists.txt:1482](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/CMakeLists.txt#L1482) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ec]: [rdna_expert_cache.cu:1–3, 47–127](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_expert_cache.cu#L47-L127) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ec-guard]: [rdna_expert_cache.cu:59–67](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_expert_cache.cu#L59-L67) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ec-batch]: [rdna_expert_cache.cu:128–152](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_expert_cache.cu#L128-L152) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ec-py-opt]: [rdna_expert_cache.py:77](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/fused_moe/rdna_expert_cache.py#L77) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ec-py-native]: [rdna_expert_cache.py:133–148](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/fused_moe/rdna_expert_cache.py#L133-L148) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ec-py-gate]: [rdna_expert_cache.py:35–72](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/fused_moe/rdna_expert_cache.py#L35-L72) @ b74028935393d1e21f73f80f65f506c9ea4639af.
